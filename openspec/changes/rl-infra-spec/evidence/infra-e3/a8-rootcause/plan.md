# A8 G4 根因调查：计划与运行前登记（INFRA-E3，2026-10-01；不改 tasks.md 4.6 的勾选，plan-v6 的 G1–G6 与容差不变）

目的：判定 G4 差异（更新量 0.91%、sign 99.894%、exp_avg 0.30%）属于 (1) 正常跨 DP 数值差异、(2) 切换（重分片）实现缺陷、(3) 验收标准不合理。"bf16 精度低""loss 相同"不算根因，必须落到具体算子/路径并经干预验证。

## 0. 离线（CPU）已得事实（取回的 packed 状态；脚本 `offline_g4_layers.py`，输出 `offline_g4_layers.txt`，运行前已得）
1. **不经任何重分片/恢复，A1（DP1 从头）与 A2（DP2 从头）在步 2 的状态已有同一形态的差异**：exp_avg 总体相对 L2 0.26%，按层 27 层 1e-5、26 层 2.3e-4、25 层 7.7e-4、…、0 层 3.9e-3；exp_avg 逐位相等元素比例 27 层 99.7%、26 层 87.6%、…、0 层 46%。两条线从同一种子、同一冻结数据、同一初值出发，只有 DP 不同。⇒ 差异形态不是恢复/重分片造成的（现有数据能证明"存在"，不能证明"来源算子"）。
2. B1 对 A1 的步 3 有效梯度（由 exp_avg 反推，g=(m3-0.9·m2)/0.1）：按层相对 L2：层 27 为 2.6e-5，层 26 为 7.3e-4，层 25 为 2.3e-3，层 24 为 3.1e-3，层 20 为 5.2e-3，层 0 为 1.4e-2（总体 8.3e-3）；exp_avg 逐位相等元素比例由层 27 的 99.1% 单调降到层 0 的 1.3%。形态：顶端只有极小扰动，经 bf16 激活梯度链逐层放大（每次 bf16 舍入把 fp32 级扰动变成稀疏的 1-ulp 翻转）。因此 **"最后一层差异为 0"并不准确**：层 27 各张量的 6e-6~2e-4（总体 2.6e-5）与 fp32 重建噪声同阶，只能说"从层 27 就开始，且量级在 fp32 噪声附近"，无法判定首个分歧算子。
3. 梯度归约/缩放静态核对见 `../infra-e3/a8-run2/g4-analysis.md` §1–3：fp32 归约、loss 缩放因子全为 2 的幂（DP1：loss×16/16×1 再 /16；DP2：loss×8/16×2 再 /8），理论上逐微批反向应逐位相同；因此"理论逐位相同、实测不同"本身是待解释的矛盾，不能用"bf16 精度低"收尾。

## 1. 对现有对照设计的局限（任务 1 的回答）
- 现有 B1 对 A1：两者不仅 DP 不同（2 对 1），而且一个是"连续训练"、一个是"恢复后训练"；A1/A2 是各自从头的两条线。**不存在同形的"标准恢复"臂**，因此 B1-A1 的差异里"DP 配置"与"恢复路径（含重分片、以及恢复本身）"无法分开。
- 现有数据中唯一的无恢复跨 DP 对照是 A1 对 A2（见 0.1）：它说明跨 DP 数值差异本身就存在，但 A1/A2 的 s2 混合了步 1、步 2 两步，且 A2 与 A1 的 C1/C2 起点本身就不同，无法对"同一步、同一状态"做因果切分。
- 现有数据对 G3 只覆盖 DP2 恢复两次（B1 对 B1′），没有 DP1 的重复，也没有"同形恢复 vs 连续训练"。
- 近似的标准路径：E2 的 `restore_cut`（同形恢复，cut 来自同一 DP 形状；与 `restore_cut_resharded` 共用 cut 文件和导出/导入的命名状态，但不走重分片合并/切片），已在 E2 的 A6 上验证。
- **设计（同一状态 C、同一批冻结数据、步 3）**：C = rcA1 的 C1（DP1 步 2 之后）。
  - (a) `rcSa`：DP1 标准同形恢复 C1 → 训练步 3；
  - (b) `rcSb`：DP2 标准同形恢复 C1p → 训练步 3，其中 C1p = rcBc 在重分片恢复 C1 后立即 `save_cut`（DP2 形状的 cut，内容与 C1 逐位相同，G1 已证明）；
  - (c) `rcBc`：DP2 经 `restore_cut_resharded(C1)`（弹性切换路径）→ 步 3；
  - (d) `rcDd`：DP1 经 `restore_cut_resharded(C1p)`（反向切换）→ 步 3；
  - 连续参照 `rcA1`：DP1 从头训练到步 3（同时产出 C1）。
  - 局限：C1p 是经重分片产生的（没有"独立的 DP2 原生 cut 内容为 C1"的办法，因为 DP1 与 DP2 的训练本身会产生不同的 C）。(b) 的 DP2 加载走标准路径，但其输入文件来自 (c) 的导出；这足够把"加载路径"与"DP"分开：b 与 c 加载的是逐位相同的状态；若 b≠c，必是重分片加载引入的差异。

## 2. 运行 RC-1（GPU；写在运行前）
- 平台/卡：Modal Sandbox，`A10G:2`，运行前断言 `nvidia-smi` 名称为 NVIDIA A10G 且恰 2 张；Sandbox 无自动重试（等价 `--modal-retries 0`）；硬超时 5400 s；独立 watchdog 在 5700 s 按 app id `modal app stop`；app 名 `a8rc-rc1-20261001`。不经 yeto launcher、不碰 Nebius（避开 `_worker` 重建集群问题）。
- 代码：infra-e3 提交（运行时 SHA 写入证据）；Miles pin 与镜像同 plan-v6（e3a11ab3，`@sha256:2cc5cc52…`）；learner flags 同 A8（`--rl-deterministic-trainer`，dropout 0，GBS 16，mbs 1，DistOpt，bf16，`--accumulate-allreduce-grads-in-fp32`，`--no-gradient-accumulation-fusion`，`--attention-backend unfused`）；确定性环境同 A8（NCCL_ALGO=Ring、CUBLAS_WORKSPACE_CONFIG=:4096:8、TF32 关、NVTE_ALLOW_NONDETERMINISTIC_ALGO=0）。
- 阶段：dry → gen（8 个冻结 rollout，新生成；全部 arm 共用，故与 A8 的样本不同但自洽）→ rcA1、rcBc、rcSa、rcSb、rcDd（各只训练到步 3）→ pack。每个 arm 在步 3 装 `rc_trace` 并落盘。
- 费用：约 1 h × 2 × $1.10 = $2.2；估计上限 $4.0（含取回）；单次上限 $8；本任务总上限 $25。
- 追踪内容（`tools/probes/e3_reshard/rc_trace.py`，只读探针，CPU 测试 `tests/test_rl_e3_rc_trace.py`）：对每个训练微批（前向只读 log-prob 通道 no_grad 不计），每个子模块的前向输出、到达该输出的梯度（bwd_out）、模块自身反传产生的输入梯度（bwd_in）的 bit 校验和与 L2；每个可训练参数的逐微批 bf16 权重梯度（落盘，用于离线按任意顺序重做 fp32 累加）；微批 0、1 的层级输出/梯度全张量。

### 预先声明的判读（运行前写定，事后不改）
- **复现判据**：用 plan-v6 同口径从 packed 状态算 (a)(b) 的 `update_rel_l2` 与由 exp_avg 反推的梯度相对 L2。(b) 对 (a) 的梯度相对 L2 落在 [0.3%, 3%]（A8 的 0.83% 的 ÷3~×3）算"在 A10G 上复现同量级 G4 差异"；低于 0.3% 则不能用 A10G 做归因，后续用 `H100!:2`（需先登记）。
- **R1 切换实现是否额外引入差异**（任务 1）：
  - 逐位比较 s3 状态（adapter、FP32 master、exp_avg、exp_avg_sq）：`rcBc` vs `rcSb`（同为 DP2、加载同一状态、仅加载路径不同）；`rcDd` vs `rcSa`（同为 DP1）。**两对都逐位相同 ⇒ 弹性切换路径对该步没有额外引入任何差异（排除 (2)）**；任一对不同 ⇒ 记为切换实现的候选缺陷，量级与 (b)-(a) 对比，并停下追查加载路径。
  - `rcSa` vs `rcA1`（DP1 恢复 vs 连续）：不同 ⇒ 恢复本身（与重分片无关）改变数值，需单独解释。
  - `rcSb` vs `rcSa`：同一加载路径、不同 DP 的差异 = "DP 本身"的量；与 A8 的 G4 量级比较。
- **R2 首次分歧**：对 `rcSa`（DP1）与 `rcSb`（DP2）按样本 id 对齐微批（DP2 的微批 k→rank k%2，见 plan-v6 G2）。逐微批判定：根输入、各模块前向输出、各 bwd_out/bwd_in 校验和是否逐位相同；在反向执行序（trace 里按 hook 触发顺序，层 27→0）上找第一个不同的张量，其所属模块的反传即首个分歧算子（含 LoRA matmul、TE 融合 norm+linear、attention unfused bmm/softmax 等）。同时逐微批比较每个参数的 bf16 wgrad，并用保存的 wgrad 按 DP1 序与 DP2 序（rank 内顺序累加再求和乘 1/2）离线重做 fp32 累加，与步 3 实际梯度（由 exp_avg 反推）比较，检验"累加顺序"能否解释层 27 的残差。
  - 若全部微批的前向、反传张量、wgrad 都逐位相同：差异只可能来自微批间的累加/归约/Adam，需另设计干预；
  - 若 wgrad 逐微批已不同：以首个分歧张量定位算子，并按 R3 设计干预。
- **R3 干预与基线（RC-2，若 R2 定位后另行登记）**：只在登记新的 plan 节（含 arm、判读、费用）并提交后才运行。基线：同一 DP 下改变微批处理顺序得到的自然差异（用于评估 0.1%/99.9% 门槛）。

## 3. 汇总（回填）
运行完成后在 `rc1/` 与 `analysis.md` 回填；费用与 `modal app list` 核验写入 `/home/michael/work/infra-drafts/gpu-spend.md` 与 progress.md。

## 2a. RC-1 第 1 次运行（a8rc-rc1-20261001，ap-lgopmjalnNAjtWA87DQ88w，01:28:58–01:49:37Z，A10G:2，≤$0.8）：未得结果，探针缺陷
- dry、gen（9 min）通过；rcA1 在步 3 结束后 `rc_trace.dump_trace` 的 `bits.cpu()` 报 `CUDA error: invalid argument`，容器按设计 exit 7，pack 了 rcA1 的 s2/s3（无其余 arm），app 已 stop（`rc1a-failed/app_stopped.txt`）。训练步 1–3 在装了探针的情况下正常完成（探针不破坏训练）。
- 原因（提出）：探针把逐记录的校验和先留在 GPU 上，到 dump 时才 `.cpu()`；Miles 的 offload_train 在每步结束后释放了该步分配的 GPU 内存，dump 时这些张量已失效。
- 修复：`rc_trace._put` 在 hook 内当场转主机（提交见下），CPU 测试通过（`tests/test_rl_e3_rc_trace.py`）。判据、arm、判读规则、复现判据全部不变。
## 2b. RC-1 第 2 次运行（登记）
- app 名 `a8rc-rc1b-20261001`，其余同 §2（A10G:2、硬超时 5400 s、watchdog 5700 s、费用估计 ≈$2、上限 $4、单次上限 $8）。累计本任务 GPU 花费预估：0.8 + 2 = 2.8。
## 2c. RC-1 第 2 次尝试（a8rc-rc1b，ap-mlUCPjpzZynWTUsFMo47Jp，01:50:16–01:50:22Z，≈$0）：GPU 型号断言拒绝启动
- Modal 对 `A10G:2` 这次分配到 `NVIDIA A10`（driver 580.95.05；第 1 次是 `NVIDIA A10G`），按断言（只接受 A10G）容器在第一步 exit 3，未进入任何阶段（`rc1b-refused/`）。
- 处理：A10G 与 A10 都是 24 GB 的 GA102，a8rc 的全部 arm 在同一容器内、同一型号上运行，比较只在运行内部做；断言改为"两张卡都在 {A10G, A10} 内，且两张型号相同"，实际型号写入 gpus.txt。其余不变。若与 H100 比较量级需注意卡型号不同，结论里会如实写明。
- 第 3 次运行 app 名 `a8rc-rc1c-20261001`，预估 ≈$2，上限 $4；累计预估 0.8+2=2.8。
## 2d. RC-1 第 3 次尝试（a8rc-rc1c，ap-GtogZn2BdNWd2a64kNBuBQ，02:21:32 提交、02:44 起在 A10G:2 上运行，03:04:27Z 被停，≈$0.75）：被我自己遗留的 watchdog 误停，无结论
- 现象：app 在 03:04:27Z 被 stop（modal 显示操作者为本账号），此时 gen、rcA1（含探针 dump，rc=0）已完成、rcBc 进行中；容器被终止，没有 pack，未取回任何状态。`watchdog.log`：第 1 次运行遗留的 watchdog 于 03:03:53Z 触发。
- 原因：第 1 次运行起的 watchdog（sleep 5700 s）读的是共享路径 `a8rc/out/resources.txt`，而第 2、3 次运行把输出目录复用为同一路径；第 1 次的 watchdog 到点后按该文件里**第 3 次运行**的 app id 执行了 `modal app stop`。这是我的运行脚本缺陷，不是 yeto 或探针问题；没有影响任何其他 agent 的 app（只停了 a8rc- 前缀的我自己的 app）。
- 修复：每次运行用独立输出目录 `out-<app>`，watchdog 只读自己那次的 resources.txt，运行结束时主动杀掉自己的 watchdog；已确认当前无遗留 watchdog 进程。
- 第 4 次运行 app 名 `a8rc-rc1d-20261001`，其余（代码、arm、判读、复现判据）不变；预估 ≈$2，上限 $4；本任务累计预估：0.8+0+0.75+2 ≈ 3.6。

## 3. RC-1d 结果（A10G:2，app a8rc-rc1d-20261001，ap-q8Jlkv2iheP8IiKn495OPY，03:14–03:56Z，≈$1.55；原始证据 `rc1d-a10g/`，判读 `rc1d-a10g/RESULT_rc.json`）
按 §2 预先声明的规则：
- **R1（切换实现是否额外引入差异）**：rcBc 对 rcSb（DP2，同一状态 C1，仅加载路径不同）步 3 的 adapter/FP32 master/exp_avg/exp_avg_sq **逐位相同**；rcDd 对 rcSa（DP1）**逐位相同**；rcSa 对 rcA1（DP1 同形恢复 vs 连续训练）**逐位相同**；四个恢复 arm 恢复后的状态与 C1 逐位相同。⇒ 弹性切换路径（重分片加载，DP1→2 与 2→1）对一步训练没有引入任何比特差异，也没有额外的恢复效应；**未发现切换实现缺陷**（限于本卡、本批数据、一步）。
- **DP 本身的量（rcSb 对 rcSa）**：步 3 梯度（由 exp_avg 反推）相对 L2 4.5e-10，update 相对 L2 8.0e-10，sign 一致率 1.0，exp_avg 仅 285/10,092,544 个元素不同；逐样本前向校验和全部相同；微批 0、1 的 122 个层级张量（前向输出与反传梯度，DP2 的梯度乘 0.5 以抵消 loss 缩放）全部逐位相同；逐微批 wgrad 只有 4 个微批的 layer 27 共 6 个参数在 bf16 次正规数附近有差（相对差 <1e-300 量级）；用保存的 wgrad 按 DP1 顺序与 DP2 顺序重新做 fp32 累加，相对差 3.0e-10，与实测 4.5e-10 同量级。⇒ 在 A10G 上，同状态下 DP1 与 DP2 的差异**只有 fp32 累加结合律**这一项，约 1e-10 量级。
- **复现判据**：要求 (b) 对 (a) 的梯度相对 L2 在 [0.3%, 3%]；实测 4.5e-10，**未复现**。按预先声明："不能用 A10G 做归因，后续用 `H100!:2`（先登记）"。
- 重要含义：A8 的 0.83% 不是"DP1 对 DP2 的反传在任何卡上都不同"；在 A10G 上整条反传链是逐位相同的。A8 中 A1/A2 在步 2 已出现层 27→0 的放大形态，在 A10G 上是否也有，需 rcA2 臂（本次追加）核对；A8 的差异来自 H100 上的某个因素（卡/内核选择）或数据，需在 H100 上用同一设计复核。

## 4. RC-2：同设计在 `H100!:2` 上重跑（运行前登记；追加 rcA2 与步 1–2 记录）
- 目的：(i) 在 A8 的卡上用同一状态 C1 复现/否定 (b) 对 (a) 的差异；(ii) 若复现，定位首个分歧模块；(iii) rcA2（DP2 从头训练，无任何恢复）与 rcA1 在步 1、2、3 的校验和对比，把 A8 中"A1 对 A2 步 2 就有差异"的现象放到逐微批层面。
- arm 与顺序：rcA1（DP1 从头→C1，步 1–3，记录步 1–2 的校验和、步 3 全量）、rcA2（DP2 从头→C2，同）、rcBc、rcSa、rcSb、rcDd（同 §1，步 3 全量）。探针改动：梯度/wgrad 在记录前乘 1/dp（2 的幂，精确）以抵消 DP2 的 loss/8，使 DP1 与 DP2 的反传逐位可比；dump 记录 GPU 型号、SM 数、TE/torch 版本、确定性与 TF32 开关；步 1–2 只存校验和。
- 平台：Modal Sandbox `H100!:2`，断言两张都是 `NVIDIA H100 80GB HBM3`；硬超时 3600 s（单次上限 $8：2×$3.95/h×1 h = $7.9）；容器内另设"开始新 arm 的最迟时刻 2700 s"，之后跳过剩余 arm 并 pack/取回；独立 watchdog 3900 s 只读本次 resources；app 名 `a8rc-rc2-20261001`。预估 $5–6.5，上限 $7.9。累计（含 RC-1 前 3 次的 ≈$2.0 与 RC-1d 的 ≈$1.55）≈ $3.6 → 本次后 ≤ $11.5，总上限 $25。
- 判读（预先声明）：
  - 复现：rcSb 对 rcSa 的梯度相对 L2 ≥ 0.3% → 在 H100 上复现 A8 的量级；然后按 §2 R2 找首个分歧（第一个在反向执行序中位次最前的不同校验和）；
  - 未复现（<0.3%）：则同状态一步下 DP1/DP2 在 H100 上也等价，A8 的 0.83% 必须来自步 1–2 或数据；看 rcA2 对 rcA1 的步 1、2 记录与 s2 状态；
  - R1 判据同 §2：rcBc=rcSb、rcDd=rcSa 逐位相同 ⇒ 切换无额外差异；任一对不同 ⇒ 记为候选实现缺陷并停下追加载路径。

### 4a. RC-2 补充（主 agent 同意启动时的三点，同一次运行、无额外费用；运行前登记）
1. 若复现，用逐层追踪定位**第一个**分歧张量/模块。探针另记 kernel 选择信息：`rc_trace` 对 rcSa、rcSb 用 torch profiler 抓微批 0 前向+反传的 CUDA kernel 名称与次数（只在该微批内，异常不影响训练）；dump 记录 TE/flash-attn/Megatron/cuDNN/cuBLAS 版本与全部 `NVTE_*`/`CUBLAS*`/`NCCL_*`/`CUDA_*`/`TORCH_*`/`NVIDIA_*` 环境变量（`device` 字段）。能拿到的"算法 ID"即 kernel 名（含 split-K/stream-K 等变体）。
2. 汇报分开两项结论：(A) 切换相对标准路径是否逐位一致（rcBc/rcSb、rcDd/rcSa、rcSa/rcA1）；(B) DP1 对 DP2 差多少（rcSb/rcSa；rcA2/rcA1 的步 1–3）。
3. A8 原数据与 RC 数据是否相同：A8 的冻结样本在其容器内生成、未取回，RC 的冻结样本是新生成的（同 seed 1234、同提示、`--sglang-enable-deterministic-inference`，但不同容器/卡）。运行后以"H100 上 RC 的 rcA1 步 1 逐样本 loss（hex）对 A8 的 A1 步 1"逐位比较判定：全部相同 ⇒ 数据相同；不同 ⇒ 数据不同，写明。A10G（RC-1d）的 loss 因卡不同不可比。

## 5. RC-2 结果（H100!:2，app a8rc-rc2-20261001，ap-lxoc2DogJS48ic6Bw2xENq，04:03:36–≈04:40Z，≈$4.9；证据 `rc2-h100/`，判读 `rc2-h100/RESULT_rc.json`、`device_and_kernels.json`、`data_identity.txt`）
- **(A) 切换相对标准路径**：rcBc 对 rcSb（DP2）、rcDd 对 rcSa（DP1）、rcSa 对 rcA1（DP1 同形恢复 vs 连续）步 3 全状态**逐位相同**；四个恢复 arm 恢复后状态与 C1 逐位相同；逐微批的前向/反传校验和与 wgrad 同样逐位相同。⇒ 按 §2 R1 判据，**弹性切换（重分片加载，DP1→2 与 2→1）没有额外引入任何差异**（限本卡、一步、本批数据）。A10G 同。
- **(B) DP1 对 DP2**（rcSb 对 rcSa，同一 C1）：梯度相对 L2 8.29e-3、update 相对 L2 9.13e-3、sign 一致率 99.85%、exp_avg 相对 2.96e-3——与 A8 的 G4（0.91%、99.894%、0.30%）一致，**复现**（§4 判读"复现"分支）。数据同 A8：A8 的 A1 与 RC 的 rcA1 步 1、2、3 逐样本 loss（hex）全部逐位相同（`data_identity.txt`），故步 1–2 数据与 A8 相同（A10G 上步 1 也相同）。
- **首个分歧**：16 个样本里 8 个（32–39，一个 group）优势为 0、loss 与梯度恒为 0；其余 8 个（40–47）：DP2 的 rank0 上的样本（40、42、44、46）与 DP1 逐位相同（层级前向/反传张量、全部 wgrad；仅 layer 27 个别次正规数处的 2^±1 缩放伪差），**rank1 上的样本（41、43、45、47）全部不同**。差异最早出现在整网反传的起点——logits 的梯度（`module.module|bwd_out`，[1,T,151936]）——此前前向 logits、loss 逐位相同；随后逐层放大（wgrad 相对差：layer 27 约 1e-5，26 约 9e-4，25 约 3e-3，20 约 6e-3，13 约 1e-2，0 约 1.8e-2；逐位相等元素比例 100%→10%）。与 A8 步 3 的逐层形态相同。
- kernel：rcSa、rcSb（rank0、rank1）微批 0 的 CUDA kernel 名称多重集合完全一致（131/132 个名称，差异仅为探针自己的 `×bwd_scale` 逐元素乘），GEMM/attention kernel 选择与 rank 无关（至少对微批 0 这一优势为 0 的样本）。环境：H100 80GB HBM3、132 SM、TE 2.17.0、flash_attn 2.7.4（`NVTE_FLASH_ATTN=0`、`NVTE_FUSED_ATTN=0`、`NVTE_UNFUSED_ATTN=1`）、cuDNN 9.22、torch 2.13.0+cu130、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVIDIA_TF32_OVERRIDE=0`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`、`NCCL_ALGO=Ring`、`torch.use_deterministic_algorithms(True)`、`allow_bf16_reduced_precision_reduction=True`。
- 含义：此前"逐样本反传在 DP1 与 DP2 下普遍不同、bf16 精度所致"的结论被推翻：同样本在 rank0 与 DP1 逐位相同；差异只来自 rank1 这条计算，且起点在 loss→logits 梯度这一步的输入或该步算子；逐层放大只是传播。

## 6. RC-3：判别 "输入 / rank1 进程 / 物理 GPU"（运行前登记）
- 候选：(H-input) rank1 喂给 loss 的输入不同（old_log_probs、优势、缩放、loss_mask 等；old_log_probs 来自 no_grad 的 forward-only 通道）；(H-gpu) 物理 GPU 1 上该算子数值不同；(H-rank) rank1 进程相关的状态。
- 探针追加（只读，CPU 测试 `tests/test_rl_e3_rc_trace.py`）：记录 forward-only（old log-prob）通道的层级/顶层输出校验和；每个训练微批的 loss 输入（`log_probs`=old、`advantages`、`loss_masks`、长度、`rewards`）与 loss 返回的 metrics（`pg_loss`、`ppo_kl`、`pg_clipfrac` 等）、标量 loss；logits 前向与 logits 梯度的按 token 行校验和及行 L1；DP1 的微批 9 与 DP2 的本地微批 4（即样本 41）的完整 logits 梯度张量（fp32 [1,T,151936]）供逐元素比较；设备 UUID/索引与 `CUDA_VISIBLE_DEVICES`。
- arm（容器内顺序；`H100!:2`，同 A8 的确定性环境）：`dA1`（DP1 从头→C1，步 3 深探针）、`dBc`（DP2，重分片恢复 C1，保存 C1p，步 3 深探针）；然后容器内 `ray stop`，以 `CUDA_VISIBLE_DEVICES=1,0` 重启 Ray（逻辑 GPU 0 变为物理 GPU 1）并用 `gpu_map.py` 打印映射，再跑 `dSaX`（DP1 标准恢复 C1，现在落在物理 GPU 1）与 `dSbX`（DP2 标准恢复 C1p，rank0/rank1 与物理 GPU 的对应互换）。
- 预先声明的判读：
  1. **复现**：dBc 的样本 40/42/44/46 与 dA1 逐位相同、41/43/45/47 不同（以 wgrad 与层级校验和判）；否则记录并停止归因。
  2. **H-input**：对不同的样本，若 old_log_probs、优势、loss_mask、rewards、logits 前向行校验和中任一项在 dA1 与 dBc 之间不同 ⇒ 来源是喂给 rank1 的输入；用 forward-only 通道的逐模块校验和定位 old_log_probs 的首个分歧（或数据路径）；之后追到负责代码，按实现缺陷处理并修。
  3. 若所有 loss 输入与 metrics 逐位相同、logits 梯度行校验和仍不同：用保存的 logits 梯度张量逐元素比较，判定差异落在哪些行/列（目标 token 列、响应行、低概率列）及量级；这是 loss 反传算子本身在 rank1 上的数值差异。
  4. **H-gpu vs H-rank**：dSaX 对 dA1 逐位相同 ⇒ DP1 在物理 GPU 1 上与 GPU 0 一致；不同 ⇒ 物理 GPU 1 的数值不同。dSbX 对 dBc 逐位相同且 rank1 的样本仍与 DP1 不同 ⇒ 差异跟 rank/进程/样本走，不跟物理 GPU 走；若互换后 rank0 的样本变为不同而 rank1 的变为相同 ⇒ 差异跟物理 GPU 走。（主 agent 建议的"对调样本到 rank 的分派"：分派在 rollout 侧调度，本批不改；如需要，作为后续单独登记。）
- 费用与上限：Modal `H100!:2`，硬超时 2700 s、新 arm 最迟 1800 s、watchdog 3000 s；预估 $4–5，上限 $5.9；累计已花 ≈$8.0，本次后 ≤$13.9，总上限 $25。app 名 `a8rc-rc3-20261001`。

## 7. RC-3 结果（H100!:2，app a8rc-rc3-20261001，ap-oIESN9NqLdg8NlstMLu6FJ，04:51:35–≈05:20Z，≈$3.8；证据 `rc3-h100/`，判读 `rc3-h100/RESULT_deep.json`、`cross_run_bitwise.txt`）
- **G4 差异没有出现**：dA1 对 dBc（DP1 对 DP2，同一 C1）梯度相对 L2 6.3e-10、update 4.4e-10（仅 255/1000万 exp_avg 元素不同）；8 个有梯度的样本在 DP1、DP2（含 rank1）上的 loss 输入（old log-probs、优势、mask、长度、rewards）、loss、metrics、logits 前向行校验和、logits 梯度行校验和、forward-only 通道记录、训练前向/反传记录、wgrad 全部逐位相同（wgrad 只有 layer 27 个别次正规数伪差）。
- 交换物理 GPU 后（`gpus_swapped.txt`：Ray 逻辑 GPU0=物理 UUID 5e67…）：dSaX 对 dA1、dSbX 对 dBc 步 3 状态**逐位相同**，rank 与物理 GPU 的对应互换对结果无影响。
- **跨运行比特比较（`cross_run_bitwise.txt`）**：A8 的 A1_s2/A1_s3 与 RC-2 的 rcA1、RC-3 的 dA1 逐位相同（DP1 完全稳定）；A8 的 B1_s3（DP2，从 C1 恢复）与 RC-2 的 rcBc_s3 **逐位相同**（两个独立容器、不同物理 GPU），但与 RC-3 的 dBc_s3 不同。⇒ DP2 的"坏模式"（0.83%）是确定性的、可跨容器复现的，不是硬件抽签；RC-3 的 DP2 处于"好模式"（与 DP1 一致到 1e-10）。
- RC-3 与 A8/RC-2 的区别（仅有这些）：(a) RC-3 的探针更重（forward-only 通道钩子、loss 包装与 logits 钩子、D2H 拷贝）；(b) RC-3 的 DP2 恢复 arm 之前没有跑过"DP2 从头训练"arm（A8：A1、A2、B1…；RC-2：rcA1、rcA2、rcBc…；A10G 的 RC-1d：A1、Bc… 也没有）；(c) 物理 GPU 实例不同（但 RC-2 与 A8 也不同却结果逐位相同）。
- 解释 (b)：每个 arm 是新的 learner 进程、新的 Ray actor（`MilesBackend.start_arm` 每次 `create_rollout_components`+`create_training_models`，`stop_arm` 经 `Disposer` 释放；`miles_backend.py`）；跨 arm 共享的只有：同一 Ray 集群（GCS/raylet/对象存储）、文件系统（cuts、frozen、各类 JIT/编译缓存）、GPU 与驱动。主 agent 提出的"状态从上一个 arm 泄漏到下一个"假设在此框架下对应：Ray 集群/文件缓存/GPU 残留；本批用对照检验。

## 8. RC-4：二分"坏模式"的触发因素（运行前登记；总费用上限已由主 agent 放宽到 $30）
- 目的：区分 (i) 探针（RC-3 的重探针把坏模式盖住了）、(ii) 此前跑过 DP2 从头训练 arm（跨 arm 状态）、(iii) Ray 集群级状态、(iv) stream/时序竞争。**只比较状态**（探针极少），每个 DP2 arm 都从 C1 重分片恢复、训练步 3，对 DP1 参照 eA1 比较。
- arm 与进程语义（全部是新 learner 进程、新 Ray actor；只有"Ray"一列不同）：
  | arm | DP | 探针 | 之前是否跑过 DP2 从头训练（eA2） | Ray 集群 |
  |---|---|---|---|---|
  | eA1 | 1 从头→C1，步 1–3 | 无 | 否 | 容器启动时的集群 |
  | eP0 | 2，恢复 C1 | 无 | 否 | 同集群 |
  | eA2 | 2 从头→C2，步 1–3 | 无 | — | 同集群 |
  | eP0b | 2，恢复 C1 | 无 | 是 | 同集群（eA2 用过） |
  | eP3 | 2，恢复 C1 | 仅 loss 包装（RC-3 探针中最可能扰动者：包装 `get_loss_function`、logits 钩子、行校验和 `.cpu()`；不装模型钩子、不存 wgrad、不存大张量） | 是 | 同集群 |
  | eP0r | 2，恢复 C1 | 无 | 是 | **`ray stop`+`ray start` 之后的新集群**（文件系统缓存与 GPU 仍同） |
  | eCLB | 2，恢复 C1 | 无 | 是 | 再次重启 Ray，并 `CUDA_LAUNCH_BLOCKING=1`（所有 CUDA 调用同步，排除 stream/时序竞争） |
- 预先声明的判读（"坏"=相对 DP1 梯度相对 L2 ≥ 3e-3；"好"=≤ 1e-6；同时报告是否与 A8 的 B1_s3（坏模式参照）或 RC-3 的 dBc_s3（好模式参照）逐位相同）：
  1. eP0 坏 ⇒ 不需要先跑 eA2；RC-3 的好模式由探针造成（再看 eP3 是否仍坏）。eP0 好、eP0b 坏 ⇒ **跨 arm 状态**（eA2 之后才出现）：再看 eP0r：坏 ⇒ 状态在集群之外（文件缓存/GPU/驱动）；好 ⇒ 状态在 Ray 集群内（对象存储/GCS/预热 worker 等）。eP0、eP0b 都好 ⇒ A8/RC-2 的坏模式不可由本批复现，转为排查"同一次运行内 A2 之前的 arm 序列"（本批不覆盖，另议）。
  2. eP3 好、eP0b 坏 ⇒ loss 包装（RC-3 的探针）扰动了坏模式，坏模式依赖 loss 路径的内存/时序细节；eP3 坏 ⇒ 探针不是掩盖因素。
  3. eCLB 好、eP0r 坏 ⇒ stream/时序竞争（同步化后消失）；eCLB 也坏 ⇒ 不是 launch 时序，是确定性的内存布局/分配器历史/缓存状态。
- 如确认是泄漏：再追到具体状态（缓存键、buffer、分配器），修复并加回归测试（另行登记）。
- 平台与费用：Modal `H100!:2`（断言 H100 80GB HBM3）；硬超时 3000 s，新 arm 最迟 2100 s，watchdog 3300 s；预估 $5–5.5，上限 $6.6；累计已花 ≈$11.8，本次后 ≤$18.4，总上限 $30。app 名 `a8rc-rc4-20261001`。

## 8a. RC-4 结果（H100!:2，app a8rc-rc4-20261001，ap-KW2kKkdQLJ5MgyvfrlUMF6，≈05:28–06:06Z，≈$4.7；证据 `rc4-h100/`，判读 `rc4-h100/RESULT_bisect.json`，`gate_reference_check.txt`）
- 六个 DP2 arm（eP0、eP0b、eP3、eP0r、eCLB，及 eA2 的恢复前提）**全部处于"好模式"**：相对 DP1 的梯度相对 L2 都是 6.2e-10，步 3 状态与 RC-3 的 dBc_s3 逐位相同，与 A8 的 B1_s3 不同。
- 按预先声明的判读：eP0（无探针、此前无 eA2）好 ⇒ RC-3 的好模式不是重探针造成的；eP0b（A2 之后）好 ⇒ 不需要"先跑过 DP2 从头训练"；eP3 好；eP0r（Ray 集群重启后）好；eCLB（CUDA_LAUNCH_BLOCKING=1）好。⇒ **探针、arm 顺序、Ray 集群状态、launch 时序四个因素都不是坏模式的触发条件**；本批没有复现坏模式。
- 跨容器事实汇总（同一镜像、同一代码、同一数据、同一 C1；DP1 在所有容器里逐位相同）：坏模式 = A8（09-30 11:45Z）、RC-2（10-01 04:03Z），二者 DP2 结果逐位相同；好模式 = RC-1d（A10G）、RC-3（04:51Z）、RC-4（05:30Z）。软件环境逐项相同（环境变量、TE/torch/cuDNN 版本、驱动 580.95.05、132 SM）。差别只剩**物理主机/GPU 实例**（及时间）。"坏模式"是确定性的（两个坏容器的 DP2 状态逐位相同、gate 指标 2.647e-3 一致），按容器二分，像是主机/硬件类别的属性，而不是随机竞争。
- 有一个便宜的容器级判别量：DP2 从头训练 2 步的 exp_avg 相对 DP1（A8 的 A1_s2，逐位稳定）：坏容器 2.647e-3（A8 与 RC-2 完全一致），好容器 3.8e-10（`gate_reference_check.txt`）。第 1 步各样本优势全为 0（loss 与梯度恒为 0），第 2 步样本 16–23 有梯度、权重仍是初值，故不需要任何恢复就能做同权重的 DP1/DP2 比较。

## 9. RC-5：抽到"坏主机"再诊断（运行前登记；总费用上限 $30）
- 思路：坏模式按容器二分，且单靠重复 arm 无法在同一个好容器里复现；在新容器里先花最少的钱判别是否为坏主机，是才继续深诊断。
- 每次尝试（一个新 Modal 容器，`H100!:2`，同 A8 的确定性环境与代码；profile `a8rc-h100d`）：
  1. GPU 型号断言；记录主机信息（`nvidia-smi -q`、`topo -m`、`lscpu`）；**GPU 一致性电池**（`gpu_consistency.py`，Ray 之前，缓存目录隔离不污染训练的 Inductor/Triton 缓存）：用相同输入在 GPU0 与 GPU1 上各运行 3 次，逐位比较设备间与重复间：lm_head 前向/dgrad(K=151936)/wgrad、MLP 与 LoRA 矩阵乘、bmm+softmax（注意力形态）、词表 softmax/log_softmax、64M 元素 fp32 求和、Megatron `jit_fuser`(torch.compile) 融合交叉熵的前向 loss 与反向 logits 梯度；
  2. dry、gen（8 个冻结 rollout；若上次已取回则以 `E3_FROZEN_DIR` 上传并跳过 gen）；
  3. **gate 臂 gA2**：DP2 从头训练 2 步，打包 s2，容器内与上传的参照（A8 的 A1_s2，DP1，所有容器逐位相同）比 exp_avg 相对 L2；`> 1e-5` ⇒ 坏主机继续（预期坏 2.6e-3、好 3.8e-10）；否则 `GATE good host` 并停止（pack 后退出）；
  4. 坏主机才继续：gA1d（DP1 从头 2 步）与 gA2d（DP2 从头 2 步），两者步 2 做 RC-3 同款深探针（forward-only 通道、loss 输入与 metrics、logits 前向/梯度行校验和、样本 16、17、21 的完整 logits 梯度张量、逐层校验和与 wgrad）。步 2 权重仍是初值，DP1 与 DP2 同权重可比。
- 预先声明的判读：
  - 电池在坏主机上出现 GPU0 与 GPU1 的设备间差异 ⇒ 根因落在 GPU/驱动层面的算子数值（列出算子、元素数、量级）；这不是 DP 数学，也不是切换实现；
  - 电池干净而 gate 坏 ⇒ 进程/配置级：用深探针按 RC-3 的规则定位首个分歧（loss 输入 / logits 梯度行 / 完整张量逐元素）；
  - 好主机 ⇒ 不再继续该容器；每次尝试前先向主 agent 报告并记账；**上限：最多 3 次尝试，或累计 GPU 花费达 $26 即停**，余量留给修复验证。
- 费用：好主机尝试 ≈ 电池 2 min + gen 7 min + gA2 4 min + 启动/打包 4 min ≈ 17 min ≈ $2.2（有冻结数据上传时 ≈ $1.3）；坏主机再加 gA1d、gA2d、取回约 +14 min ≈ $1.9。硬超时 2700 s（上限 $5.9）；累计已花 ≈$16.5。app 名 `a8rc-rc5a-20261001`。

## 9a. RC-5 第 1 次尝试结果（H100!:2，app a8rc-rc5a-20261001，ap-Ye1IVcnVz2Bs7aTJNLljKo，≈06:12–06:39Z，≈$3.6；证据 `rc5a-h100/`）
- 电池（GPU0 对 GPU1）：12 个算子（lm_head 前向/dgrad/wgrad、MLP/LoRA 矩阵乘、bmm+softmax、词表 softmax/log_softmax、64M 求和、compiled 融合交叉熵前向 loss 与反向 logits 梯度）设备间与重复间全部逐位相同（`rc5a-h100/gate_and_battery.txt`，`host_info/gpu_consistency.json`）。电池只存了布尔值，没有输出校验和，因此**不能跨主机比较**——下一步应补。
- gate 触发（`GATE exp_avg_rel_l2=1.255e-02`，阈值 1e-5）。但进一步看（`state_comparisons.txt`）：这台主机上 **DP1（gA1d）、DP2（gA2、gA2d）三者对 A8 的 DP1 参照都差 1.255e-2，而 DP1 与 DP2 彼此只差 6.2e-10**。即：这台主机的 DP1 与 DP2 一致（G4 意义下是"好"），但整机与另外几台主机的 step-2 结果不同。gate 的设计前提（"DP1 在所有容器里逐位相同"）在这台主机上不成立，所以这次的"坏主机"是**与参照主机不同的主机**，不是"DP2 的 rank1 出问题"。
- 数据相同：gA1d 的步 1、2 逐样本 loss（hex）与 A8 的 A1 逐位相同（权重也相同，前向相同）；不同的是反传。
- **同一位置的首个分歧**（`dp1_hostA_vs_hostX_step2.txt`）：RC-2 主机（DP1 从头，步 2）与 RC-5a 主机（DP1 从头，步 2）同权重同数据：全部 8 个有梯度的微批前向逐位相同，反传的首个不同张量是 `c0.module.module|bwd_out`（logits 的梯度），之后 `output_layer|bwd_in`、整网各层不同。与 RC-2 中"DP2 的 rank1 样本"的首个分歧位置相同。
- 深探针的 DP1 对 DP2（gA1d 对 gA2d，同一主机）：8 个有梯度样本的 loss 输入、metrics、logits 前向/梯度行校验和、forward-only 通道、完整 logits 梯度张量（样本 16、17、21，bf16 [1,384,151936]）、训练前向/反传校验和全部逐位相同，wgrad 只有次正规数伪差。
- 状态：本次未继续第 2 次尝试（主 agent 指示在此停止并交接）。app 已 stop，无 watchdog 进程。
