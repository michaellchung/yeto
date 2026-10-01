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

## 10. RC-6：kernel 选择因果检验（运行前登记；接手者续作；本任务累计已花 ≈$20.2，剩余上限 $10，本次预估 ≈$3，硬上限 $5.3）
### 10.0 静态核查（CPU，已得）
1. Miles `compute_log_probs`（`math_utils.py`）**无条件**走 Megatron `fused_vocab_parallel_cross_entropy`；`megatron/core/jit.py` 的 `jit_fuser = torch.compile`（torch≥2.2），Miles 与 yeto 都没有调用 `disable_jit_fuser`，也没有设置任何 `TORCHINDUCTOR_*`/`TRITON_*`。Megatron 的 `--deterministic-mode` 只要求 `cross_entropy_loss_fusion=False`（`training/determinism.py`），但该开关只作用于 Megatron 自己的 LM loss，Miles 的 log-prob 路径绕过它。⇒ A8 "确定性模式"下，loss→logits 梯度依然经过 `torch.compile`（Inductor/Triton）生成的 kernel。loss 本身另有 `compute_policy_loss` / `compute_approx_kl`（`@torch.compile(dynamic=True)`），其反传产生 fused CE 的 `grad_output`。
2. 镜像里的 torch 2.13.0+cu130（本地 `/tmp/review-miles-venv` 同版本）默认 `inductor.max_autotune=False`、`coordinate_descent_tuning=False`、`triton.autotune_pointwise=True`、`fx_graph_cache=True`；`triton_heuristics.pointwise` 在 `autotune_pointwise=True` 时给出 **≥2 个 config（num_elements_per_warp 256/64）并按实测耗时挑选**，persistent/普通 reduction 也有按 hint 的 config 集。⇒ 默认就存在"按实测耗时选 kernel config"的机制；Inductor 与 Triton 的缓存目录默认为 `/tmp/torchinductor_<user>` 与 `~/.triton/cache`，**同一容器内的两个 rank 进程和所有 arm 共享**（谁先编译谁写 `.best_config`/cubin，后来者直接读）。这使"每个容器内确定、容器之间不同"的现象在机理上说得通（假设，待验证）。
3. 此前电池只在**单个进程**里跑 GPU0 与 GPU1（两卡共用同一份编译产物和 autotune 结果），不能暴露"每个 rank 进程各自编译/autotune"的差异；训练时每个 rank 是独立进程、同时冷编译。
### 10.1 设计（`tools/probes/e3_reshard/kernel_probe.py`，容器里无 Ray、无训练；`kernel_probe_run.py` 启动；`kernel_probe_compare.py` 离线汇总；CPU 测试 `tests/test_rl_e3_kernel_probe.py`；已在 torch 2.13 CPU 上冒烟通过）
- 每个 worker 是独立进程，用 `CUDA_VISIBLE_DEVICES` 绑 GPU0 或 GPU1，**同时启动**（像训练的两个 rank）。输入固定：8 个样本（token 数 TS=[384,347,291,402,256,331,365,318]，V=151936，种子固定，bf16 logits→fp32→Megatron fused CE→Miles `compute_policy_loss`→backward）。GPU0 的样本顺序从样本 0 起，GPU1 从样本 1 起（`rot`），即两进程首个形状不同（torch.compile 先静态再动态）；两者最终都算完 8 个样本，因此可逐样本逐位比较。
- 每个样本记录：loss、log-prob、fused CE 的 `grad_output`、logits 的 bf16 梯度的 sha256；跑完后哈希 Inductor/Triton 缓存里所有 cubin（kernel 名、num_warps、num_stages）和 `.best_config`（autotune 选择）。
- 阶段（`PHASES`，写死，事后不改）：
  - (a) 现状：默认 Inductor 设置，每次**冷缓存**、两进程共享缓存目录，重复 3 次；
  - (b) 关闭全部 torch.compile（`TORCHDYNAMO_DISABLE=1`，融合 CE 与 loss 都走 eager，即非融合路径），1 次；
  - (c1) 关 autotune（`max_autotune=0`、`coordinate_descent_tuning=0`、`max_autotune_pointwise=0`、`triton.autotune_pointwise=False`），冷共享缓存，重复 2 次；
  - (c2) 固定缓存：先由单个进程冷编译一份缓存（关 autotune），再让两个 rank 只读这份缓存，1 次；
  - (d) 即 (a) 内部 GPU0 与 GPU1 的各自冷编译 config 对比（每个 worker 独立记录 `.best_config`/cubin）。
### 10.2 预先声明的判读（不得事后放宽）
- "逐位相同" = 同一变体内所有 worker（两个 GPU × 所有重复）对全部 8 个样本、全部 4 个字段（loss/logp/grad_output/logit_grad）的 sha256 完全相等。
- **确认根因（autotune/kernel 选择）**需同时满足：(a) 内出现 worker 间 logit_grad 或 grad_output 不相等，且这些 worker 的 `.best_config`/cubin（num_warps 等）集合与其它 worker 不同；(c1) 与 (c2) 内所有 worker 逐位相同。若 (b) 内也不等（eager 都不确定）⇒ 归为硬件/库层（H2），不是编译选择。
- 若 (a) 内也全部逐位相同：本主机不暴露该机制，**不能**确认也不能排除；只记录 cubin/config 指纹作为跨主机参照；是否再抽一台主机（≈$3）由剩余预算决定（剩余 < $5 则停并如实汇报"未确认"）。
- 若 (a) 内 worker 间差异存在但 config 集合相同：编译选择被排除，转向其它来源（如静态/动态 shape 特化、同一 kernel 的硬件差异），如实记录。
- 费用：容器启动 ≈4 min + 7 个 worker 对 ≈ 2 min 各 ≈ 14 min + 取回 ≈ 22 min ≈ $2.9；`Sandbox timeout=2400 s`（上限 $5.3）；独立 watchdog 2700 s；app 名 `a8rc-k1-20261001`；断言 `H100 80GB HBM3` 两张、Miles pin；`--modal-retries 0`（本脚本不重试）。

### 10.3 RC-6 第 1 台主机结果（app a8rc-k1-20261001，ap-CBditBKdGuIVOpv8wFY41V，06:54:07–06:59:54Z，H100!:2，GPU e98f3a30/9cf118ea，≈$0.8；证据 `rc6-k1/`，汇总 `rc6-k1/RESULT_kprobe.json`）
- 全部 worker 退出码 0，driver rc=0；GPU 断言与 Miles pin 通过；torch 2.13.0+cu130、triton 3.7.1。
- 变体 (a)：3 次冷缓存重复 × 2 个独立进程（GPU0/GPU1，首个形状不同）= 6 个 worker，对 8 个样本、4 个字段（loss/logp/grad_output/logit_grad）**全部逐位相同**。(c1)（4 worker）、(c2)（3 worker，固定缓存）同样全部逐位相同；(a)(c1)(c2) 三者之间也逐位相同（以 a_r0_g0 为参照，差异计数全 0）。
- autotune 选择确实会变：(a) 与 (c1) 的 cubin 集合里 `triton_poi_fused_copy__div_split_unsqueeze_1`（softmax 除法）的 num_warps 为 8 对 4；而输出逐位相同 ⇒ **pointwise 的 autotune 选择不影响数值**；关键的 reduction（`triton_red_fused_copy__exp_sub_sum_unsqueeze_2`、`triton_red_fused_max_0`）只有单一 config（16 warps，启发式而非计时选择），所有 worker 一致。
- 变体 (b)（`TORCHDYNAMO_DISABLE=1`，融合 CE 与 loss 走 eager）：两个 worker 彼此逐位相同，但与 (a) 的编译路径 8 个样本全部不同（loss/logp/grad_output/logit_grad 都不同）——**编译路径与 eager 路径在数值上本来就不是同一计算**（预期内；说明如果某台主机走的是另一条路径，结果就会整体不同）。
- 判读（按 §10.2）：**(a) 内无差异 ⇒ 本主机不暴露该机制；H1（计时 autotune 选出不同数值的 kernel）在本主机上被直接反证（autotune 选择变了、数值没变）**；不能据此排除"别的主机上编译路径本身不同"。按登记规则：剩余预算足够再抽主机（每台 ≈$0.8），见 §11。

## 11. RC-6b：跨主机抽样（运行前登记）
- 目的：用同一份探针（代码不变，`kernel_probe.py` 的 TS/PHASES 不变，使哈希可与 §10.3 直接比较）在另外 3 台 H100!:2 主机上各跑一遍完整 PHASES（含 (a)(b)(c1)(c2)），**并行**起 3 个 Sandbox（app `a8rc-k2/k3/k4-20261001`）。每台每个 worker 的哈希与 cubin 指纹与 host 1 比较。
- 预先声明的判读：
  - 某台主机的变体 (a) 对 host 1 的 (a) 在任一样本任一字段不同 ⇒ **找到主机相关的编译路径数值差异**：按字段定位（loss/logp ⇒ 前向 kernel；grad_output ⇒ loss 反传；logit_grad ⇒ CE 反传），对照该主机内 (b)/(c1)/(c2) 与 host 1 的相应变体是否相等，并比对两台主机的 cubin sha/num_warps/triton 版本，得出根因；若该主机 (c1)/(c2) 与 host 1 相等而 (a) 不等 ⇒ autotune/缓存因果成立；若 (b) 与 host 1 (b) 相等而 (a)(c) 都不等 ⇒ 编译产物（Triton/ptxas/硬件）主机相关；若 (b) 也不等 ⇒ 硬件/库层。
  - 三台全部与 host 1 逐位相同 ⇒ 该探针覆盖的链（fused CE + `compute_policy_loss` 的编译 kernel）在 4 台主机上无主机相关性，**H1 被彻底排除**；根因在这条链之外（需要真实训练里的其它算子，例如 grad_output 的上游输入或 lm_head 的 dgrad 以外的路径），转为"未确认"并如实汇报，停止抽样。
- 费用：每台 ≈ 6 min ≈ $0.8，合计 ≈ $2.4；每个 Sandbox 硬超时 `KPROBE_TIMEOUT_S=720`（每台上限 $1.6，合计上限 $4.7）；独立 watchdog 800 s；累计已花 ≈ $21.0，本批后 ≤ $25.7（上限 $30；任务剩余上限 $10 内）。

### 11.1 RC-6b 结果（3 台并行，07:02–07:07Z，各 ≈4.6 min、≈$0.6，合计 ≈$1.8；证据 `rc6-k2/`、`rc6-k3/`、`rc6-k4/`；app a8rc-k2/k3/k4-20261001 均 stopped）
- 4 台主机（k1 GPU e98f3a30/9cf118ea；k2 4d528074/a5136919；k3 a7f6c356/70e5ed9e；k4 090644fc/e5e509c3；全部 H100 80GB HBM3、驱动 580.95.05、triton 3.7.1、torch 2.13.0+cu130）× 4 个变体 × 每主机 15 个 worker：对 host 1 的参照（(b) 对 host 1 的 (b)），**8 个样本 × 4 个字段的差异计数全部为 0**。主机内 autotune 选出的 num_warps 集合每台都稳定一致；cubin 的 sha 在同一主机的不同冷编译之间都不稳定（k1 的 a_r0 与 a_r1 已不同），不能当指纹用，名称/num_warps 在四台主机上相同。
- 判读（按 §11）：**三台与 host 1 全部逐位相同 ⇒ fused CE + `compute_policy_loss` 这条编译链在 4 台主机、8 张 GPU 上没有主机/GPU/rank/冷编译/autotune 相关性；H1 被排除。根因仍未确认。**
- 局限（必须写明）：这是**孤立的单链测试**，输入是固定种子的合成 logits（不是训练里的 logits、old_logp、优势），也没有整网环境（显存布局、之前 kernel 产出的输入、Ray 多进程、真实 lm_head 输出）。因此它不能复现"整网里的主机差异"；也**不能与 RC-2/A8 的坏模式校验和直接比较**：坏模式的 loss→logits 梯度只有逐行校验和/少数样本的完整梯度张量（RC-5a 的样本 16、17、21，RC-2 无完整张量），真实输入 logits 没有保存，无法逐位重放。零成本核对：现有记录里 logits 前向只有校验和（`<logits>|fwd` 的 bits），没有张量，所以这条链无法用真实输入重放。

## 12. RC-7：真实训练里给融合交叉熵打探针，在新抽的主机上定位首个分歧（第三轮接手；运行前登记；用户已批准本轮，上限 $7）
### 12.0 背景与目的
- RC-6/6b 在孤立合成链上排除了 H1（autotune），根因仍未确认。已知：差异的首个位置是 logits 的梯度（`c0.module.module|bwd_out|0`），前向 logits、loss 逐位相同；约 3/5 的主机（RC-2/RC-5a）与参照不同。**未被比较过的**：CE 反传的输入（forward 保存的 softmax/`exp_logits`、`grad_output`、`target_mask`、`masked_target_1d`）与 CE 前向各中间量。
- 目的：在**与 RC-2/RC-4 相同的 harness、整网真实训练**里，记录每次融合 CE 调用的各阶段校验和；在新容器里抽主机；抽到"与参照不同"的主机后，**同一进程同一主机**用同一份（反传前克隆的）输入把 `calculate_gradients` 重算两次（再走一次同一编译函数、再走 eager），与原输出比较。
### 12.1 探针（`rc_trace.install_trace(..., ce_probe=True, ce_save=1)`；只读：包装 `megatron.core.fusions.fused_cross_entropy` 的四个 `jit_fuser` 函数，始终返回原调用的结果）
- 前向每次调用（训练前向与旧策略 forward-only 前向都含；顺序编号 `seq`）：`logits_in`（进入 `calculate_logits_max` 的 logits）、`logits_max`、`logits_shifted_in`（`calculate_predicted_logits` 的输入）、`target`、`target_mask`、`masked_target_1d`、`pred_sumexp`（predicted_logits+sum_exp）、`exp_logits`（除法前）、`softmax`（除法后 = 保存给反传的张量）、`loss`；记 `softmax` 的 data_ptr 供与反传配对。
- 反传（`calculate_gradients`，它**原地**改写 softmax，所以先克隆两份）：输入 `softmax_in`、`grad_output`、`target_mask_in`、`masked_target_in` 的校验和，输出 `out`（bf16）；然后用克隆输入 (i) 再调一次同一个编译函数 `recompute_compiled`，(ii) 调 eager 版（Megatron 同一函数体，不经 torch.compile）`recompute_eager`；记录 `compiled_equal`/`eager_equal`（与原输出 `torch.equal`）。
- 每个 rank、最后一步里**前 1 个 `grad_output` 非零的反传调用**保存完整输入（softmax_in fp32、grad_output、target_mask、masked_target_1d）与输出 `out` 到 host（`big`），供离线逐元素比较；另有 loss 级探针（RC-4 的 eP3 同款：loss 输入、logits/logits 梯度的行校验和）。无模型钩子（`module_hooks=False`）。
- 臂 `gA2e` = DP2 从 0 训练 2 步（与 gate 臂 gA2 相同；`harness.CE_ARMS`），配置 `a8rc-h100e`（`modal_run.PROFILES`）：冻结 rollout 取 RC-5a 取回的 8 个（逐样本 loss 与 A8 相同），容器内 gate 与 A8 的 A1_s2 比 exp_avg（阈值 1e-5，**只用于分类，不再因 gate 停臂**——只有这一个臂）。记录 `host_info/`（`nvidia-smi -q`、`-L`、topo、lscpu、`/proc/cpuinfo` model name、nproc、query.csv：vbios、时钟、ECC、power limit、pci bus）。单测：`tests/test_rl_e3_ce_probe.py`（探针不改变结果；记录齐全；`TORCHDYNAMO_DISABLE=1` 的 CPU 子进程）。离线比较：`ce_compare.py`。
### 12.2 抽样、预算与回收
- 第 1 批：**3 台并行**新容器（Modal `H100!:2`，断言 `NVIDIA H100 80GB HBM3` ×2 与 Miles pin，`--modal-retries 0`：脚本本身不重试；app `a8rc-ce1/ce2/ce3-20261001`；每个 Sandbox `timeout=1020 s`；独立 watchdog `WD_S=1100` 按各自的 resources.txt 停 app）。每台预计 ≈11–13 min ≈ $1.5–1.7；**最坏**（跑满超时）每台 ≈$2.24，3 台 ≈ $6.7 ≤ $7 上限。预估本批 ≈ $4.5–5。
- 第 2 批（仅当第 1 批结束后 实际花费 + 2.24 ≤ 7 且 §12.4 的"所需样本"不满足）：最多再 1 台，同配置。不再有第 3 批。累计 A8 任务已花 ≈ $22.8，本轮上限 $7 ⇒ ≤ $29.8 < $30。
- 取回：只拉 `packed/`（state、trace）+ evidence；停 app 并核对 `modal app list`；台账只追加自己的行。
### 12.3 主机分类与"探针是否改变行为"的判据（运行前固定）
- 分类：gate 的 exp_avg 相对 L2（对 A8 的 A1_s2）≤ 1e-5 为 **good**，> 1e-5 为 **differs**。参照"好"= 与参照一致的主机；两台 differs 主机之间若 state 不同，也可互为比较对象。
- 探针不改变行为的检验：每台的 `gA2e_s2` exp_avg 与三个已存的同口径（DP2 从 0 训练，步 2）状态之一**逐位相同**：A8 `A2_s2`（类 B1，对参照 2.647e-3）、RC-4 `eA2_s2`（good，3.8e-10）、RC-5a `gA2_s2`（类 X，1.255e-2）。逐位相同 ⇒ 探针对该类无影响；都不相同 ⇒ 记为"新类别"，**不能**排除探针影响，如实报告。
### 12.4 判读（运行前固定；"首个分歧阶段"按 `ce_compare.py` 的阶段顺序、对同一 rank 的同序调用、两台 state 不同的主机之间）
所需样本：至少一对 state 不同的主机（good/differs 或 differs/differs），两台都成功取回 trace。
1. 首个分歧阶段 = `logits_in`（CE 的输入 logits）⇒ 差异在 CE 之上游（lm_head 输出或更前）；本轮不再往前，记录并列为下一轮"继续往前一级追"（logits、lm_head 输出、前一层）。
2. `logits_in` 相同而首个分歧在 CE 前向某阶段（`logits_max`…`softmax`、`loss`）⇒ 同输入不同输出：前向 kernel/硬件层面主机相关；记录该阶段、元素差异（若有保存的张量）。
3. CE 前向全部相同、`softmax_in` 相同，而 `grad_output` 不同 ⇒ 差异在 loss 反传（`compute_policy_loss` 反传）而非 CE kernel；`grad_output` 相同而 `out` 不同 ⇒ **同输入的 `calculate_gradients` 输出不同**，进入 4。
4. 同一主机内重算（在 differs 主机上）：
   - `compiled_equal` 与 `eager_equal` 均为 True（重算 == 原输出），且（两台主机输入已不同，见 1/3）⇒ **差异在上游**，继续往前一级追；
   - 两台主机输入相同、输出不同，且在 differs 主机上同输入重算仍等于其原输出（`compiled_equal=True`）而 ≠ good 主机的输出 ⇒ **kernel/硬件层面的主机相关非确定性**，记录 GPU 型号、驱动、频率、SM 数、ECC、vbios、pci 等（host_info/）；若 `compiled_equal=False`（同机同输入重算 ≠ 原输出）⇒ 主机内非确定性（竞争/未初始化内存等），单独记录；
   - `eager_equal=False` 而 `compiled_equal=True` ⇒ 编译与 eager 本来就是不同计算（RC-6 已知），**不据此判因**，只记录。
5. 所有 CE 阶段（含 `out`）在 state 不同的主机间都相同 ⇒ 差异产生在 CE 反传输出之后（lm_head dgrad 及以下）或探针改变了行为；记录并按 §12.3 检验。
6. 抽不到 state 不同的主机（第 1、2 批共 ≤4 台均与同一 state 逐位相同）⇒ 如实报告"未抽到"，不再加批；`compiled_equal`/`eager_equal` 与 `probe_errors` 仍报告。探针自身出现 `probe_errors` 或 `fwd_ptr_match` 缺失 ⇒ 如实报告，不据此下根因结论。
- 不事后放宽：以上判读与 §12.3 的检验在运行前提交，不改；不改 pin、不改 tasks.md 4.6 勾选、不降级 4.7/4.8。

### 12.5 RC-7 批 1 结果（3 台并行，07:21:30–07:29:31Z，各 ≈7.5–8 min，合计 ≈$3.1；apps a8rc-ce1/ce2/ce3-20261001 均 stopped；证据 `rc7-ce/`，比较 `rc7-ce/ce_compare_ce1-3.json`、`states_vs_known.py` 的输出见下）
- 三台全部通过 GPU 断言（H100 80GB HBM3 ×2，驱动 580.95.05，132 SM）与 Miles pin；vbios：ce1 96.00.DB.00.02，ce2/ce3 96.00.DA.00.0C；CPU：ce1 Model 143（Sapphire Rapids），ce2/ce3 Model 207（Emerald Rapids）；ECC 开、无未纠正错误。
- gate：三台 `exp_avg_rel_l2=1.255e-02`（> 1e-5），**三台的 `gA2e_s2` exp_avg 都与 RC-5a 的 `gA2_s2`（类 X）逐位相同**（对 A8 `A2_s2` 1.225e-2、对 RC-4 `eA2_s2` 1.255e-2、对 A8 的 A1_s2 1.255e-2）。按 §12.3：探针对类 X 无影响（探针版 gA2e 与 RC-5a 无探针 gA2 逐位相同）。
- CE 探针：每台每 rank 16 次前向调用、8 次反传调用（4 次 `grad_output` 非零），`fwd_ptr_match` 8/8，`probe_errors` 空；**同一主机内重算：`compiled_equal` 与 `eager_equal` 全部 True（同输入、编译与 eager 两种重算都与原输出逐位相同，eager 与原输出最大绝对差 0.0）**。
- **三台之间所有 CE 阶段（前向 10 个阶段、反传 5 个字段）、logits 行校验和、logits 梯度行校验和、保存的完整张量（softmax_in、grad_output、out，两个 rank 各 3.5–3.6 千万元素）逐位相同，差异元素数 0**。
- 按 §12.4 规则 6：**三台与 RC-5a 属同一 state，没有 state 不同的主机对 ⇒ 未抽到可比较的 "好/坏" 对**；规则 1–5 不适用（没有分歧阶段）。这本身是重要事实：**类 X 在 4 台不同主机（含两种 CPU、两种 vbios）上逐位可复现，不是"每台主机各自随机"**；按时间序：A8（09-30）、RC-2（04:03）、RC-3（04:51）、RC-4（05:28）的 DP1/参照类为"ref"，RC-5a（06:12）、ce1–3（07:21）全为类 X。同时 gate 臂（gA*：`last_step=2`、无 `cut_at_step2`）与 A 臂（A1/A2/rcA*/eA*：`cut_at_step2`、`last_step≥3`）在配置上有差异，而此前所有类 X 的数据都来自 gate 臂，**这是一个未被分离的混杂因素**（步 2 的训练本身与这两项无关，理论上不应影响，但没有直接检验）。

## 13. RC-7b：分离"配置 vs 抽主机/时间"（运行前登记；用户批准本轮上限 $7；批 1 已花 ≈$3.1，本批最坏 $3.4，合计最坏 $6.5）
- 动机：§12.5 的 4 台类 X 主机全部来自 gate 臂配置；"ref/B1/good"类全来自 A 臂配置。要判断类 X 是（a）当下 Modal 主机池的性质，还是（b）臂配置的后果，需要在**当前**主机上跑**A 臂配置**。
- 臂 `cA2e`（`harness.CE_ARMS`）：DP2 从 0 训练，`cut_at_step2="C2"`、`last_step=3`（与 RC-4 的 `eA2` 完全同配置），加同一 CE 探针（`ce_save=0`，不存大张量），`trace_steps=(1,2)`（步 2 的 CE 记录保留）。配置 `a8rc-h100f`。gate 在该臂后，参照与 §12.3 相同（对 A8 的 A1_s2，阈值 1e-5）。
- 2 台并行新容器（app `a8rc-cf1/cf2-20261001`；Sandbox `timeout=780 s`；独立 watchdog 850 s；断言 GPU 名；无重试）。预估每台 ≈ 9–10 min ≈ $1.2–1.3；最坏每台 $1.71，合计 $3.4。批 1 实际 ≈$3.1 ⇒ 本轮最坏 ≈ $6.5 < $7；A8 累计最坏 ≈ $29.3 < $30。**之后不再起任何运行**。
- 预先声明的判读：
  1. 两台的 `cA2e_s2` exp_avg 都与类 X 的 `gA2_s2` 逐位相同 ⇒ **配置（cut/last_step）被排除**；类 X 是当前主机池的性质，与 A8/RC-2/3/4 时期的主机池不同——这是"主机/环境相关"，进一步分离需要拿到 ref 类主机（未能抽到），根因机制未查明。
  2. 任一台的 `cA2e_s2` 与参照一致（≤1e-5，good/ref 类）而类 X 仍是 gate 臂 ⇒ **配置因素成立（同批主机池里 A 臂配置与 gate 臂配置给出不同数值）**；此时把该台 `cA2e` 步 2 的 CE 记录与类 X 的 `gA2e`（ce1，已验证三台一致）按 §12.4 规则 1–5 比较（"好"= cA2e），定位首个分歧阶段，并结合 §12.1 的同机重算标志下结论。
  3. `cA2e_s2` 与类 X、ref 都不同（如 B1 类 2.647e-3 或新类）⇒ 如实报告类别，并对任意两个 state 不同的状态对按 §12.4 比较。
  4. 两台之间 state 不同 ⇒ 同配置下主机相关，按 §12.4 比较这两台。
  5. 探针错误/`fwd_ptr_match` 缺失/容器失败 ⇒ 如实报告。
- 不事后放宽；不改 pin、不改 tasks.md、不降级 4.7/4.8。

### 13.1 零费用离线核查（主 agent 要求；结果在 RC-7b 之前得出）：gate 臂（last_step=2、无 cut）与 A 臂（cut_at_step2、last_step≥3）的 Miles argv/运行时超参是否不同
- **argv 逐项相同**：`evidence/**/miles_args*.json` 里 a8-run2（A1/A2/B1/B1p/B2/RT/gen/dry）、RC-1a/1d、RC-2、RC-3、RC-4、RC-5a（gA2/gA1d/gA2d）所有 arm 的 `argv`、`argv_profile`、`overrides`、`parsed_profile` **完全相同**（脚本比对：除 dev-gather 的另一种 profile 外差异为空）；RC-7 的 gA2e 与 RC-5a 的 gA1d 同样为空（本轮 `miles_args.arm.gA2e.json`）。相关项：`--num-rollout 8`（固定，**不随 `last_step` 变**，`last_step` 只是 harness 的循环上界，`harness.py` 的 `while step < spec.last_step`）、`--lr 1e-5 --lr-decay-style linear --lr-decay-iters 8 --lr-warmup-iters 0 --min-lr 0`、`--seed 1234 --rollout-seed 1234`、`--rollout-batch-size 2 --n-samples-per-prompt 8 --global-batch-size 16`、`--num-steps-per-rollout 1`、LoRA rank16 canonical、`--deterministic-mode`。
- **步 1–3 实际应用的 lr 逐位相同**（各 run 的 `arms/*/events.jsonl` 的 `train` 事件 `applied_lrs`）：所有 arm（A8 的 A1/A2、RC-2 rcA1/rcA2、RC-4 eA1/eA2、RC-5a gA2/gA1d/gA2d）步 1 = 1e-5、步 2 = 8.75e-6、步 3 = 7.5e-6；步 1 的 `grad_norm` 全为 0.0；**步 2 的 grad_norm 不同**：A8 A1/RC-2 rcA1/RC-4 eA1/eA2 = 0.9685642719268799（ref 类，含 RC-4 的好 DP2）；A8 A2/RC-2 rcA2 = 0.9686059355735779（B1 类）；RC-5a 的 gA2/gA1d/gA2d = 0.9683535099029541（类 X，DP1 与 DP2 相同）。⇒ 差异只在步 2 的梯度，不在 lr/调度。
- **打包状态里的调度/超参摘要相同**：`packed_index.json` 的 `scheduler`（a5eafcaa0d）、`hyper`（44eba6f139）、`step`（10db114ecc）、`counters`（094195aaa3）、`shape` 在 A8、RC-2、RC-4、RC-5a 的所有 s2 状态里**相同**；只有 `adapter`/`param`/`exp_avg`/`exp_avg_sq` 不同：adapter 摘要 ref 类 `e69628972b`（A8 A1、RC-2 rcA1、RC-4 eA1/eA2）、B1 类 `78e13fdf93`（A8 A2、RC-2 rcA2）、类 X `286d666b2d`（RC-5a 的 gA2/gA1d/gA2d；RC-7 的 gA2e 三台与之逐位相同）。
- seed/数据切分：argv 相同；数据同一份 8 个冻结 rollout（逐样本 loss 逐位相同，`rc2-h100/data_identity.txt`、RC-5a）；**cut 的分支**只在步 2 训练**之后**（`harness.run_arm`：训练 → `save_cut` → `_dump` → trace dump），不在步 2 的前向/反传之前，且 `backend.train` 里不含与 `last_step` 相关的分支（`miles_backend.py`）。因此**已知没有任何 argv/lr/seed/调度层面的配置差异**可以解释类 X；A 臂与 gate 臂的唯一剩余差别（cut 的 save/flush 发生在步 2 之后；`last_step` 仅为循环上界）理论上不影响步 2 的反传，**RC-7b 直接检验它**。结论："坏模式不是超参/lr 配置差异造成的"这一点有证据；A8 的 A1 对 A2（同一配置、同一容器）本身的 0.83% 差异与臂配置无关（两者同配置）。

### 13.2 RC-7b 结果（2 台并行，07:34–07:43Z，各 ≈7–9 min，合计 ≈$2.1；apps a8rc-cf1/cf2-20261001 均 stopped；证据 `rc7-ce/cf1`、`cf2`、`states_vs_known_cf.txt`、`ce_compare_ce1_cf1_cf2.json`）
- 两台通过 GPU 断言与 Miles pin；vbios 分别为 96.00.CF.00.01、96.00.D0.00.02（加上批 1 的 DB.00.02、DA.00.0C，共 4 种 vbios）；CPU Model 143。gate 都是 `exp_avg_rel_l2=1.255e-02`。
- 按 §13 判读 1：两台的 `cA2e_s2`（A 臂配置：`cut_at_step2`、`last_step=3`，与 RC-4 的 `eA2` 同配置）exp_avg **都与类 X 的 RC-5a `gA2_s2` 逐位相同**（对 RC-4 好状态 1.255e-2）。⇒ **臂配置（cut/last_step）被排除**，类 X 是"当前这批主机/时期"的性质；至此类 X 已在 **6 台**相继抽到的主机（RC-5a、ce1–3、cf1–2；4 种 vbios、2 种 CPU 型号）上逐位复现。
- CE 探针：cf1/cf2 的每 rank 前向 16 次、反传 8 次（4 次非零）、`fwd_ptr_match` 8/8、无 probe_errors；同机重算 `compiled_equal`/`eager_equal` 全 True。cf1、cf2 与 ce1 之间所有 CE 阶段、logits 行校验和、logits 梯度行校验和逐位相同（`ce_compare_ce1_cf1_cf2.json` 的 first-diff 全空）。
- 按 §12.4 规则 6：**没有 state 不同的主机对，无法在 CE 层面定位 good/bad 分歧阶段**；本轮不再起运行（预算与规则均已用尽）。

### 13.3 零费用离线核查 II（主 agent 要求）：步 2 各类实际消费的输入是否逐位相同（脚本 `rc7-ce/inputs_cmp.py`、`logp.py`，输出 `inputs_cmp.txt`、`logp.txt`）
比较对象：RC-3 `dA1`（DP1 从 0 训练，深探针，状态与 A8 参照 A1_s2 逐位相同 = ref 类 R）的步 2 trace 对 RC-5a `gA1d`（类 X）的步 2 trace（均有 loss 级与 forward-only 记录）；RC-2 `rcA1`（ref）对 `gA1d`；DP2：RC-2 `rcA2`（B1 类）对 RC-5a `gA2d`（X）按 rank。
- **逐位相同的输入（DP1，ref 对 X，步 2，16 个微批）**：`advantages`、`loss_masks`、`response_lengths`、`total_lengths`、`loss`（pg loss）、`logits_row_bits` 16/16；`<root-input>`（token ids、位置等）32/32，forward-only 通道的 `<root-input>` 32/32；**训练前向所有模块输出（含 logits）27040/27040、forward-only（old-policy）前向所有模块输出（含 logits `module.module`、final_layernorm、output_layer、各层）496/496 逐位相同**。（rewards 字段此版探针未记录；sample_indices 在 RC-3 的探针版本里未记录，所以那两项没有跨类可比数据。数据同一性另见 `rc2-h100/data_identity.txt`：A8 与 RC 的逐样本 loss 逐位相同。）
- **不同的"输入"：`log_probs`（old-policy log-prob，loss 的输入）在 16/16 个微批上不同**（2609 个 token 里 1455 个不同，最大绝对差 9.5e-7，约为 fp32 在 1 附近的几个 ulp；例如 -2.2421e-4 对 -2.2397e-4，-1.2540e-4 对 -1.2552e-4），`loss_metrics`（如 `train_rollout_logprob_abs_diff` 0.011533367 对 0.011533363）16/16 不同。**但产生这些 log-prob 的 logits 在两类里逐位相同** ⇒ 差异产生在 **logits→log-prob 的计算（Miles `compute_log_probs` → Megatron 融合 CE 前向：max、exp、sum_exp 归约、log）本身**，即 CE **前向**在 ref 类与类 X 里对同一 logits 给出不同的位（量级符合 `sum_exp` 求和顺序/内核实现不同的 1–几个 ulp）。这是此前"前向逐位相同"说法的重要修正：逐位相同的是 logits 与 pg loss，不是 old log-prob。
- 反传侧：首个分歧模块与 RC-2/RC-5a 一致（logits 梯度起）；`logit_grad_row_bits` 16 个里 6 个相同（grad_output=0 的样本梯度恒 0，因此相同）。
- **第 1 步结束后的参数状态 s1**：harness 只在步 2、3、8 做状态 dump（`DUMP_STEPS=(2,3,8)`），**没有 s1 dump，无法直接比较**；间接证据：步 1 的 grad_norm 全为 0.0（所有 arm），步 2 起点的所有模块前向输出（含 logits）跨类逐位相同，说明步 2 实际使用的参数在 bf16 前向意义下相同。
- **DP2 分 rank 的新事实**（`inputs_cmp.txt`）：RC-2 `rcA2`（B1 类）的 **rank1** 与 RC-5a `gA2d`（X）的 **rank1** 的全部前向/反传记录**逐位相同**（反传 13576/13576、13432/13432）；而 B1 类 **rank0** 对 X 的 rank0 反传不同（9049/13432 相同），且 B1 的 rank0 已知与 DP1 ref 逐位相同。⇒ 数值上存在两种"模式"：R（ref：A8 的 DP1、B1 的 rank0、RC-3/RC-4 的好主机）与 X；**模式是按进程（rank）而不是按主机或臂配置分配的**：A8/RC-2：rank0=R、rank1=X；RC-3/RC-4：两个 rank 均 R；RC-5a 与 RC-7/7b 的 6 台主机：两个 rank 均 X。
### 13.4 本轮结论（运行前登记的规则下如实陈述）
- **根因机制仍未查明**。已查明/收敛的：(1) 差异的本质是**融合 CE 前向（logits→log-prob）对同一 logits 在两种模式 R/X 下给出不同位**（old log-prob 就已不同，随后 `ratio`、`grad_output`、softmax 保存值、logits 梯度、整网反传全部带着这个差异；不是 CE 反传 kernel 的问题）；(2) 在 X 模式的 6 台主机（含 4 种 vbios）上 CE 全部阶段逐位可复现，同机重算（编译/eager）等于原输出，因此 X 模式内部是确定的；(3) 模式按进程分配（B1 类的 rank0=R、rank1=X），不是按配置（RC-7b）、不是按超参/lr（§13.1）、不是按数据（§13.3）；(4) 时间上：A8、RC-2/3/4（含 RC-4 的所有 DP2 好臂）为 R 为主，RC-5a 起的 6 台全为 X。
- 因此这不是"DP1 对 DP2 的归约数学"、不是切换实现缺陷（RC-1d/RC-2/RC-3 已证），也没有证据表明验收门槛不合理（同模式内 DP1 对 DP2 在 1e-10 量级）；**它是"同一程序、同一 logits，在不同进程/时期落到两种不同数值模式"的环境相关问题**，机制（R/X 由什么选择）未查明。
- 剩余可能方向（均未检验）：(a) Inductor/Triton 编译产物的差异——对同一融合 CE，归约核的向量化宽度/pointer 与整数参数的 16 对齐特化（`divisibility`）、静态/动态形状特化顺序，可能随进程的分配历史而变（RC-6 的孤立合成链没有复现该分配历史）；(b) 编译缓存（`/tmp/torchinductor_*`、`~/.triton`）在容器内被多进程/多 arm 共享，产物由先编译者决定（RC-6 的 c2 只覆盖了合成链）；(c) 镜像之外的、随时期变化的主机环境（驱动 JIT 缓存、Modal 宿主机池更换；4 种 vbios 与 2 种 CPU 都表现为 X，所以 vbios/CPU 型号本身不像决定因素）；(d) 因 RC-6 的合成链在 X 时期跑，**R 模式从未在受控条件下复现**，因此 RC-6 的"这条链主机无关"只对 X 模式成立。
- 建议的最小下一步（需要新预注册与预算，本轮预算已用尽）：在真实进程里直接取出被调用的 Triton kernel 的元信息（`num_warps`、`size_per_thread`/vector 宽度、`divisibility` 特化、`RBLOCK`，可由 `torch._inductor` 的 kernel 源码/`CompiledKernel.metadata` 取得）并与"能出 R 模式"的环境比较；难点是目前抽不到 R 模式的主机——可以反过来用 X 模式下强制不同的 pointer/size 对齐（例如把 logits 切片偏移 1 个元素）看能否在受控下制造出 R 模式（对 `sum_exp` 的前向输出做逐位比较，几分钟的单机实验，≈$1）。
