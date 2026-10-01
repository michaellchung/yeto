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
