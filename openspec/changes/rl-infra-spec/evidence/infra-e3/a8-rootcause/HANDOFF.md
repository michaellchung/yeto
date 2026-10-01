# A8 G4 根因调查 交接（INFRA-E3，2026-10-01 06:45Z；到 RC-5 第 1 次为止；接手者请先读完本文件再读 plan.md）

分支 `infra-e3`（worktree `/home/michael/work/infra-e3`），远端 `origin/infra-e3`。证据目录：`openspec/changes/rl-infra-spec/evidence/infra-e3/a8-rootcause/`（下文路径都相对于它，除非写绝对路径）。**tasks.md 的 4.6 勾选状态未改，4.7/4.8 未降级**，progress.md 仅追加记录。用户要求：不得按"合法否定结论"勾 4.6，必须查明根因属于 (1) 正常跨 DP 数值差异 (2) 切换实现缺陷 (3) 验收标准不合理；"bf16 精度低""loss 相同"不算根因。GPU 规则：只用 Modal（或经批准的 Nebius），固定型号并断言；本任务总上限 **$30**（用户已放宽），单次 ≤$8；每次运行前先在 `/home/michael/work/infra-drafts/gpu-spend.md` 记一行、把计划写进 plan.md 并提交、并向主 agent 报告；运行后报告结果；用完核对 `modal app list`（只管 `a8rc-` 前缀）。

## 1. 一句话现状
- **切换实现没有缺陷**（逐位证据，三次运行）；**DP1 对 DP2 的数学/归约没有问题**（同一主机上两者在 1e-10 量级一致）；A8 的 G4 失败来自**依赖具体 Modal 主机/GPU 实例的、确定性的反传数值差异，起点在 loss→logits 的梯度（forward 逐位相同）**。为什么某些主机会这样、具体是哪个算子/代码路径，**尚未查明**。因此目前既不是 (1) 正常跨 DP 数值差异，也不是 (2) 切换缺陷，(3) 判据不合理也没有证据支持——更像第四类：环境/主机相关的数值不一致（需要把它收敛到具体算子并做因果验证）。
- 费用：本任务累计 ≈ **$20.2**（上限 $30；剩余 ≈ $9.8，建议留 ≥$4 给修复验证）。逐次见 §3。无任何在跑的云资源（核对：`modal app list` 里 a8rc-rc1/rc1b/rc1c/rc1d/rc2/rc3/rc4/rc5a 均 stopped；无 watchdog 进程）。

## 2. 已证实 / 已排除（每条附证据）
已证实：
1. **切换路径逐位干净**：同一状态 C1、同一冻结数据、步 3：DP2 经重分片加载 == DP2 标准同形加载（`rc1d-a10g/RESULT_rc.json`、`rc2-h100/RESULT_rc.json`、`rc3-h100/RESULT_deep.json` 的 `states.rcBc_vs_rcSb`/`dBc_vs_dSbX`）；DP1 经反向重分片 == DP1 标准加载（`rcDd_vs_rcSa`）；DP1 同形恢复 == 连续训练（`rcSa_vs_rcA1`）；恢复后状态 == C1。A10G 与 H100 都如此。
2. **同一主机上 DP1 与 DP2 一致到 1e-10**：RC-3（`rc3-h100/RESULT_deep.json`：6.3e-10）、RC-4 的 6 个 DP2 arm（`rc4-h100/RESULT_bisect.json`：6.2e-10）、RC-5a（`rc5a-h100/state_comparisons.txt`：6.2e-10）。剩余差异只是 fp32 累加结合律（`rc2-h100/RESULT_rc.json` 的 `reaccumulate_DP1_vs_DP2_order` 3.3e-10 对应）。
3. **A8 的 0.83% 被复现**：RC-2（H100，04:03Z）上 DP2 对 DP1 梯度相对差 8.29e-3、update 9.13e-3、sign 99.85%、exp_avg 2.96e-3，且 A8 的 B1_s3 与 RC-2 的 rcBc_s3 **逐位相同**（`rc3-h100/cross_run_bitwise.txt`）。A8 与 RC-2 的 DP2 从头训练 2 步 exp_avg 对 DP1 差 2.647e-3（两处所有位数一致，`rc4-h100/gate_reference_check.txt`）。数据与 A8 相同（`rc2-h100/data_identity.txt`：A8 的 A1 与 RC 的 rcA1 步 1–3 逐样本 loss 逐位相同）。
4. **坏结果只出现在 DP2 的 rank1 样本**（RC-2）：16 个样本里 8 个优势为 0（梯度恒 0），其余 8 个：rank0 上的样本（40、42、44、46）与 DP1 逐位相同，rank1 上的样本（41、43、45、47）全部不同；首个分歧张量是 **logits 的梯度**（`c0.module.module|bwd_out|0`，[1,T,151936]），其前向 logits、loss 逐位相同；随后逐层放大（wgrad 相对差 layer27 ≈1e-5 → layer0 ≈1.8e-2；`rc2-h100/RESULT_rc.json`、plan.md §5）。
5. **同一首个分歧位置也出现在"主机不同"的比较里**（RC-5a）：RC-2 主机与 RC-5a 主机上 DP1 从头训练步 2（同权重同数据）：前向全部逐位相同，反传首个不同张量同样是 `c0.module.module|bwd_out|0`（`rc5a-h100/dp1_hostA_vs_hostX_step2.txt`）。RC-5a 主机整机（DP1 与 DP2 都）对 A8 的 DP1 参照差 1.255e-2，而 DP1 与 DP2 彼此 6e-10。
6. **坏结果按容器/主机二分、确定性**：好 = RC-1d(A10G)、RC-3、RC-4；坏（DP2 rank1）= A8、RC-2；RC-5a 是第三种（整机与参照主机不同，DP1=DP2）。软件环境逐项相同（镜像 digest、环境变量、TE 2.17.0、torch 2.13.0+cu130、cuDNN 9.22、驱动 580.95.05、132 SM；`rc2-h100/device_and_kernels.json`）。
已排除（因素 → 证据）：
- 探针扰动：RC-4 的无探针 DP2 arm（eP0）是好模式；A8 本身没有探针却是坏模式（`rc4-h100/RESULT_bisect.json`）。
- "此前跑过 DP2 从头训练 arm"的跨 arm 泄漏：eP0b（A2 之后同 Ray）好；eP0r（Ray 集群重启后新进程）好（同上）。进程语义：每个 arm 都是新的 learner 进程 + 新的 Ray actor（`miles_backend.py` 的 `start_arm`/`stop_arm`；`Disposer` 释放），共享的只有 Ray 集群、文件系统、GPU。
- launch 时序/stream 竞争：eCLB（`CUDA_LAUNCH_BLOCKING=1`）好。
- 物理 GPU 编号/rank 与物理 GPU 的对应：RC-3 交换后（`rc3-h100/gpus_swapped.txt`）DP1、DP2 的结果与交换前逐位相同。
- GPU kernel 选择：rcSa/rcSb 微批 0 的 CUDA kernel 名称多重集合 rank0/rank1/DP1 完全一致（`rc2-h100/device_and_kernels.json`；仅限微批 0，那是优势为 0 的样本）。
- loss 输入不同（old_log_probs、优势、mask、rewards）、loss 缩放（power-of-two，静态核对见 `../a8-run2/g4-analysis.md` §1–3）：RC-3 在深探针下这些全相同（但 RC-3 是好主机，故只说明"在好主机上输入相同"）。
- 数据不同：`data_identity.py` 结果（RC-2、RC-5a）逐样本 loss 与 A8 相同。
- 梯度归约/累加：fp32 且 power-of-two 缩放；离线重做 DP1/DP2 顺序累加差 3e-10。
- 旧结论"最后一层差异为 0 / bf16 反传不可避免"被推翻：层 27 并非 0（2.6e-5，与重建噪声同阶），且同样本在 rank0 上与 DP1 逐位相同（`offline_g4_layers.txt`、plan.md §0）。

## 3. RC-1 … RC-5 每次结果与费用（详细在 plan.md 对应小节）
| 次 | 平台/时间 | 结果 | 费用 |
|---|---|---|---|
| RC-1 第 1 次（`rc1a-failed/`） | Modal A10G:2，01:28–01:49Z | 探针缺陷：dump 时才 `.cpu()`，offload 后 GPU 内存失效；训练步正常 | ≈$0.8 |
| RC-1 第 2 次（`rc1b-refused/`） | A10G | 分配到 `NVIDIA A10`，旧断言拒绝，exit 3 | ≈$0 |
| RC-1 第 3 次（`rc1c-killed/`） | A10G，02:44–03:04Z | 被我遗留的旧 watchdog 误停（读了复用的 resources.txt），无结论；脚本已改为每次独立目录 | ≈$0.75 |
| RC-1d（`rc1d-a10g/`） | A10G，03:14–03:56Z | 四路径同起点：切换路径逐位干净；DP1 对 DP2 仅 4.5e-10，**未复现 G4**（按预登记改用 H100） | ≈$1.55 |
| RC-2（`rc2-h100/`） | H100!:2，04:03–04:40Z | **复现 G4**；rank1 样本独有、起点 logits 梯度；数据同 A8；kernel 名一致 | ≈$4.8 |
| RC-3（`rc3-h100/`） | H100，04:51–05:20Z | 深探针 + 交换物理 GPU：**这次没有出现差异**（好模式）；A8 的 B1 与 RC-2 的 rcBc 跨容器逐位相同 ⇒ 坏模式确定性、按容器二分 | ≈$3.75 |
| RC-4（`rc4-h100/`） | H100，≈05:28–06:06Z | 二分：探针/arm 顺序/Ray 重启/CLB 全部排除，六个 DP2 arm 都是好模式 | ≈$5.0 |
| RC-5 第 1 次（`rc5a-h100/`） | H100，≈06:12–06:39Z | 电池干净（但无校验和）；gate 触发但原因是整机与参照主机不同；DP1=DP2；首个分歧位置同 logits 梯度 | ≈$3.6 |
合计 ≈ $20.2（按 H100×2 ≈ $7.9/h、A10G×2 ≈ $2.2/h 与实际运行时长估算，未查 Modal 账单）。台账：`/home/michael/work/infra-drafts/gpu-spend.md`（已逐次记录；RC-5a 那一行的"待回填"需补成 ≈$3.6、stopped）。

## 4. 当前假设与下一步
共同点：**差异的首个位置是 loss→logits 梯度这一步，forward 逐位相同**。这一步的代码路径：Miles `loss_hub/losses.py:policy_loss_function` → `math_utils.calculate_log_probs_and_entropy` → `compute_log_probs` → Megatron `megatron/core/fusions/fused_cross_entropy.py` 的 `_VocabParallelCrossEntropy`，其 `calculate_logits_max / calculate_predicted_logits / calculate_cross_entropy_loss / calculate_gradients` 都是 `@jit_fuser`（`torch.compile`），`calculate_gradients` 末尾 `.to(torch.bfloat16)`。
- H1（首选）：torch.compile/Inductor/Triton 生成的 `calculate_gradients`（或其输入 `exp_logits`）在不同主机上数值不同：编译产物或 autotune/specialization 选择与主机相关（但本批软件环境相同，编译缓存文件系统相关性未查；A8 的 B1 与 RC-2 的 rcBc 逐位相同说明每种模式是确定的）。
- H2：GPU 实例级硬件/固件差异使同一 kernel 在不同实例上结果不同（RC-5a 电池 GPU0 对 GPU1 一致，但没有校验和可跨主机比较）。
- H3：某些 GPU 对（rank1）在 A8/RC-2 主机上有别于 rank0 的行为（如 RC-2 的 GPU 4852a4d7/42f4be0c），原因未知。
下一步（建议顺序，低成本优先）：
1. **给电池加输出校验和**（每个算子输出的 sha256 + 设备 UUID），把电池跑在多台新容器上（每次 ≈$0.4，只起容器不训练，可连续抽多台），看哪个算子在不同主机/GPU 上输出位级不同；并记录 `nvidia-smi -q`、`lscpu` 与"好/坏"的对应。注意 `lscpu` 在容器里 Model name 是 unknown。
2. **保存 `calculate_gradients` 的输入与输出**（`exp_logits`、`grad_output`、`target_mask`、`masked_target_1d`、输出 bf16）：在"坏"与"好"主机上各抓同一样本（RC-5a 已有完整 logits 梯度张量的做法，见 `rc_trace.py` 的 `logit_grad_samples`），在同一台主机上离线用 eager 重算 `calculate_gradients` 并与 compiled 比较，判定是"输入不同"还是"同输入不同输出（编译产物/硬件）"。
3. 因果干预（需先登记）：把 `jit_fuser` 关掉（Megatron 有 `--no-...`/环境开关：查 `megatron/core/jit.py`，或在探针里把 `fused_cross_entropy.calculate_*` 替换为未编译版本），或设 `TORCHINDUCTOR_*` 禁用缓存/autotune，在已知的"坏"主机上验证 DP2 对 DP1 的 G4 指标是否降到 1e-9。找到"坏"主机的办法：A8/RC-2 是坏（DP2 rank1），RC-5a 是"整机不同"，gate 脚本 `gate.py` 现在只判断"与参照 DP1 是否一致"，其结论需配合 `gA1d` 对 `gA2d`（同主机 DP1/DP2）一起看；建议把 gate 改成"同一容器内先跑 DP1 再跑 DP2 的 2 步并比较"，不依赖外部参照。
4. 若确认是编译/主机相关的数值非确定性：修复方式（关闭 jit_fuser 或固定编译产物/环境）+ 针对性回归测试；之后在"坏"主机上重跑 A8 的 G4。若证明不可避免：需另行论证判据（同主机基线 + 更多 seed/往返）并另行预注册，**不得为通过而放宽**；4.6/4.7/4.8 的状态由用户决策。
## 5. 脚本入口与用法（均在 `tools/probes/e3_reshard/`，测试 `tests/test_rl_e3_rc_trace.py`、`tests/test_rl_e3_harness.py`，CPU 通过：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q tests/test_rl_e3_harness.py tests/test_rl_e3_rc_trace.py`）
- `modal_run.py <profile> <outdir> <learner_flags.txt> <app-name> <repo snapshot>`：启动 Modal Sandbox。profile：`a8rc`（A10G）、`a8rc-h100`（RC-2）、`a8rc-h100b`（RC-3）、`a8rc-h100c`（RC-4）、`a8rc-h100d`（RC-5，电池+gate）。硬超时、`arm_deadline_s`（容器内新 arm 最迟时刻）、GPU 型号断言、Ray 重启/交换 GPU（`swap_gpus_before`、`env_restart_before`）都在 PROFILES 里。环境变量 `E3_DRY_PYTHON`（本地 dry-run 用的 python，须有 yeto 依赖）、`E3_REF_FILE`（gate 的参照状态，A8 的 `A1_s2.pt`）、`E3_FROZEN_DIR`（上传冻结 rollout 以跳过 gen；RC-5a 取回的在 `/home/michael/work/infra-e3-gpu/a8rc/frozen_rc5a`，其内容是 8 个 `rollout_N.pt`，注意它与 A8 的数据 loss 逐位相同）。
- 包装脚本（不在仓库）：`/home/michael/work/infra-e3-gpu/a8rc/go.sh <app-name>`（环境 `PROFILE=...`、`WD_S=...`；每次运行输出到 `out-<app-name>/`，自带按**本次** resources.txt 停 app 的 watchdog，结束时杀掉它——旧版本曾误停过后续运行）；快照用 `git archive HEAD | tar -x -C <dir>/yeto` 再复制 `/home/michael/work/infra-e3-gpu/b3a8r/yeto/gsm8k_reward.py` 到快照根；learner 参数 `/home/michael/work/infra-e3-gpu/a8rc/learner_flags.txt`；运行前先 `local_dry.py <profile> <flags> <out.json>`。Modal 可能排队（RC-1c 等了 20 分钟才起），首次 A10G 可能给 A10（断言已放宽）。
- `harness.py`：`ArmSpec`（dp、restore、standard、trace、trace_opts、deep…）；arm 集合 `RC_ARMS`、`DEEP_ARMS`、`BISECT_ARMS`、`HOST_ARMS`。同一状态 C1 的四路径：`rcA1`（DP1 连续，产出 C1）、`rcBc`（DP2 重分片恢复，存 C1p）、`rcSa`/`rcSb`（DP1/DP2 标准同形恢复）、`rcDd`（反向重分片）。
- `rc_trace.py`（Ray actor 内 `run_plugin`）：`install_trace`（模块前向/反传钩子、校验和、wgrad、forward-only 通道、loss 包装、logits 梯度行校验和与完整张量、kernel 名 profiler、设备/环境信息）与 `dump_trace`；梯度记录前乘 `1/dp`（2 的幂，精确）以便 DP1/DP2 逐位可比。
- 离线分析：`compare_rc.py <run dir>`（四路径状态比较、逐微批追踪首个分歧、wgrad、fp32 重累加）；`deep_compare.py <run dir> [--step N --prefix p --pairs a:b,...]`（loss 输入/metrics/行校验和/forward-only/完整梯度逐元素）；`bisect_compare.py <run dir> --bad <坏模式参照> --good <好模式参照>`；`gate.py`（容器内 gate）；`gpu_consistency.py`（电池）；证据目录里的 `offline_g4_layers.py`（A8 状态按层分析）、`data_identity.py`（两次运行逐样本 loss 是否逐位相同）。
- 本地数据（不在仓库，体积大）：A8 汇总状态 `/home/michael/work/infra-e3-gpu/b3a8r/out/work/packed/`；RC 各次取回 `/home/michael/work/infra-e3-gpu/a8rc/out-a8rc-rc{2,3,4}-20261001/`、`out-a8rc-rc5a-20261001/`（其中 `work/packed/` 有状态与 trace，`work/arms/*/events.jsonl` 有事件）；A10G 的 `out1a`、`out1b`、`out1c-killed-by-old-watchdog`、RC-1d 在 `out-a8rc-rc1d-20261001/`（目录命名见该目录）。
## 6. 未解问题 / 注意
- 为什么特定主机会得到不同的 logits 梯度（forward 相同）？这是核心未解问题；H1/H2/H3 都还没有被直接检验。
- "坏主机"如何低成本稳定抽到仍不清楚：A8、RC-2 为坏（rank1），RC-3、RC-4、RC-1d 为好，RC-5a 为"整机不同"；gate 逻辑需要改（见 §4.3）。
- 判据本身：同一主机上 DP1 对 DP2 为 6e-10，远低于 0.1%/99.9% 的门槛，门槛本身不是问题；问题是某些主机上同一计算给出不同位。
- 不要改 tasks.md 4.6 的勾选；对 4.6/4.7/4.8 的建议：目前**不应**按"bf16 精度低"收尾；等 §4 的 1–3 步给出算子级根因与因果验证后，再决定是修复（并复测 A8）还是另行预注册判据。
- 约束：不向 radixark/miles、sgl-project/sglang 提 PR；不推 main、不强推；GPU 运行前后向主 agent 报告；任何阶段性结果先写入 plan.md 并提交。
