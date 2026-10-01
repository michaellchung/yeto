# A8 G4 根因调查 交接（INFRA-E3，2026-10-01 07:55Z 更新；到 RC-7/7b 为止；接手者请先读完本文件**§8**（最新，覆盖 §1/§4/§7 中与之冲突的说法）再读 §1–§7 与 plan.md §12–§13）

分支 `infra-e3`（worktree `/home/michael/work/infra-e3`），远端 `origin/infra-e3`。证据目录：`openspec/changes/rl-infra-spec/evidence/infra-e3/a8-rootcause/`（下文路径都相对于它，除非写绝对路径）。**tasks.md 的 4.6 勾选状态未改，4.7/4.8 未降级**，progress.md 仅追加记录。用户要求：不得按"合法否定结论"勾 4.6，必须查明根因属于 (1) 正常跨 DP 数值差异 (2) 切换实现缺陷 (3) 验收标准不合理；"bf16 精度低""loss 相同"不算根因。GPU 规则：只用 Modal（或经批准的 Nebius），固定型号并断言；本任务总上限 **$30**（用户已放宽），单次 ≤$8；每次运行前先在 `/home/michael/work/infra-drafts/gpu-spend.md` 记一行、把计划写进 plan.md 并提交、并向主 agent 报告；运行后报告结果；用完核对 `modal app list`（只管 `a8rc-` 前缀）。

## 1. 一句话现状
- **切换实现没有缺陷**（逐位证据，三次运行）；**DP1 对 DP2 的数学/归约没有问题**（同一主机上两者在 1e-10 量级一致）；A8 的 G4 失败来自**依赖具体 Modal 主机/GPU 实例的、确定性的反传数值差异，起点在 loss→logits 的梯度（forward 逐位相同）**。为什么某些主机会这样、具体是哪个算子/代码路径，**尚未查明**。因此目前既不是 (1) 正常跨 DP 数值差异，也不是 (2) 切换缺陷，(3) 判据不合理也没有证据支持——更像第四类：环境/主机相关的数值不一致（需要把它收敛到具体算子并做因果验证）。
- 费用：（截至 RC-5a）本任务累计 ≈ $20.2；**最新累计见 §8（≈$28.0，上限 $30，已无预算再起运行）**。逐次见 §3。无任何在跑的云资源（核对：`modal app list` 里 a8rc-rc1/rc1b/rc1c/rc1d/rc2/rc3/rc4/rc5a 均 stopped；无 watchdog 进程）。

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

## 7. RC-6 / RC-6b（后半程接手，2026-10-01 06:54–07:07Z；最新状态以本节为准，覆盖 §4 的"下一步"）
- **费用**：RC-6 ≈$0.8，RC-6b ≈$1.8，本任务累计 ≈ **$22.8**（上限 $30；本接手段剩余 ≈ $7.2，$10 的接手上限里已用 ≈ $2.6）。所有 `a8rc-` app（含 k1–k4）stopped，无 watchdog/launcher 进程；台账已追加。
- **静态核查（plan §10.0）**：Miles `compute_log_probs` 无条件走 Megatron `fused_vocab_parallel_cross_entropy`（`jit_fuser = torch.compile`，无人禁用；Megatron 的 deterministic-mode 对它无效）；torch 2.13 默认 pointwise 做 ≥2 个 config 的计时 autotune；Inductor/Triton 缓存默认在同一容器内被两个 rank 与所有 arm 共享。
- **因果检验（plan §10–§11，`tools/probes/e3_reshard/kernel_probe*.py`，`tests/test_rl_e3_kernel_probe.py`）**：每个 rank 一个独立进程、冷编译、两个进程首个形状不同；链 = fp32 logits → fused CE → `compute_policy_loss` → backward；4 台 H100 主机（8 张 GPU）× 变体 (a) 现状 ×3、(b) 关闭全部 torch.compile、(c1) 关 autotune ×2、(c2) 固定缓存：每个样本的 loss、logp、grad_output、logits 梯度的 sha256 **全部逐位相同**（(a)=(c1)=(c2)；(b) 自身一致但与编译路径不同——编译与 eager 本来就不是同一计算）。autotune 选择确实会变（softmax 除法 kernel 8 对 4 warps）而数值不变；reduction kernel 单一 config。⇒ **H1（计时 autotune / 跨主机选出不同数值 kernel）被排除**（在这条链上），**根因仍未确认**，4.6/4.7/4.8 状态不变。
- **局限**：孤立单链、合成输入，不含整网环境（显存布局、上游 kernel 产出的输入、Ray 多进程），也不能与坏模式校验和直接比（真实 logits 没存，只有校验和与少数完整 logits 梯度张量）。所以"这条链主机无关"成立，"整网里为什么某些主机的 loss→logits 梯度不同"没有回答。
- **剩余可能性与建议的下一步（需要新登记，约 $4–5，我的剩余预算够做 1 轮）**：差异的首个已测位置是 logits 梯度，但**CE 反传的输入没有被单独记录过**。建议在真实训练里给 `fused_cross_entropy.calculate_gradients` 打 monkeypatch 探针（在 `rc_trace.py` 的 `install_trace` 里）：记录 (softmax/exp_logits, grad_output, target_mask, masked_target_1d) 的 sha256 和输出的 sha256，并在同一进程里用同一输入再算一次（编译版 + eager 版）比较；再用 `gate.py` 的"与参照 DP1 的 exp_avg 差"在新容器里抽主机（RC-2/RC-5a 说明约 3/5 的主机与参照不同，命中率不低）。判读：同输入重算与原输出逐位相同且 exp_logits 或 grad_output 在"坏/不同"主机上与"参照一致"主机上不同 ⇒ 差异在上游（forward 的 softmax 张量或 loss 反传），不在 CE 反传 kernel；同输入重算不同 ⇒ kernel/硬件非确定性。同时记录 forward 保存的 `exp_logits`（softmax）校验和——它在 loss 逐位相同时仍可能不同（loss 不依赖除法之后的 exp_logits），这是目前唯一"前向张量没被比较过"的缺口。
- 修复尚未落地（根因未确认，没有可修的对象）；没有推 fork、没有改 pin。若主 agent 想先做保守缓解，唯一已有因果依据的开关是"编译 vs eager 是不同计算"这一点，但**没有证据表明关闭 compile 会消除跨主机差异**，不建议在无验证时当作修复写进 profile。

## 8. 第三轮：RC-7 / RC-7b + 两项离线核查（2026-10-01 07:21–07:55Z；最新状态以本节为准）
- **费用**：RC-7（3 台）≈$3.1，RC-7b（2 台）≈$2.1，本轮 ≈ $5.2（上限 $7）；本任务累计 ≈ **$28.0**（上限 $30，剩余 ≈ $2，不足以再做有意义的两台抽样，只够 1 台 ≈$1.2 的单机实验）。所有 `a8rc-` app（ce1–3、cf1–2 及此前全部）均 stopped，无 watchdog/launcher 进程。台账 `/home/michael/work/infra-drafts/gpu-spend.md` 已追加（每次启动前一行、回填一行）。
- **做了什么**（计划与判读规则在运行前提交：plan §12 提交 60011da，§13 提交 f02394a）：在真实训练（gate 臂 DP2 从 0 训 2 步）里给 Megatron 融合 CE 的四个 `jit_fuser` 函数打只读探针（`rc_trace.install_trace(ce_probe=True)`，`tests/test_rl_e3_ce_probe.py`，`ce_compare.py`）：前向各阶段与 `calculate_gradients` 的输入/输出校验和；反传前克隆输入，同一进程内用同一输入再用编译函数、eager 各重算一次并比较；每 rank 保存 1 组完整输入输出。批 1：3 台新 H100!:2；批 2（RC-7b）：2 台，A 臂配置（`cut_at_step2`、`last_step=3`，同 RC-4 的 eA2）+ 同探针。
- **结果**（plan §12.5、§13.2）：5 台主机的 gate 都是 1.255e-2，exp_avg 与 RC-5a 的 `gA2_s2` **逐位相同**（"类 X"，加上 RC-5a 共 6 台、4 种 vbios、2 种 CPU 型号）；探针对该类无影响。X 模式内部是确定的：5 台之间 CE 前向 10 个阶段、反传 5 个字段、保存的完整张量（softmax_in/grad_output/out）逐位相同；同机同输入重算（编译、eager）与原输出逐位相同。按预登记规则 6：**没有 state 不同的主机对，未能在 CE 层面定位 good/bad 分歧阶段**。
- **配置因素被排除**（RC-7b）：A 臂配置在当前主机上同样是类 X；离线核查（plan §13.1）：所有 arm 的 Miles argv/超参逐项相同（`--num-rollout 8`、lr 1e-5 linear、decay-iters 8…），实际应用的 lr 相同（1e-5/8.75e-6/7.5e-6），scheduler/hyper/step/counters 摘要相同；步 1 的 grad_norm 全 0，步 2 的 grad_norm 三类各不同。
- **离线核查 II 的新发现**（plan §13.3，`rc7-ce/inputs_cmp.txt`、`logp.txt`）：ref 类(RC-3 dA1) 对 类 X(RC-5a gA1d) 的步 2：token/mask/advantage/长度/pg loss/所有模块前向输出（含 logits，训练与 forward-only 两个通道）**逐位相同**；但 **old log-prob（`log_probs`）在 16/16 微批不同**（2609 个 token 里 1455 个，最大绝对差 9.5e-7，几个 ulp），因此差异的位置是 **logits→log-prob 的融合 CE 前向**本身，而不是 CE 反传 kernel；这修正了此前"前向逐位相同"的说法（逐位相同的是 logits 和 pg loss）。另外 B1 类（A8/RC-2）的 rank1 与 X 类的 rank1 全部记录逐位相同，rank0 与 ref 相同——**模式 R/X 是按进程（rank）分配的**，不是按主机、配置或数据；时间上 A8/RC-2/3/4 以 R 为主，RC-5a 起 6 台全 X。s1 状态没有 dump，无法直接比较（间接证据见 plan §13.3）。
- **结论（如实）**：**主机/进程相关的数值差异，具体机制未查明；切换实现已证实无缺陷**；差异的最早已知位置收敛到融合 CE 前向（logits 逐位相同、log-prob 不同），不是反传 kernel、不是数据、不是超参、不是臂配置、不是 H1（autotune，RC-6）。**R 模式从未在受控条件下复现**，所以 RC-6 的孤立链结论只对 X 模式成立。
- **剩余方向与最小下一步**：见 plan §13.4（Triton kernel 对齐/向量化特化、编译缓存共享、时期性主机环境变化）。建议的最小实验（≈$1，单机）：在 X 模式主机上用受控的 pointer/size 对齐（如 logits 切片偏移）看能否制造出 R 模式，并取出 `triton_red_fused_..._sum` 的 kernel 元信息比对；需要先预注册并经用户确认，预算已几乎用尽。
- **对 4.6/4.7/4.8 的建议**：不改 tasks.md 勾选，不降级 4.7/4.8（状态由用户决策）；不应以"bf16 精度低"收尾；A8 的 G4 比较在 R 模式的 DP1 与 X 模式的 rank 之间不对等（同一代码给出两种数值模式），若要收尾需要用户决定：(i) 继续查明 R/X 选择机制并修复/钉死（例如固定到一种模式后重跑 A8 的 G4），或 (ii) 另行预注册"同模式基线"的判据（同模式内 DP1 对 DP2 在 1e-10，远低于 0.1%/99.9% 门槛），不得为通过而放宽现有判据。
- 其它：本轮提交 60011da、f02394a、981d8e0 及本节提交；未推 main、未强推、未提 PR、未改 pin；本地大数据在 `/home/michael/work/infra-e3-gpu/a8rc/out-a8rc-{ce1,ce2,ce3,cf1,cf2}-20261001/`（`work/packed/` 含 trace 与完整张量；`go7.sh`/`go7b.sh` 为本轮启动脚本，快照 `yeto-rc7`、`yeto-rc7b`）。
