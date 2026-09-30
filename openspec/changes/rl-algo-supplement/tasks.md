# Tasks

> 阶段一对应第 1–6 组，阶段二对应第 7–12 组。GPU 运行一律遵守 design D2、D9 与 BRIEF：判据事先提交，不放宽判据；失败时先修复再重跑；运行前写明卡型、硬超时和上限；结束后提供无残留证明。每项的费用上限见 design D9，括号内为该项上限。`max_policy_staleness` 固定为 0。INFRA、IMG 负责的文件只以补丁或请求方式交付（design D8）。

## 1. 阶段一门禁与准备（无 GPU）

- [ ] 1.1 编写阶段一 GPU 测试计划 `evidence/phase1-plan.md`：对 4.x、5.x、6.x 的每一项逐条列出判据、最小证据、对照组、卡型、硬超时、费用上限和唯一前缀，并汇总阶段一上限 ≤ $35。验证：计划已提交，并得到**用户确认**（确认记录写入 `progress.md`）。确认之前不得开始第 2 组以后的 apply。
- [ ] 1.2 核实前置状态：trainer v2 补丁（GMPO num/den）是否已合入 integ-decl；1a G3（g3c/g3d）所用的 launcher 回传磁带 harness 是否能直接复用于 1b；当前 pin 与 `FORK_COMMITS`。验证：结论与提交号或行号写入 `progress.md`。补丁未合入时，按 design D5 把 GMPO 标为暂缓。
- [ ] 1.3 记录全量测试基线（`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors -p no:cacheprovider -rfE`），失败 id 存入 `baseline-failures.txt`。验证：文件存在，命令与计数写入 `progress.md`。

## 2. clipfrac 定性与 fork 观测指标（CPU，fork 小提交）

- [ ] 2.1 在 miles-next-venv 中离线复现 pg_clipfrac 疑点：compile 与 eager 两种模式、`eps_clip==eps_clip_high` 与 `≠` 两种情形，同时对照手算结果（design D3）。验证：`evidence/clipfrac-cpu/report.md` 给出“复现并定位根因”或“不能复现”的结论和可重跑脚本。
- [ ] 2.2 如果 2.1 确认是 fork 缺陷：在 `yeto/ports` 上做最小修复，并补 CPU 单测，证明修复前后 pg_loss 逐位相同、clipfrac 与手算一致。验证：fork 单测通过，经独立审查。如果不是缺陷，本项按合法否定结论记录并勾选。
- [ ] 2.3 在 fork 中新增 `dual_clipfrac`（A<0 且 dual 下界生效的 token 比例），附 CPU 单测：默认参数下 loss 逐位不变，构造张量上的比例与手算一致。验证：fork 单测通过，经独立审查。
- [ ] 2.4 在 fork 中记录超采样每次提交的批大小以及丢弃、补采计数，附 CPU 单测。验证：fork 单测通过，经独立审查；指标键名写入 `progress.md`。
- [ ] 2.5 把 2.2–2.4 的提交合并为一次 pin 更新请求交给 IMG：快进 `yeto/ports`、重建镜像、更新 pin；在 ALGO 侧更新 `FORK_COMMITS`，并按 git diff 为 `MILES_DECLARED_PINS` 中的 4 项做迁移论证。验证：新 pin 在 Modal T4 上通过完整 parse_args + validate_parsed_args（$0.5），迁移论证写入 `progress.md`，`tests/test_rl_algorithm_flags_upstream.py` 通过。
- [ ] 2.6 ALGO 侧接入新指标：在 fake 与 adapter 的指标映射中加入 `dual_clipfrac` 与超采样计数；需要改 trainer 取数时，以补丁 `infra-drafts/patches/algo-supp-metrics.patch` 交给 INFRA。验证：单测通过，补丁在临时副本上 apply 后相关测试通过。

## 3. 用户代码类机制永不声明（CPU）

- [ ] 3.1 在 adapter 中加入“不可声明”名单（`corrections:custom`、`plugins`、`losses:custom_loss`、除 vendored Dr.GRPO reducer 以外的任意 `custom_pg_loss_reducer`），并加单测断言它与 `MILES_DECLARED` 不相交，同时断言放行开关在多岛下仍被拒绝。验证：新单测通过，全量测试失败集合与 1.3 基线一致。
- [ ] 3.2 在 `docs/MILES_RL.md` 的 “Declaration policy” 中写入这条规则，并同步修正 rl-algo-capabilities 页第 6 节列出的四处文档与代码不一致。验证：文档中的 dry-run 示例按所写命令执行，输出一致。

## 4. 单卡 G1 生效验证（A10G；前置：1.1 已确认，2.5 新 pin 已就绪）

- [ ] 4.1 在新 pin 上跑同 seed 的默认 GRPO 对照组（$2.0，含对旧 pin 的一次复核），`num_steps_per_rollout=2`，3 轮。验证：通用判据满足，指标 jsonl 存入 `evidence/g1-baseline/`，供 4.2–4.7 与 5.x 共用。
- [ ] 4.2 CISPO、SAPO、GMPO 各跑一次 G1（$4.5），判据采用 loss-variants progress 修订版中的 (a)–(f)，不做修改；如果 1.2 判定 GMPO 暂缓，只跑 CISPO 和 SAPO。验证：每个变体的判据逐项写入 `evidence/g1-variants/results.md`；通过的变体在 `MILES_DECLARED` 中各自单独声明并附证据路径；adapter `check()` 单测接受已声明项，仍拒绝未通过项。
- [ ] 4.3 对 4.2 通过的每个变体，与已声明的 `corrections:tis` 组合做冒烟（$3.0），判据为 (a)(b)(c)，GMPO 另加 (e)。验证：结果写入 `evidence/g1-variants-tis/`；组合不另作声明，结果记为“组合 GPU 冒烟通过”。
- [ ] 4.4 dual_clip：`eps_clip_c=1.01`，每轮 2 步，3 轮（$1.5）。判据：`dual_clipfrac` > 0 且有限，加上通用判据。验证：通过后声明 `features:dual_clip`；如果比例恒为 0，记为合法否定结论，不声明。
- [ ] 4.5 KL：k1、k2、low_var_kl、kl_unbiased 各跑一次（$4.0），放置位置为 `loss`，coef 0.01，带显式 `kl.ref_model`。判据：第 1 轮之后 `kl_loss` 存在、有限、> 0，并与已声明的 k3 运行数值不同。验证：每项单独判定、单独声明，结果写入 `evidence/g1-kl/`。
- [ ] 4.6 opsm_rollout（$1.5）：沿用 1a trigger 的小阈值，每轮 2 步。判据：`opsm_clipfrac` > 0，并且来源记录显示 π_old 取自 rollout logprob。验证：通过后声明 `corrections:opsm_rollout`。
- [ ] 4.7 no_rewards_normalization 与 grpo+whiten 各跑一次（$2.5）。判据：advantage 的均值和方差与对照组不同，并与离线重算结果一致（容差 1e-5）。验证：通过后分别声明；单测断言 grpo+whiten 的变 DP 边仍被 `reshard` 拒绝；`docs/MILES_RL.md` 记录这一限制。

## 5. 既有弱证据复核（A10G）

- [ ] 5.1 over_sampling 强证据（$2.0）：使用 2.4 的指标，配置上能触发动态过滤。判据：至少一轮的提交批大小大于 rollout batch size，并且被过滤组数 > 0。验证：`evidence/g1-over-sampling/results.md`；通过后把声明证据替换为新路径；不通过时撤回声明，并在 `progress.md` 说明。
- [ ] 5.2 clip_higher 在新 pin 上独立重跑（$1.5），沿用 g1j 配置与 g1j 预登记的判据。验证：通过后 `MILES_DECLARED_PINS` 中这一项改记为“在该 pin 上实测”；不通过时撤回声明。
- [ ] 5.3 只有当 2.2 做了修复时才执行（$1.5）：在 GPU 上核对修复后 pg_clipfrac 与离线重算一致，并且 pg_loss 与修复前逐位相同。验证：结果写入 `evidence/g1-clipfrac/`；2.2 为否定结论时，本项记为不适用并勾选。

## 6. 两岛 G3 与阶段一收尾（A10G 1+1）

- [ ] 6.1 1b 的 8.4 G3（$4.0）：两岛 strict-avg，组合配置为 clip-higher + token 级聚合 + overlong（软惩罚与过滤），3 轮，使用 launcher 回传磁带 harness，不带放行参数。验证：两岛算法哈希一致，每轮外层同步后的权重哈希一致，不变量无失败，两岛有效样本数已记录；结果同步回 rl-algo-grpo-knobs 8.4 的完成记录。
- [ ] 6.2 （可选，需用户在 1.1 中确认）CISPO 两岛 strict-avg（$3.5），前提是 4.2 已经声明 CISPO，判据采用 loss-variants 6.3 的判据。验证：结果写入 `evidence/g3-cispo/`；未获确认时记为未执行。
- [ ] 6.3 阶段一收尾：更新 `docs/MILES_RL.md` 的声明清单与证据路径；逐项汇总费用（≤ $35）并提供无残留证明；`progress.md` 按五种状态列出每项结果，并列出仍为“可表达未开放”的机制。验证：全量测试失败集合与基线一致；`openspec validate rl-algo-supplement --strict` 通过。

## 7. 阶段二门禁（无 GPU）

- [ ] 7.1 在阶段一完成之后，编写阶段二 GPU 测试计划 `evidence/phase2-plan.md`（格式同 1.1，上限 ≤ $25，含 MoE）。验证：计划已提交，并得到**用户确认**。
- [ ] 7.2 向 INFRA 提出接口请求：cut manifest 中的 ref 版本字段（rollout 编号 + ref 权重哈希），与 rl-infra-spec 4.1/4.2 对齐。验证：请求写入 `rl-infra-spec/alignment.md` 的追加待批准项，并记录 INFRA 的回复或排期。

## 8. rollout 采样参数进入身份（CPU + A10G）

- [ ] 8.1 在 spec 中加入 `sampling.temperature/top_p/top_k`：默认不输出；把对应参数从 `UNMAPPED_OBJECTIVE_FLAGS` 移出；实现吸收与冲突检测；拒绝 temperature≠1 与 rollout logprob 作为 π_old 的组合。验证：单测覆盖默认哈希不变（R0 与默认 GRPO 快照不改即通过）、吸收、冲突报错和拒绝；upstream parse_args 解析通过。
- [ ] 8.2 G1（$2.5）：temperature 0.7 与 top_p 0.9 各跑一次。判据：SGLang 请求参数中出现所设的值（来源于日志或事件），规范化 spec 包含该值，通用判据满足。验证：通过后声明；在 `docs/MILES_RL.md` 记录字段。

## 9. ref 周期更新（CPU + A10G；依赖 7.2）

- [ ] 9.1 实现 `kl.ref_update_interval` 的字段、翻译与拒绝规则（KL 不生效时拒绝；在需要 cut 的 profile 下，只要 INFRA 尚未提供 ref 版本字段就在启动检查中拒绝），并在事件中记录 ref 版本。需要 driver 或 trainer 配合时以补丁交付。验证：单测通过；补丁在临时副本上 apply 后测试通过。
- [ ] 9.2 G1（$2.0）：间隔为 1，3 轮。判据：每轮的 ref 版本递增，第 2 轮之后的 `kl_loss` 小于同 seed 下间隔为 None 的运行。比较方向事先固定。验证：通过后声明，证据写入 `evidence/g1-ref-update/`。

## 10. loss mask 策略（CPU + A10G）

- [ ] 10.1 核实 Miles 在当前 pin 下的实际 loss mask 行为与可实现的变体，固定 `rollout.loss_mask_policy` 的枚举（默认值等于当前行为）。验证：结论与行号写入 `progress.md`；枚举集合在本项中固定之后不再扩展。
- [ ] 10.2 实现字段、翻译与拒绝规则（未知取值时列出允许值）。验证：单测确认默认哈希不变，非默认值进入哈希。
- [ ] 10.3 G1（$1.5）：一个非默认取值。判据：训练侧的 masked token 数与对照组不同，并与离线按策略重算的结果一致。验证：通过后声明。

## 11. MoE routing replay（CPU + L40S 或 A100）

- [ ] 11.1 CPU 核实 MoE 候选（design D10）：OLMoE-1B-7B 是否满足 Megatron bridge 支持、SGLang 返回 `routed_experts`、可挂 LoRA；不满足时改用 Qwen1.5-MoE-A2.7B。验证：结论写入 `progress.md`，并确定卡型与费用估算。
- [ ] 11.2 把 `moe.rollout_routing_replay/routing_replay` 移入 spec 与哈希：吸收旧的 agent 开关，非 MoE 模型时拒绝，从 `UNMAPPED_OBJECTIVE_FLAGS` 移出 `--use-routing-replay`；`cli/launcher/learner` 的接线改动以补丁交主 agent 指定的负责人。验证：单测覆盖默认哈希不变、旧开关被吸收、稠密模型被拒；在 `docs/MILES_RL.md` 写入迁移说明。
- [ ] 11.3 GPU（两次运行合计 $12）：分别在开启与关闭 routing replay 下运行，其余配置相同，3 轮。判据：开启时有训练侧使用 rollout 路由的证据，且 `train_rollout_kl` 不大于关闭时的值，比较方向事先固定。验证：通过后声明；证据写入 `evidence/g1-moe-replay/`。

## 12. RLOO 与阶段二收尾

- [ ] 12.1 在 yeto reward 分派器中实现 RLOO 阶段：组大小 ≥2，配套 `std_normalization=false`，身份独立；在文档中说明它与 Dr.GRPO 只差常数缩放。验证：单测覆盖 [1,0,0,1]→[2/3,−1/3,−1/3,2/3]、组大小为 1 时被拒、哈希与 Dr.GRPO 不同。
- [ ] 12.2 G1（$1.5）。判据：分派器输出的 advantage 与离线重算一致（容差 1e-6），通用判据满足。验证：通过后声明。
- [ ] 12.3 阶段二收尾：逐项汇总费用（≤ $25）并提供无残留证明；`progress.md` 按五种状态列出结果；在 `progress.md` 的后续清单中列出 OPD、critic/PPO/VAPO、异步/staleness>0/partial rollout、GiGPO 等后续 change。验证：全量测试失败集合与基线一致；`openspec validate rl-algo-supplement --strict` 通过。
