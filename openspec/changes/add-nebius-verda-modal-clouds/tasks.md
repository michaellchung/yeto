## 1. 信号注册表与 RunPod 迁移（行为不变）

- [x] 1.1 在 `yeto/shape/providers.py` 定义 `CloudSignals` 协议（`name`、`available()`、`offerings(regions)`、`scores(asks)`）和注册表字典；验证 `python -c "from yeto.shape.providers import CLOUD_SIGNALS"` 可导入且注册表为空字典以外的合法结构
- [x] 1.2 把 `RunPodProviders` 改为实现该协议并登记为 `runpod`，`offerings` 仍走 sky 目录；验证 `tests/test_shape_providers.py` 全部通过且新增"RunPod 目录缺 H200 时输出告警"的测试通过
- [x] 1.3 让 `build_shape` 遍历注册表，去掉 `runpod_providers` 具名参数（保留测试注入口 `signals: dict | None`）；验证 `tests/test_shape_plan.py` 全部通过，且用同一组 fake 输入的 `--clouds aws,runpod` 方案与改动前逐字段相同（在测试里用快照断言）
- [x] 1.4 解除 AWS 凭据硬依赖：只有 `clouds` 含 `aws` 时才检查 AWS 凭据；`--clouds` 缺省为 `aws` 加所有 `available()` 为真的云；指定的云缺凭据时报错并列出期望路径；验证新增测试覆盖"只有 nebius 凭据 + `--clouds nebius`"成功和"`--clouds aws,verda` 缺 Verda 凭据"报错两个场景
- [x] 1.5 信号契约的边界测试：伪分 0 使形状不可启动且不产生"信号缺失"告警，None 产生告警并按假定分或严格模式处理；一朵云的 `scores` 抛异常时其余云的候选和方案不受影响；凭据检测为假的云在整次规划中零网络请求（用计数 mock 断言）；验证 `tests/test_shape_plan.py` 新增这三个用例并通过

## 2. `--regions` 多云语法

- [x] 2.1 新增区域解析函数：`cloud:region` 拆分、无前缀默认 `aws`、`all` 和 `cloud:all`；返回 `dict[cloud, set[str] | ALL]`；验证新增单元测试覆盖混合写法、纯旧写法、`all`、未知云名报错四个用例
- [x] 2.2 `catalog._to_rows` 和 `build_shape` 按解析结果对每朵云过滤，AWS 语义与改动前一致；验证 `test_regions_all_searches_every_catalog_region` 与新增的"nebius:eu-north1 只留该区域"测试通过
- [x] 2.3 未知区域报错时列出该云目录中的区域；`launch_argv` 输出带云前缀和原生区域的 `--gpu` 条目；验证 `test_launch_argv_and_render_share_the_command` 更新后通过，且 `verda:8xh100@FIN-03` 能被 `gpu_spec.parse_gpu_spec` 原样解析
- [x] 2.4 更新 `yeto/cli.py` 的 `--regions` 和 `--clouds` help 文本与 README 的 `--regions` 段；验证 `yeto shape --help` 输出包含 `cloud:region` 示例

## 3. Nebius 信号

- [x] 3.1 `NebiusSignals`：凭据检测读 `~/.nebius/NEBIUS_IAM_TOKEN.txt` 和 `NEBIUS_TENANT_ID.txt`；GPU 映射表（H100/H200/B200/L40S 到 Nebius platform 名）；`offerings` 走 sky 的 nebius 目录并保留 SpotPrice；验证 mock sky 目录的单元测试通过，且映射表的每个 sky GPU 名都在 `catalog.PEAK_TFLOPS_BF16` 中
- [x] 3.2 `scores` 调 Capacity API resource-advice，按 (region, platform, preset) 把 preemptible 或 on-demand 容量映射为 9/6/0，接口失败为 None，缓存 900 秒；验证用假 HTTP 响应的测试覆盖有货、货不足、无货、接口错误四种返回
- [x] 3.3 实时价覆盖：Capacity 或定价接口返回价格时覆盖目录 spot 价并在 `Offering` 上标记来源；规划输出注明来源；验证测试断言覆盖后的价格被用于预算计算且 render 输出含来源标记
- [x] 3.4 `rdma_capable(cloud, instance_type)` 取代 `efa_capable`，Nebius 8 卡 SXM 机型返回 True；验证 `tests/test_shape_catalog.py` 新增用例通过且 AWS 现有用例不变
- [x] 3.5 Nebius 多区域项目校验：从 sky 配置读取每个区域的 project id，fleet 涉及的区域缺 project id 时在提交前报错；验证新增测试用两个区域、一个缺 project 的配置触发报错并列出缺失区域
- [x] 3.6 `pyproject.toml` launcher extra 改为 `skypilot[aws,runpod,nebius,verda]`；验证在 WSL 中 `pip install -e ".[launcher]"` 成功且 `sky check` 列出 Nebius 和 Verda 行

## 4. Verda 信号

- [x] 4.1 `VerdaSignals`：凭据检测读 `~/.verda/config.json` 或 `VERDA_CLIENT_ID/SECRET`；`offerings` 调公开 `/v1/instance-types` 建目录（机型名到 sky GPU 名的映射，每机 GPU 数从机型名解析，spot 价取接口值），不调 sky 目录；验证用今天抓取的 70 条真实响应做 fixture 的测试通过，且 8 卡 H100/H200/B200 都进入 offerings
- [x] 4.2 location 处理：优先调 location 接口，失败则用固定表 FIN-01/FIN-02/FIN-03/ICL-01 并告警；对每个 (机型, location) 生成一条 offering；验证测试覆盖接口成功与回退固定表两种路径
- [x] 4.3 `scores` 调鉴权的可用性接口，可用为 9、不可用为 0、失败为 None。实现时发现 `GET /v1/instance-availability?is_spot=true` 是批量接口（一次返回每个 location 有货的机型列表），所以改为一次批量请求加 15 分钟缓存，原设计的"只查价格前 24 的形状"限制作废；验证用假响应的测试覆盖有货、无货、location 缺失、接口失败四种返回，以及多个 ask 只发一次请求
- [x] 4.4 Verda 单节点限制：`build_shape` 对 Verda 不生成多节点岛，模型需要多机时告警；验证 `test_multi_node_island_when_model_demands` 的 Verda 变体断言无多节点方案且告警存在

## 5. Modal 信号与远程 learner 运行器

- [x] 5.1 `ModalSignals`：凭据检测读 `~/.modal.toml` 或 `MODAL_TOKEN_ID/SECRET`；静态价格表（GPU 单价 + CPU + 内存，记录日期，区域系数按 2026-09-23 官方页：宽 1.15 窄 1.75）；`scores` 恒为固定值不联网；多容器非整机形状被排除；表日期超过 90 天告警；验证单元测试覆盖价格合成、区域系数、非整机排除、过期告警
- [x] 5.2 新建 `yeto/modal_runner.py`：定义 Modal app 与函数，参数为 learner id、syncer 地址、训练模式、GPU 与数量、节点数、镜像摘要；单容器直接 torchrun，多容器用 `clustered(size, rdma=True)` 并从 cluster info 取 rank 和主节点 IP；SFT 镜像从 requirements 构建，RL 镜像 `from_registry` 按摘要，摘要不一致时启动前报错；验证不依赖 Modal 凭据的单元测试覆盖 torchrun 命令拼装、环境变量集合、镜像摘要校验
- [ ] 5.3 手动接入模式：`python -m yeto.modal_runner --learner-id N --syncer-addr HOST:PORT ...` 可独立运行；验证在有 Modal 凭据的机器上以 `--external-learners 1` 起一个 1 卡 SFT fleet，手动接入 Modal 岛后 syncer 事件 tape 中出现该 learner id 并完成至少 2 轮同步（记录到 docs/CLOUDS.md）
- [x] 5.4 launcher 路由：`gpu_spec` 解析出 `modal` 云时 launcher 不构造 `sky.Task`，在 syncer 就绪后 spawn Modal 函数；`--gpu modal:2x4xh100` 在启动前报错；syncer 地址非公网且未提供 `--syncer-public-addr` 时在启动前报错；验证 `tests/test_launch_auto.py` 新增用例覆盖路由、非整机报错、地址校验，且 mock 掉 Modal SDK
- [x] 5.5 生命周期：`yeto down` 取消该 fleet 的 Modal 函数；容器被抢占时以同一 learner id 重试，重试耗尽向 syncer 报告；`yeto status` / `yeto logs` 列出并读取 Modal 岛；验证 `tests/test_runs_cli.py` 新增用例覆盖 status/logs/down 对 Modal 岛的处理（mock SDK）
- [x] 5.6 RL 岛断点：`--spot` 语义在 Modal 上映射为 Modal Volume 挂到 `--rl-completed-groups-path`；验证单元测试断言 Volume 挂载路径与 sky 岛的存储挂载路径一致
- [x] 5.7 Modal 区域语义：`--regions` 未提及 modal 时 `launch_argv` 生成的 Modal 条目无 `@region`、运行器调用无 `region=` 参数、价格为基础价；提及时价格乘系数且 render 输出注明系数来源；验证单元测试覆盖未指定与指定两种路径
- [x] 5.8 抢占重试：mock Modal 函数首次抛出抢占异常，断言运行器以同一 learner id 重新 spawn，重试次数耗尽后向 syncer 发送该岛退出事件；验证 `tests/test_launch_auto.py` 新增用例通过

## 6. RL 岛定价钩子

- [x] 6.1 `build_shape` 新增 `island_shape` 输入（每岛 GPU 数、是否单节点、是否需要容器镜像、spot 是否需要对象存储）；cli 从 `--training-mode rl`、`--parameter-mode`、`--rollout-num-gpus`、actor 卡数推出；验证新增测试覆盖分卡模式（8 卡单节点）与共卡模式（允许多节点）的候选过滤
- [x] 6.2 引入 `VERIFIED_DOCKER_IMAGE_CLOUDS`、`VERIFIED_SPOT_STORAGE_CLOUDS`、`RDMA_CLOUDS` 常量（初值：aws 和 runpod 进镜像集合，aws 进存储集合），规划器据此排除 RL 候选并告警，未验证存储的云 RL 岛按按需价；验证测试覆盖"未验证镜像的云被排除并告警"和"未验证存储的云按按需价"
- [x] 6.3 `yeto shape --training-mode rl` 的 render 和 JSON 输出含岛形状和排除原因；验证 `test_json_dict_shape` 的 RL 变体通过

## 7. head 凭据挂载

- [x] 7.1 凭据路径表（AWS/GCP/RunPod/Nebius/Verda/Modal）；`_make_head_task` 只挂载 fleet 涉及的云，缺失时提交前报错并给出期望路径；AWS 不在 fleet 中时不再打 `~/.aws` 警告；验证 `tests/test_head_mode.py` 新增用例覆盖"只挂用到的云"、"缺 Nebius 凭据报错"、"fleet 无 AWS 时无警告"
- [x] 7.2 断言 learner 岛（sky 与 Modal）的 file_mounts 和 envs 不含任何云凭据；验证新增测试遍历 SFT、RL、Modal 三类岛的任务定义断言无凭据路径与凭据环境变量

## 8. 真机验证与文档（需凭据，结论回填常量）

- [ ] 8.1 Nebius：在 eu-north1 起 1 卡 H100 SFT 岛完成 2 轮同步；验证 syncer tape 有该岛记录，结果记入 docs/CLOUDS.md
- [ ] 8.2 Nebius RL 云敏感项：`image_id: docker:` 启动 RL 岛、`--spot` 的 `sky.Storage` 挂载、2 节点 `network_tier=best` 的 NCCL 日志显示 IB 传输；验证三项各自通过或失败，结论写入 6.2 的三个常量与 docs/CLOUDS.md
- [ ] 8.3 Verda：在 FIN-03 起 1 卡 H100 SFT 岛完成 2 轮同步；再验证 `image_id: docker:` 启动 RL 岛；验证结论写入常量与 docs/CLOUDS.md
- [ ] 8.4 Modal：8 卡 H100 单容器 RL 岛（CyberGym 外部 API 环境）完成 2 轮同步；记录 WAN 同步耗时；验证 docs/CLOUDS.md 记录耗时与是否需要调大 `pipeline`
- [ ] 8.5 用真实凭据各跑一次 `yeto shape --clouds nebius`、`--clouds verda`、`--clouds modal`，对照今天的目录价核对输出；验证 docs/CLOUDS.md 附三份输出摘要
- [ ] 8.7 混合 fleet 真机验收：`--gpu aws:1xl4@us-east-1,modal:1xh100` 一次启动，两个岛在 syncer tape 中同轮次完成至少 2 轮同步，`yeto down` 后 `sky status` 与 Modal 控制台均无残留；验证 tape 摘要和两侧清理截图记入 docs/CLOUDS.md
- [x] 8.6 新建 docs/CLOUDS.md（每家云的凭据位置、区域或位置、已验证限制、Windows 需 WSL 的说明）并更新 README 的 `--gpu` / `--regions` / `--clouds` 段；验证 `scripts/check_name_parity.py`（若适用）与文档链接检查通过

## 9. 整体验证

- [ ] 9.1 全量 `pytest tests/` 通过，无需任何云凭据。（2026-09-23 在 WSL 跑过：本 change 涉及的 11 个测试文件全部通过；全量为 1727 通过 / 34 失败 / 26 错误，其中 23 失败 + 26 错误在改动前的 HEAD 基线上同样存在（缺 `miles` 包、缺 Rust syncer 二进制），另外 11 个失败是 Windows checkout 的 CRLF 让 `scripts/*.sh` 里的 `set -o pipefail` 报错，在 LF 检出的同一份代码上通过。本 change 没有引入新的失败；要让此项打勾需在 LF 检出、装了 miles 和 syncer 二进制的环境重跑。）
- [ ] 9.2 `yeto shape --clouds aws,runpod --regions us-east-1,us-east-2`（旧写法）在有 AWS 凭据的机器上输出与变更前一致的方案（对照变更前保存的一份 JSON 输出）
- [x] 9.3 场景对照表：在 change 目录新建 `acceptance.md`，列出 5 个 spec 的全部 34 个场景，每个场景对应至少一个测试函数名或任务组 8 的真机记录条目；验证表中没有空行，且列出的每个测试函数名在 `tests/` 中 grep 得到
