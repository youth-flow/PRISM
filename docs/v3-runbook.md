# V3 执行手册

V3 是独立的 `hh_finite_pool_v3` 协议，配置为 `configs/experiment.v3.yaml`，输出默认 `runs/olmo1b-hh-v3`。本手册是执行入口说明；工程测试通过不代表完整 HH 数据、真实 GPU 显存或正式学习效果已经验收。

2026-11-17用户已授权直接开始V3全池准备与Design；三λ分布求出后必须暂停并汇报，由用户决定唯一N或是否更换求解算法。后半段入口已准备，不因代码存在就自动放行。

本轮只有 Empirical／PRISM、seed11、**一档共同偏好预算 N**。N 在三档 Design 稀疏性报告之后由用户决定。配置中的预算保持 null；只有显式记录用户决定的 `freeze-study` 能放行抽样。保持原目标、simplex 及求解器，不加入覆盖 cap、熵项或 uniform floor。全池 HH 人工回答的 teacher-forcing 特征与 log-prob 构建双候选 loglinear 代理 K；不为 K 另外生成4096×4回答。

## 本机检查

以下命令不提交 Slurm 作业，不下载模型，不调用付费 API：

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:PYTHONUTF8 = "1"
python -m prism.v3 validate --config configs/experiment.v3.yaml
python -m prism.v3 plan --config configs/experiment.v3.yaml
python -m pytest -q tests/test_v3_evaluation.py
```

计算命令使用 `--execute`，但该标志不会开启付费 Judge。模型计算在已验收的集群计算节点完成。原 V2 配置、权重、产物、账本及未知费用预留均不改动。

## 六阶段

下例的 `$cfg` 在集群 Bash 中应设为 online1 下冻结 YAML 的绝对路径。每个成功阶段生成 `receipts/<stage>.json`；下游逐层核对配置、输入及输出 SHA256。失败／部分产物不算阶段完成。

1. **数据与共享资产**：`download-data` 下载固定 revision 的公开 HH 文件；`prepare-data --source-manifest ...` 离线导入并保留源哈希、排除／重复／冲突统计和按上下文隔离的划分。`extract-features --shards 32 --shard-index i` 分片前向，`merge-features --shards 32` 全覆盖归并。`prepare-shared` 生成共用定标／critic预热轨迹。人工回答不截断后沿用标签；不能把160800原始行直接冒称全部验收通过。
2. **Design**：`geometry` 构建固定 v／b／p／K，`design` 分别求 λ=.001/.01/.1，`design-report` 输出收敛、支持、ESS、累计质量及相对经验目标。相对 FW gap 必须通过原 1e-4；未收敛候选不能用于正式选样。ESS不是不放回预算的硬上限，微小正权也不冒称有效覆盖。
3. **选样与 Reward**：收到用户 N 决定后，执行 `freeze-study --budget N --user-budget-confirmed`，助手选择已验收的 Design λ。随后 `sample`、`reward`、`normalize-reward`。抽样按剩余权重不放回，零权不补齐；预算超过正权支持直接失败。Reward 只训练2048维线性头，三档 L2 由双方共同开发集平均 NLL选一个公共值。归一化在固定4096条 SFT回答上分别计算 μ／σ，随后冻结。
4. **PPO**：`policy --method empirical`／`prism`，使用该同一 N 的 Reward。先完成2048 episodes工程节点、8192开发节点，再按统一配方训练到65536。独立 critic LoRA与冻结 Reward 参数隔离，critic-only预热另记128步；正式应有512轮／4096个真实 AdamW更新。保存未更新概率路径、真实梯度裁剪、critic泛化、固定训练和开发探针。仅 loss／进度条完成不算学习效果验收。
5. **评价生成与 KL**：对 `sft`／`empirical`／`prism` 运行 `evaluate-generate --method METHOD --shards S --shard-index i`，再 `merge-evaluation --method METHOD --shards S`。各分片保持全局索引、prompt级随机种子、覆盖和顺序；SFT回答共用。三个模型均T=1、cap512。保存长度／EOS／截断及 actor−SFT 序列 log-ratio；Monte Carlo KL负值保留。
6. **Luna与汇总**：`judge-export` 生成一个不可变本机包。Luna medium、AB／BA、TIE=.5，三项比较×500题×两顺序=3000次**有效判断**。本机执行后，将结果 JSONL 和相邻 `.billing.json` 一同带回；`judge-import --results ...`、`aggregate`、`audit --complete` 完成独立核对。原始请求数和有效判断数分别报告，网络未知费用不作为零费用。

各计算入口均需 `--config "$cfg" --execute`；`freeze-study` 的 N 不应由调度脚本猜测。最终500题不用于正则、PPO选参或 checkpoint选择。主要比较是 PRISM 对 Empirical，配套为两者分别对 SFT；胜率按每题 AB／BA均值汇总，bootstrap以prompt为单位，并报告胜平负、换序分歧、helpful／harmless分项。单seed区间不代表跨训练seed不确定性。

## Slurm 与不可变源码

```bash
bash scripts/yancheng/submit_v3.sh --dry-run design --config "$cfg" --execute
PRISM_ARRAY=0-31%8 bash scripts/yancheng/submit_v3.sh --dry-run extract-features --config "$cfg" --shards 32 --execute
```

默认 `PRISM_ALLOW_SUBMIT=0`。正式提交前必须已有检查过的源码 commit、运行镜像 SHA 和资源分区；单次设置 `PRISM_ALLOW_SUBMIT=1`。脚本在提交时将已提交源码固定到 `PRISM_WORK/releases/<commit>`，排队作业验证该快照及文件清单，不依赖后来变化的仓库 HEAD。所有文件必须位于 online1，登录节点仅提交、归档和只读调度查询，不跑模型。

每任务一张GPU；特征／评价用独立分片扩展并行，不能将单进程 PPO直接换成多卡。提交器按数组任务展开数核对个人运行＋排队合计不超过50，不提高优先级、不占他人reservation、不使用exclusive。GPU分区逐次指定，避免站点拒绝多分区请求。

默认资源：特征／共享生成／评价为2CPU／16G／1GPU；PPO为8CPU／64G／1GPU，优先A10080GB；Design／几何为16CPU／24G。q_amd_share 已验收CPU配置约每核心1.5GiB，16CPU／64G不是已验证合法配置。全 HH 峰值内存仍须基准；需要更大主存时，用站点支持的配套核心／内存数（例如显式48CPU／72G），或先优化分块内存。其他CPU分区按其实际规范，不把此限制套到GPU节点主存。

PPO默认申请24小时，并非声称完整65536 episodes可在24小时结束。每2048 episodes独立做耐久保存，不强制每次执行昂贵探针。若分区最长时限不覆盖完整训练，使用显式 `--stop-after-episodes` 检查点和下一作业 `--resume` 接续；配置／数据／Reward／优化器／RNG身份必须一致。检查点已存而probe中断时，恢复先补齐同一检查点的probe，保持训练RNG。总工期由真实新配方吞吐更新，不能照搬 V2 的几小时。

`cpu-smoke` 在计算CPU节点运行完整测试；`model-smoke` 在GPU计算节点运行真实小模型路径。它们不等于正式模型／全量学习效果验收。所有 Judge请求只在本机；集群执行器不转发 OpenAI key，GPU不等待API。

## 本机 Luna：独立账本与恢复

导出的 `judge/packet.json` 可以移至本机，无需在本机安装模型权重。先检查，不发送请求：

```powershell
scripts/windows/run_v3_judge.ps1 -Packet C:/private/v3/packet.json -Results C:/private/v3/results.jsonl -Ledger C:/private/v3/v3-eval.sqlite
```

V3 的美元上限尚未由用户指定，因此不能直接执行付费任务。上限明确后，显式传 `-BudgetUsd <已授权金额> -Execute`，密钥通过本机环境变量或仓库外 `-KeyFile` 提供，绝不打印密钥。默认8并发，可显式1–16；模型／reasoning／tier等有效参数由包固定，不能静默从Flex改标准或反向切换。

新账本文件名要求 `v3-*.sqlite`，旁边的 `.v3.json` 绑定 packet／config／不可变金额。已有无归属账本、V2账本、不同包或不同金额均拒绝，不重置消费。请求派发前预留保守上限，响应先落盘再结算；成功已记录任务不会重复请求，已返回但JSONL丢失的响应从同一账本恢复。未知请求继续占预留，不擅自释放。中断后使用同一 packet／results／ledger／cap重新执行，仍保留原3次语义尝试，不增加额外评价网络槽位。

每次退出都会导出 `results.jsonl.billing.json`，包含逐dispatch的原始返回、usage、实际tier、已知费用和未知预留。导入时重验包、配置、生成收据、方向／seed／顺序、原始文本、usage及预留金额；不把usage估算说成供应商实际账单。公开仓库不保存结果、账本、模型、原始HH对话或凭据。
