# V3：HH人工偏好全池设计与单预算RLHF

以2026-11-17用户最新指令为准：分支v3，配置configs/experiment.v3.yaml，输出runs/olmo1b-hh-v3。
已授权直接启动全池准备与Design；三档PRISM求出后暂停汇报，用户决定唯一预算N或是否更换求解算法。

## 数据、模型与划分

| 项目 | 设置 |
|---|---|
| 模型 | OLMo-2-0425-1B-SFT，revision 0d85a3d037876ce6ac7d4311d994400fc66ac27f |
| 数据 | Anthropic/hh-rlhf，revision 09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa |
| Design | 官方train 160800原始pair全部建库，报告有效M |
| 方法/seed | Empirical、PRISM / 11 |
| 偏好预算 | 仅一档，待用户看Design分布后决定，不自动取65536 |
| PPO prompts | train中32768不同完整上下文，两方法共用，允许与Design重叠 |
| Reward开发 | test中2048个人工比较对 |
| PPO开发/最终评价 | test中互斥512/500个上下文，与训练上下文隔离 |

仅使用helpful-base、helpful-online、helpful-rejection-sampled、harmless-base偏好子集。
保留原文件SHA、行号和来源；解析完整共同对话及最后两回答，按内容hash固定A/B，标签另存。
重复、冲突和解析异常分别记录；完整人工回答不截断后沿用标签。统一EOS计入特征及序列logp。
全池编码上限4096；策略及评价prompt上限2048。数据、模型、密钥及产物不进Git。

## 全池特征、K和Design

全部有效人工回答由冻结FP32 SFT做teacher-forcing前向，取得2048维EOS特征和完整序列logp。
GPU独立分片，严格检查全局索引、顺序、覆盖及hash；不是重新生成这些回答。

沿用原loglinear代理：t为每对中心化logp，b=(X^T X+0.01 I)^(-1)X^T t；
p_A,p_B=softmax(phi_A^T b,phi_B^T b)，v_i=phi_iA-phi_iB，
K_HH=(1/M) sum_i p_iA p_iB v_i v_i^T。

K是HH有限双候选协方差代理，不冒称真实LoRA策略的精确G^T Fdag G。
以pair平均，不是将同prompt全部候选联合归一化。不再为K生成4096x4个新SFT回答。

固定同一v/K和surrogate ridge=0.01，matrix ridge比较0.001/0.01/0.1。
目标为trace[K(lambda I+sum_i q_i v_i v_i^T)^(-1)]，原simplex约束。
保持既有联合非负归一化L-BFGS-B，500次，原q空间相对FW gap<=1e-4。
不加cap、熵或uniform floor；未收敛产物保留诊断，不放行正式选择。

报告支持数、ESS、最大权重、top1%/10%质量、95%/99%覆盖、微小正权、K谱，
以及各自lambda下F(q)/F(emp)。跨lambda的原始F不直接排名；ESS不是WOR预算硬上限。
助手负责选Design正则。用户确定N后显式freeze-study --budget N --user-budget-confirmed；
选择文件绑定三份报告及权重。未冻结N时sample/Reward/PPO均拒绝开始。
Empirical均匀有效pair，PRISM按剩余权重顺序不放回；不伪造包含概率为Nq，不补采零权边。

## Reward与共享SFT

冻结backbone，2048到1无bias线性BT，每个选中pair等权。
L-BFGS-B最多2000次，梯度无穷范数目标1e-6；L2候选0.001/0.01/0.1。
双方共用2048对开发集，以平均BT NLL选一个公共系数，严格并列取更强L2。
6次CPU头拟合，选中两头直接复用；报告训练/开发NLL、准确率、Brier及置信度校准。

共同4096条固定SFT回答用于定标；各Reward按相同回答计算自己的mu/sigma并冻结，
r_normalized=(r-mu)/sigma。轨迹复用作critic预热；另外512条PPO开发prompt生成共用SFT基线。
共享生成总计4608条，不用于K；最终500题不参与选参。

## PPO配方

| 参数 | 设置 |
|---|---|
| Actor | LoRA r16/alpha32/all-linear/dropout0 |
| Critic | 独立SFT副本+独立LoRA r16/alpha32+带bias value head |
| Episodes | 65536，共同32768 prompt池两遍 |
| Rollout/minibatch | 128/64，每轮2 minibatches |
| Microbatch/accumulation | 2/32 |
| PPO epochs/真实更新 | 4/4096次，512 rollout轮 |
| LR actor/critic LoRA/value head | 1e-5/3e-5/1e-4 |
| 日程 | 5% warmup，cosine到初始LR的10% |
| AdamW | beta=(.9,.999)，epsilon=1e-5，weight decay0 |
| gamma/GAE lambda/KL beta | 1/.95/.05，标准化Reward单位 |
| policy/value clip，vf_coef | .2/.2，.1 |
| 梯度裁剪 | actor/critic各自实际执行norm1 |
| 生成 | T1/top-p1/top-k0/cap512/首EOS停止 |
| 数值 | 一致FP32/禁TF32/零更新概率闸门 |

新执行器为v3_single_card_streaming_ppo_v1，不冒称旧TRL只改配置。
Clipped policy/value loss、终点p1放置和GAE沿用TRL0.15.2数学；
为使microbatch等价，梯度按整个minibatch有效token归一化。
生成及时提取selected logp并释放全词表logits；每2048 episodes另做耐久保存，与昂贵探针分离。
checkpoint绑定源码、模型、数据、
Reward、优化器、采样器游标及RNG；恢复不重跑已完成训练或共享生成前缀。

Critic从归一化Reward头初始化；2048条SFT轨迹4遍MC回报预拟合，minibatch64，共128步。
另外最多128条轨迹检查heldout EV>0及MSE优于常数，再开始actor更新。
使用最终N的Empirical Reward，验收2048工程节点和8192开发节点，不另设32K偏好档。
8192共用闸门要求train奖励正增益、dev配对奖励及RLHF目标mean-1.96SE>0，
dev critic MC拟合优于常数。失败保存检查点暂停，PRISM不放行。
正式终点65536；检查点8192/16384/32768/65536，不按最终测试结果选checkpoint。

## 评价、费用和资源

HH500，SFT/Empirical/PRISM统一生成与prompt级seed。主比较PRISM对Empirical，
配套为两者分别对SFT。Luna medium，AB/BA，胜1/平0.5/负0，先逐题平均。
主评3000有效判断；原始调用和重试另计。Prompt bootstrap2000次，报告胜平负、
换序分歧、helpful/harmless分项、长度/EOS/截断及MC KL，负值不裁零。

人工训练标签无付费调用。Luna仅本机、独立V3账本、显式新USD上限；当前未授权付费评价，
默认关闭。不复用/重置V2账本、旧费用及未知预留。实际tier、usage及预留均保留证据。

GPU每任务1张、独立分片并行，不因资源授权擅自将PPO改成多卡。
特征/生成2CPU/16G/1GPU；PPO8CPU/64G/1GPU，优先A10080GB；
Design/几何16CPU/24G，遵循CPU分区每核内存规定，峰值待实测。
个人运行/排队合计不超过50；源码在提交时固定快照，所有集群文件在online1。
登录节点只提交/归档/调度查询；正常15-30分钟检查，异常/重要阶段立即处理。

## 2026-11-17本次准备和启动状态

本机隔离Python3.12/Torch2.8 CPU使用Transformers4.52.3、PEFT0.15.2、Accelerate1.7；
V3最新专项87项全部通过，包含真实tiny OLMo2前向、LoRA/critic更新、stream/full一致、
累积等价、独立裁剪、逐元素checkpoint恢复、共享生成恢复、离线费用导入。
最终本机全库UTF8回归375 passed/14 skipped；生产Python3.11 SIF全库1986918已393 passed、零跳过。
首轮1986890的6项失败来自模拟本机Judge测试继承Slurm作业号，测试环境已独立隔离；
生产集群API禁用规则及显式拒绝测试均保留。两次记录都保存，不将工程检查视为正式PPO学习效果。

固定8份HH原文件已下载并hash归档。本机全量预检得到159682有效train pair、
158861训练上下文；test8493有效pair中71条与train上下文重叠，排除后8422。
原train1118条/test59条隔离；合计原因：角色歧义219、空回答183、相同回答775。
无静默截断或猜标签，三个开发/评价划分及32768策略上下文足额。

用户恢复SafeConnect VPN后SSH已接通。生产HH预处理1986895用时3分39秒完成，
最终统一tokenizer得到的上述数量与预检一致；data receipt SHA256为
02f44f50488d972d974afacb71fcae9df85d6ca2025cdb9d79fcfb3ea8d39a01。
AMD/Intel A100及4090真实FP32 1B模型验收1986892–94均通过。

18:16时32片特征数组1986926/27/28已有11片RUNNING（7张AMD A100、2张Intel A100、2张4090），
余片随空卡接续；特征尚未完成。源466babb固定于独立release，配置字节及研究算法未变。
归并1986930（8CPU/12G）→全池K及独立审计1986932（16CPU/24G）→
三lambda Design1986934/36/37（各16CPU/24G）→报告1986939已按afterok排队。
旧未启动数组及其归并在保留元数据/hash后普通取消，0分配；没有hold/release或优先级操作。
完成后暂停等用户N/算法决定，当前未提交sample/Reward/PPO，新增付费API为0。

18:48更新：32片特征全部完成，1986930归并通过159682训练对与2048开发对的全覆盖、
顺序、有限性及哈希检查；GPU均释放。K构建及独立审计1986932通过，
独立LU的b相对误差2.25325e-12、概率最大绝对差3.83742e-11、
显式协方差K相对误差1.25029e-12。K数值秩2048、有效秩32.9236；
双候选pA×pB中位数约1.05865e-11，零乘积0条。这是有限HH回答代理几何，
不将数值复核通过当作代理等同真实策略几何的证明。

三λ Design均在18:47:44实际开始，每个16CPU/24G，使用同一几何、原求解器及1e-4 gap；
当前尚无Design分布结论。报告后另有CPU只读阶段1987080核对全链receipt与N未冻结状态。

## 2026-11-17 19:00：三档Design已验收，按用户要求暂停

三份Design均已COMPLETED，1986939报告及独立CPU全链收据复核1987080均通过。
保持原normalized-nonnegative L-BFGS-B、simplex、目标、500步上限及1e-4相对FW gap，
没有增加概率下限、覆盖约束或均匀混合，没有对输出权重做阈值截断。
几何及求解使用不可变源466babb733a3cfc2e66710b104a04301d071f720；文档提交不改变该版本。

| matrix λ | 正权支持数 | 占159682池 | ESS | 覆盖95%质量所需对数 | 覆盖99%所需对数 | 独立相对FW gap | 迭代 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 14189 | 8.886% | 9563.81 | 10738 | 12625 | 9.55339e-5 | 22 |
| 0.01 | 14075 | 8.814% | 9462.07 | 10641 | 12512 | 7.74466e-5 | 23 |
| 0.1 | 13026 | 8.157% | 8671.45 | 9860 | 11600 | 5.82210e-5 | 25 |

三档均无0<q<1e-15的权重，不是靠微小正权虚增支持。前1%（1597对）质量分别为
24.8164%/25.0725%/27.2248%。各自目标相对同λ下经验分布的比值为
0.358245/0.375753/0.461537；不同λ的原始目标值不能直接作优劣排名。

建议优先考虑λ=0.001：三档中支持最多、ESS最大，λ=0.01与其接近。
当前已验收分布的顺序加权不放回预算硬上限就是正权支持数，最大14189；
ESS不是预算硬上限。N=16384/32768/65536在当前三档都不可行，不能暗中从零权补齐。
若N取满14189，λ=0.001的全部正权对都会入选，随机性只剩顺序。
仅更换求解器不能保证扩大支持；本次已满足原全局gap，不能归因为未收敛。
N及是否修改求解方案由用户下一步决定，尚未冻结study-selection。

本机已逐字节核对并保存三份q与report，对照原receipt SHA重验，另重算支持/ESS/累计质量。
分布报告SHA256为c392b03b47a04c0b9c84d4e4ff577adb16b8c54bfb357c5071db28757032cc26，
报告receipt为63e9247670ea7dda6258bc5b994197eade191e67b6d2ded457e1c7af02a58d69，
独立readiness审计为e9cd415f3af42e4fc98d967a39cadae6fefe86f72f41a2bca692bc8db247bc02。
原q、报告、收据和PNG/PDF累计质量图保存在忽略的local/v3-execution/evidence，原件在online1。

| 任务块 | 实际耗时 | 分配资源累计 |
|---|---|---|
| CPU全库验收（含首轮环境失败及复验） | 2:35 + 2:39 | 0.348889 CPU核时 |
| 三类真实1B GPU验收 | 28/29/27秒，并行 | 0.023333 GPU卡时、0.186667 CPU核时 |
| HH正式预处理 | 3:39 | 0.243333 CPU核时 |
| 32片特征 | 18:16:05–18:43:26，墙钟27:21 | 4.068889 GPU卡时、8.137778 CPU核时 |
| 全覆盖归并 | 1:06 | 0.146667 CPU核时 |
| K及独立LU/显式协方差复核 | 2:20 | 0.622222 CPU核时 |
| 三λDesign | 8:09/5:07/5:59，并行 | 5.133333 CPU核时 |
| 分布报告及全链独立审计 | 1:16 + 1:43 | 0.156667 CPU核时 |

从首份生产验收18:05:23到独立审计18:59:27，共54分04秒，含修复、提交与排队间隔。
合计4.092222 GPU卡时、14.975556分配CPU核时，包含本轮失败验收及已分配诊断；
取消的未启动任务0分配。数据及计算资源时长不是供应商费用账单，新增付费API为0。
19:00:48核对本人Slurm队列为空。N仍null、API关闭，sample/Reward/共享SFT生成/PPO/最终评价
均未启动；原V2账本、结果、研究源码及配置保持不变。

## 2026-11-17：用户确认N=14000、Design λ=0.001，继续后序

用户明确选择N=14000配λ=.001并要求继续，原Design后暂停已解除。固定配置字节不改，
选择经CPU1987189独立LU验收后写入study/selection.json；抽样1987202两方法各14000独特对，
交集1269、现成人工标签并集26731，新增标注API为0。首轮六头拟合1987203在8CPU/12G完成。
核对发现抽样元数据common_gumbel_keys=true不准确：实际共用seed、在各自正权支持顺序上生成噪声，
没有保证同一pair使用同一个Gumbel键。此次只修正日志描述，不改加权不放回算法、seed或选中对；
保留旧抽样及Reward批次和收据hash，重建正确描述的依赖后逐字节核对两个sampling JSONL一致。
旧拟合不是科学数值失败，也不得直接覆盖旧收据。全池几何/Design/用户选择不重跑。

为利用空闲GPU，共享4608条SFT回答增加独立分片及CPU归并，先做真实单卡/分片等价诊断。
最终SFT基线500题已在原固定源466babb提交四片1987213，不作为选参数据。
粗略后序工期：抽样/Reward约5–20min，共享定标生成分片后约30–90min，
critic预热约0.5–2h，先Empirical到8192验收后两方法65536 PPO约1–3天，
最终两actor评价生成/主Judge/汇总约1–3h；以后以真实128/2048/8192吞吐修订。
若critic泛化或共同开发验收未通过，保留现场处理，不能把计划时长当完成承诺。

用户随后已明确批准V3最后Luna medium评价新增硬上限$5，独立V3账本，
实际结算估算与未知请求预留同时占用上限，不能重置或转移旧V2费用。
固定YAML不改；最终本机执行器以显式BudgetUsd=5及不可变packet绑定执行合同。
操作入口见[v3-runbook.md](v3-runbook.md)。
