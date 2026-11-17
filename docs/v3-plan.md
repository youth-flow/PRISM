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
最终全库UTF8回归375 passed/14 skipped；生产Python3.11 SIF仍需同套验收，尚不能把本机测试称正式GPU学习效果已通过。

固定8份HH原文件已下载并hash归档。本机全量预检得到159682有效train pair、
158861训练上下文；test8493有效pair中71条与train上下文重叠，排除后8422。
原train1118条/test59条隔离；合计原因：角色歧义219、空回答183、相同回答775。
无静默截断或猜标签，三个开发/评价划分及32768策略上下文足额。

目前集群VPN断开、SSH超时，尚未提交V3 GPU或Design作业。
恢复后在固定源码中完成SIF验收、全池特征与三lambda Design，完成后暂停等用户N/算法决定。
操作入口见[v3-runbook.md](v3-runbook.md)。
