# 后续初步试验：预算、Judge、Reward与PPO建议

用户于2026-11-15明确：这仍是初步试验，使用现有DeepSeek约50元和OpenAI约50美元
额度；同意把偏好标注预算作为后续实验变量，并再次确认Reward仅训练线性头、
不微调backbone（也不增加Reward LoRA）。PPO actor原定LoRA训练不受此约束。
以下记录建议与约束，尚未启动新标注
或完整训练，也未把建议参数写入冻结实验配置。
其中PPO短程选参已随后完成，当前推荐actor1e-5/critic3e-5/lambda.95作为下一轮
起点，确认集优势未达显著；完整记录见[短程报告](ppo-parameter-pilot.md)。

## K与重建Design

K是loglinear代理概率下的prompt内特征协方差，给Design的不同方向赋权。
输入链路为固定候选token→EOS特征/序列logp→b→代理概率→K→最优q。
实测FP32相对旧精度的特征变化2.374%、b变化14.688%、K变化86.922%，说明
候选间差异和代理概率的变化不能用原特征整体百分比简单推算。原求解器正确求解
原输入下的问题；旧证书不说明旧输入对精度稳定。

重建Design就是用一致精度重算该链路并重新抽取query，不涉及重新训练SFT或先调用
Judge。同一候选池和研究参数下，q不随当前标签预算N改变，不用为三个预算重复求解。
固定旧候选token的FP32重算和新q证书已完成，见numerical-protocol.md。
作为初步有限候选实验，可以显式保留这些候选文本，仅重建特征/几何/查询依赖，
并如实记录其原始生成精度；不能把复用旧token描述为重新从FP32策略抽样。
若后续研究要求整个生成分布也严格切至FP32，则另行生成新版候选。

## 建议的预算轴

- 两方法：Empirical、PRISM；两个seed：11、23。
- 建议标签预算：1024、2048、4096；合计12个Reward/策略单元。
- 建议nested_prefix：每方法/seed独立抽4096次并标注一次，较小预算取同一观察流
  的前缀。每次重复抽到同一边仍获取新的独立判断；跨预算复用已有观察。
- 标注总量2×2×4096=16384；各预算完全重新独立标注则为28672。
- 各预算Reward/PPO从头拟合，不继承小预算模型。预算间结果相关，统计时保留
  seed和prompt配对关系，不作为独立重复实验计算。
- 所有预算先完成便宜的Reward拟合/开发验证；PPO超参数只在公共开发设置校准一次，
  最终同等应用于两方法与全部预算。

当前冻结v1/v2配置的independent耦合未被静默修改；nested_prefix是新多预算配置的建议。

## Judge与金额

建议先在约200对具有代表性的开发样本上比较原设置与候选设置，包含明确偏好、
近似平局和复杂题。训练与评价均盲化方法身份，AB/BA用于校准，保留统一rubric。

| 用途 | 建议候选 |
| --- | --- |
| 训练Judge | deepseek-flash，thinking enabled、reasoning_effort low，严格偏好工具；输出上限先1024，超时相应放宽并真实验收 |
| 常规评价 | gpt-6-luna，reasoning medium、输出上限2048，保留AB/BA与TIE |
| 开发集参考/抽检 | gpt-6.1-sol high，用于小批独立参考，结合少量人工检查 |

DeepSeek thinking模式不接受有效的temperature控制，不能把降温当作该方案的一部分。
输出预算包含思考token，原64不能直接沿用；1024仍是待实测的初始cap，必须检查截断。
若候选未改善开发集质量，继续选择配置，不凭“开了thinking”认定合格。

金额按实际返回usage、所有失败/重试累计，含此前消费，保持用户总额度。
先用开发样本测得每个有效标签的实际成本，确认能在额度内完成后执行；尚未实现或
启用新的跨进程支出硬限额，不宣称现有worker已经会自动按人民币金额停机。
建议把DeepSeek新增训练/校准控制在约35–40元内并留余量；OpenAI先按5–10美元
规划校准、留出标签和Luna评价，不必花完整个50美元。

以每次1000输入、300输出、无缓存、空闲Flash价为例，16384次为约36.04元，
未含重试和校准；若思考更长则会超出该估算，必须按小批实测调整。该算例是token
成本演算，不是已测得的新配置费用或保证。预算标签数和实际HTTP次数分别报告。

2026-11-15官方依据：
[DeepSeek价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、
[思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)、
[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)、
[OpenAI价格](https://developers.openai.com/api/docs/pricing)。

## Reward泛化与工业做法

末端token/EOS表示接标量线性头是常见实现。关键区别在于，常见RM还微调骨干或
LoRA；本项目为固定2048维SFT特征，只估计线性头，表达能力更受限制。
这个冻结特征设定是当前采样理论和实验的组成部分，不直接换成LoRA Reward。

公开数据规模差异很大：InstructGPT为约3.3万训练prompt，每题4–9个候选排序，
不能误写为仅3.3万比较对；Llama2报告的Meta自采偏好比较约141.8万，连公开数据
合计约291.9万。它们是规模参照，不是本项目最低数据量要求。

改进优先级：统一精度→改善训练Judge→预算曲线→独立prompt级开发验证。
开发集看排序准确率、BT logloss与校准；不能按各自训练准确率横比方法。
若标签与预算改善后仍不佳，再在公共开发配置比较L2=0.01/0.03/0.1，选择统一值
用于所有方法，不逐方法挑最有利参数。若固定表示仍构成瓶颈，如实报告该表示类的
限制，保持用户确认的仅训练头设定，不改为可训练骨干或Reward LoRA。

依据：[InstructGPT](https://arxiv.org/html/2203.02155v1)、
[Llama2](https://arxiv.org/html/2307.09288v2)、
[TRL RewardTrainer/PEFT](https://huggingface.co/docs/trl/reward_trainer)、
[末端线性投影实现](https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/llama/modeling_llama.py)。

## 覆盖与PPO强度的判断

集中可以把预算用于重要方向，也可能牺牲覆盖或放大代理失配。覆盖少本身不是
错误；不以“让分布更均匀”为修复目标，先观察独立开发集与预算曲线。

原3e-6 actor/critic共同学习率、4096 episodes是偏保守的有效优化设置。
我的建议是先在短程开发实验比较actor 3e-6与1e-5，critic独立用3e-5，保留原
KL=.05、batch64、2 PPO epochs和LoRA设置；它们是建议，尚未改入现用配置。
4096作为初步训练量，若开发集收益仍持续上升，再把所有正式单元统一扩至8192。
不能只延长一个方法或对每个预算挑不同的训练强度。数值闸门通过不等于PPO已充分训练。

## 费用估算与PPO选择依据补充

以下为新设置的估算，API费用与集群核时/卡时分开。假设每次约1000输入token，
DeepSeek空闲价且不计缓存优惠；输出包含推理token。

| 项目 | 请求规模 | 输出均值假设 | 纯token费用 |
| --- | ---: | ---: | ---: |
| Flash thinking low训练标签 | 16384有效标签对应的基本请求 | 200–350 | 29.49–39.32元 |
| Luna medium常规评价 | 18000次有效判断 | 200–500 | $3.60–6.30 |
| Sol high小批参考 | 200对×AB/BA=400次 | 200–500 | $1.60–2.80 |

加校准、缓存写入和失败/重试余量，新增DeepSeek规划约35–45元；Luna约$4–8，
Sol参考约$2–4，OpenAI合计约$6–12。前轮已返回usage对应约3元和约$1也计入
总额度。新配置尚未实测，若Flash平均输出500，训练请求本身就约49.15元，
无法再直接装入50元累计额度，须在小批阶段重新安排。高峰DeepSeek价格约翻倍，
因此训练API优先安排空闲时段。不能把这些假设费用当作已测得账单。

Reward头为d→1，d来自骨干hidden_size：本项目2048→1、2048个无bias权重；
Llama3 8B为4096、Llama2 13B为5120、Llama2/3 70B为8192。这是对应骨干构建
标量RM时的头输入维数，不是声称各公司采用完全相同的RM训练方案。
来源：[Meta官方模型配置](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)。

PPO短程校准建议固定Reward、prompt、seed和4096总episode学习率日程，实际先执行
512–1024 episodes，对比三组：

| 组别 | Actor LR | Critic LR | 作用 |
| --- | ---: | ---: | --- |
| A | 3e-6 | 3e-6 | 修复精度后的原学习率基线 |
| B | 3e-6 | 3e-5 | 隔离critic学习速度影响 |
| C | 1e-5 | 3e-5 | 再检验约3.3倍actor步长 |

两方法最终共同采用一套设置；先在固定开发Reward校准，再在另一Reward验证。
依据是FP32固定终点正对照256步：Empirical MSE从206.726起，3e-6只降至184.129，
3e-5降至57.191；PRISM从65.451起，分别降至54.562和16.938。这直接支持critic
独立提高学习率，但不是完整GAE价值学习或PPO收益证明。Actor的1e-5是待检验候选，
不是已经验证的最优值。原四组未检出稳定自身Reward增益，使检验更大有效步长有意义。
短跑以独立开发prompt上的自身Reward、KL、value误差和梯度稳定性验收，不要求
每次Adam更新的surrogate loss单调。4096作为首个正式训练量，只有开发曲线仍上升
才把所有单元统一延长；若采用8192新日程，则全部方法同样采用，不把不同日程
的前4096步冒称完全相同。该三组及追加D/E的两Reward短程对照后已完成；
实测结果与独立确认见[短程报告](ppo-parameter-pilot.md)，本节前文为执行前建议。

## 2026-11-16 V3标签池讨论：尚未授权执行

用户明确缓存/全18000比较的提问是为V3考虑，要求当前先不改。V2继续冻结的
有放回draw与逐draw新判断协议，仅执行2048/4096；没有追加全池标注。

只读统计当前四个4096查询前缀：Empirical seed11/23分别3654/3638种edge，
PRISM分别512/510种edge；四流16384观察合计7021种edge（按固定prompt与候选
索引定义）。因此缓存可显著减少调用，但沿用一次标签给重复draw加权，与每draw
取得新的随机判断不同，不能把缓存行算作新增独立观测；影响方向尚未实测。

用户提供的Han/Goyal/Ma《Which Pairs to Compare for LLM Post-Training?》
（arXiv:2606.19607v1，2026-06-17）已核对正文第2–3节及附录F，第49–50页
设计与抽样公式另行渲染检查：

- 理论第5页采用i.i.d. edge及BT标签；附录C第25页明确给定edges后的独立标签。
- 实际IMDb prompt选择为1000取175、不放回；response选择每prompt八候选取一pair。
  文中情感分类器明确用于reward/KL评价，但训练标签生成细节未充分说明，不能
  据此断言已标注所有28种候选pair。
- HH从已有160800 chosen/rejected比较中按权重不放回选N，直接使用已有标签；
  GPT4.1用于最终评价，不是其HH训练标注器。N为训练比较条数，并非新付费调用数。
- 其设计使用末token特征、中心化logp的ridge代理、softmax协方差以及trace目标，
  与本项目A3–A5同构；HH还标准化/PCA至128维，使用g=beta*feature_difference。
  带ridge时缩放等价于lambda/beta^2的有效阻尼，不能仅删beta而不核对阻尼。
- 理论的i.i.d.界不能直接冒称已证明实际加权不放回的相同界；Reward→PPO还需
  计入Reward估计/正则偏差、策略表示限制与PPO求解误差，不能直接移植DPO保证。

V3可讨论固定公共标签库、两方法均不放回取N种比较、报告选样次数/唯一pair/
实际调用及建库总费用。若复用现有7021种edge，需预先固定取哪次判断、canonical
方向及同一Judge协议，不能挑标签；剩余10979种只是在此定义下的覆盖缺口，
不是已经批准的新增API任务。可另设18000全量Reward参考，但不保证其PPO最好。
当前候选文本已复用V1，FP32特征/几何重新绑定验收；PPO自身轨迹仍须当前actor
重新生成，不直接沿用旧策略rollout。

后续澄清：DPO直接训练策略并具有隐式reward解释；RLHF的Reward训练也可以使用
外部离线偏好对，并非必须由本次SFT生成。当前V2使用SFT候选是冻结研究约定。
外部给定回答可在同一冻结SFT上teacher-forcing前向，获得EOS feature与序列logp，
再计算v=phi_chosen-phi_rejected。外部池上的协方差仍是池内几何代理，不能
直接当成完整SFT输出分布的精确Fisher。可讨论保留独立无标签SFT rollout估计
目标K、用外部已标注比较提供设计向量的两来源方案；这不是已验证的V3实现。

当前固定PRISM q已只读核对：18000项中6142项>0，11858项=0，负数0项，
min=0，sum=1，max=0.010873718897217487；前180项质量0.8859323177752944。
2048/4096不放回在支持数上可行，8192不能在此q上直接执行。加权顺序不放回
每次移除已选项并重新归一化，实际包含概率不等于N*q；若要更贴合相应理论，
须另行定义包含概率/覆盖约束和设计问题，不给当前q暗加uniform质量。全部仅为
V3讨论，当前V2没有变更抽样、标签或几何。
