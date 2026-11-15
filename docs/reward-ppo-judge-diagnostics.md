# Reward、PPO 与 Judge 的事后诊断（2026-11-15）

本次只读取既有候选特征、四组固定标签/Reward权重、PPO日志与Luna原始判断，
在本机计算，没有训练新模型、修改冻结实验或新增API调用。
以下是解释性诊断，不替代或改写原胜率、KL、IFEval主结果。

## PPO 强度与分数设置

| 参数 | 四单元共同值 |
| --- | --- |
| total_episodes | 4096 |
| rollout batch | 64 = microbatch 2 × accumulation 16 × mini-batches 2 |
| rollout updates | 64 |
| PPO epochs per rollout | 2 |
| learning rate | 初始3e-6，线性衰减；末次日志4.6875e-8 |
| KL coefficient | 0.05 |
| policy/value clip | 0.2 / 0.2 |
| value loss coefficient | 0.1 |
| gamma / GAE lambda | 1.0 / 0.95 |
| max gradient norm | 1.0 |
| actor LoRA | r=8、alpha=16、dropout=0、all-linear |
| response cap / temperature | 512 / 1.0 |
| reward whitening / 固定区间缩放 | 未开启 / 未使用 |

每单元64条日志的平均单次policy approxkl约1.88e-4，平均clip fraction约
0.000184–0.000289。它们不同于最终500条回答的序列KL MC估计。
现有证据支持更新幅度小，但不证明只要提高学习率或训练量就会改善独立评价。

Reward是冻结EOS特征上的原始线性分数R=w^T phi，输出无固定上下界。
BT使用sigmoid(R_a-R_b)预测二元偏好；PPO使用原始R，未把它压到0–1。

## 实际分数与拟合

用原12000条SFT候选、完整2048维特征和四个原权重重算如下。
这些是本候选池的观测范围，不是所有文本上的理论界，也不是PPO新回答的个体极值。

| Reward | 最小值 | 中位数 | 最大值 | 自身训练集准确率 |
| --- | ---: | ---: | ---: | ---: |
| Empirical11 | -11.430 | 13.383 | 38.263 | 94.14% |
| Empirical23 | -16.978 | 3.813 | 20.505 | 94.63% |
| PRISM11 | -11.569 | 6.608 | 22.894 | 83.01% |
| PRISM23 | -14.299 | 8.469 | 32.135 | 85.16% |

不同Reward的零点与尺度不同，不以分数绝对值横向评判好坏。
同prompt公共偏移在BT差分中消失；分数尺度会影响PPO奖励与固定KL系数的相对关系。
是否造成当前优化收益弱，仍需验证，不能由本表直接作因果判断。

原标签/权重的SHA256全部与各自receipt一致；pool.npz哈希为
`c92c0c61e69c069e83fff082eabe2cbca53e586f7675748ec97358da1eadd7cd`。
用原特征独立重构四个mean BT+0.01/2||w||²目标及梯度，与原report误差小于1e-10。
四个梯度无穷范数仍低于1e-6，未发现数值拟合错误。
各自训练集不同，表内训练准确率不能直接用于方法优劣比较。

PRISM11/23训练集中，相同边观察到的相反标签使任何确定性按边预测的训练
准确率最高只能达到83.40%/85.25%，当前83.01%/85.16%已接近这个经验上限。
这个上限针对这份随机标签样本，不是对真实偏好正确率的估计。

## 不新增标签的交叉留出诊断

对训练seed11的两个Reward，使用原Empirical seed23标签；对训练seed23
的两个Reward，使用原Empirical seed11标签。每个方向剔除两个被测Reward
任一训练集出现过的全部prompt，两方法在完全相同的剩余标签上比较。
候选特征曾用于无标签Design，本诊断只保证这些prompt未用于被测Reward拟合。
没有重拟合、调参、选择标签或新增调用。

| 被测Reward训练seed | 验证观察数 / prompts | Empirical准确率 | PRISM准确率 | Empirical/PRISM BT logloss |
| --- | --- | ---: | ---: | --- |
| 11 | 655 / 551 | 58.47% | 56.18% | 1.221 / 1.219 |
| 23 | 644 / 550 | 55.36% | 60.02% | 1.214 / 1.244 |

验证标签正类比例为49.01%/47.98%，并非极端类别不平衡。
总是预测概率0.5的logloss是log(2)=0.693；四个模型在该诊断上的logloss
均更高，提示概率预测过度自信等校准问题，不能仅凭拟合收敛认定泛化足够。

按prompt成组做2000次配对bootstrap，PRISM减Empirical的准确率差：
seed11=-2.29个百分点，95%区间[-6.99,2.26]；seed23=+4.66个百分点，
区间[-0.16,9.48]。两区间均包含0，不据此认定一种方法的Reward普遍更好。

这提供了两种Reward泛化偏弱的实际线索，但属于事后、同候选池分布的
交叉留出诊断，不是预先保留的独立新benchmark，也不能代替对PPO新回答的验证。
1024标签不足不能仅由2048参数得出；还取决于有效维度、覆盖、正则及标签噪声。
本例PRISM只有266/269条不同边，意味着1024次观察并没有提供1024个不同设计方向。

## Judge能力与一致性

DeepSeek训练Judge为deepseek-flash、thinking disabled、T=1、输出cap64，
原3次语义尝试、统一v4工具字段；Luna评价为gpt-6-luna、reasoning low、
输出cap1024、原3次尝试、AB/BA与平局。模型支持这些模式不等于本任务
偏好语义准确性已获认证；现有格式、哈希、方向与次数验收不能代替人工校准。

| 比较 / seed | 回答逐字相同的prompt数 | Judge平局比例（1000次） | AB/BA判决不一致（500对） | AB/BA严格胜负翻转 |
| --- | ---: | ---: | ---: | ---: |
| Empirical vs SFT / 11 | 44/500 | 13.2% | 22.4% | 18.0% |
| Empirical vs SFT / 23 | 44/500 | 13.0% | 23.2% | 18.8% |
| PRISM vs SFT / 11 | 45/500 | 12.9% | 24.0% | 21.0% |
| PRISM vs SFT / 23 | 44/500 | 13.2% | 22.2% | 17.4% |
| PRISM vs Empirical / 11 | 194/500 | 41.6% | 17.0% | 13.4% |
| PRISM vs Empirical / 23 | 202/500 | 43.5% | 16.2% | 10.8% |

所有逐字相同的回答在两个顺序上都判为TIE，基础一致性检查通过。
因此直接胜率接近50%部分反映两种策略的输出相同，而非完全由Judge分不清造成。
原始Luna记录逐条核对completed、A/B/TIE及flip还原后计算，本表不修改原判断。

训练DeepSeek的重复边判断成对不一致约25%–30%，但重复比较的观察不是独立
测试集，该数值也不是Judge错误率；近似平局、T=1随机性和顺序效应都可能贡献。
Luna的AB/BA不一致也不是语义错误率，平均两顺序只能缓解顺序效应，不能校准真值。
没有人工或更强独立Judge的盲评样本，不能保证两Judge已足够可靠，也不能直接
认定它们不胜任。应先对已有回答做小规模独立校准，再决定是否更改Judge设置；
新推理强度或新协议的判断必须另记，不能混入本轮冻结评价。

官方能力/参数参考（2026-11-15查阅）：
[DeepSeek模式](https://api-docs.deepseek.com/guides/thinking_mode/)、
[Luna模型](https://developers.openai.com/api/docs/models/gpt-6-luna)。

原Design仍通过原q空间相对gap9.11865834e-5；高度集中是当前代理K、ridge与
设计准则下求得的一份数值近优分布。它不证明所有近优解都必须这样集中，
也不直接证明这种设计对下游真实偏好有益或有害。

下一步判断应同时考虑Reward覆盖/泛化、PPO优化有效性和Judge可靠性，
不把唯一原因归于PPO训练量。原始产物、本轮主要指标与超参数均保持不变。
