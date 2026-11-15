# 后续初步试验：预算、Judge、Reward与PPO建议

用户于2026-11-15明确：这仍是初步试验，使用现有DeepSeek约50元和OpenAI约50美元
额度；同意把偏好标注预算作为后续实验变量。以下记录建议与约束，尚未启动新标注
或完整训练，也未把建议参数写入冻结实验配置。

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
用于所有方法，不逐方法挑最有利参数。若固定表示仍构成瓶颈，再定义独立的
可训练骨干RM对照，不把它混入当前冻结特征结论。

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
