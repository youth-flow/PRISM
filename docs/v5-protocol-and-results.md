# V5：OLMo-1B / HH / 线性 Reward 统一实验协议

V5 将已经验证过的研究口径收敛为一条可复现实验主线，并把同一 SFT revision、
同一 HH preference pool 上已经完成的 Reward/PPO 评价统一登记。建立 V5 本身不重跑
模型、不调用付费 API；历史结果只从独立验收过的摘要提取，并保存来源 SHA256。

## 冻结配置

| 项目 | V5 固定值 |
|---|---|
| SFT | `allenai/OLMo-2-0425-1B-SFT`，revision `0d85a3d...ac27f` |
| 偏好数据 | `Anthropic/hh-rlhf` 四个正式 subset，revision `09be8c5b...a1fa` |
| 表征 | 冻结 SFT 的 FP32 EOS 2048 维特征 |
| Reward | 无 bias 的 Bradley–Terry 线性头；正式 L2=0.1 |
| 正式偏好预算 | Empirical 14,000 对；PRISM 14,000 对 |
| PRISM Design | λ=0.001；加权无放回 Gumbel 抽样 |
| PPO | 每方法 16,384 episodes；rollout 128；minibatch 64；4 epochs |
| Actor | LoRA r16/α32，LR 1e-5 |
| Critic | 独立 LoRA r16/α32，backbone LR 3e-5，head LR 1e-4 |
| 约束 | KL=0.05，clip/value clip=0.2，FP32，TF32 off |
| 学习率时钟 | 保留已验证的 65,536-episode warmup-cosine-floor 日程，在 16,384 封存策略 |
| 正式评价 | 500题 Luna medium AB/BA、官方 IFEval 541题/834指令、相对 SFT sequence KL |

配置文件是 `configs/experiment.v5.yaml`。`study.budget` 和
`study.selected_design_lambda` 仍保持 null，这是执行引擎的防误启动约定；V5 的既定选择
单独封存在 `v5.preference_budget=14000` 和 `v5.design_lambda=0.001`，正式执行时仍需
生成 selection/admission receipt。

## V5 Reward 对象与结果

严格按同一个 1B SFT revision 和原始 HH preference pool，已完成五个 Reward 实验族、
共二十个评价点：8k 与 14k Empirical/PRISM 各三个 L2 候选、全池冻结线性三个 L2，以及
全池 LoRA 连续训练的四个规定里程碑和一个开发 NLL 选中点。

### 14k Empirical/PRISM 的全部 L2 候选

| 方法 | L2 | 训练准确率 | 开发准确率 | 开发 BT NLL |
|---|---:|---:|---:|---:|
| Empirical 14k | 0.001 | 70.4000% | 56.7383% | 0.768926 |
| PRISM 14k | 0.001 | 72.7286% | 58.1055% | 0.699250 |
| Empirical 14k | 0.01 | 69.4857% | 57.4219% | 0.711835 |
| PRISM 14k | 0.01 | 72.4714% | 58.2031% | 0.688042 |
| **Empirical 14k（选中）** | **0.1** | **66.5357%** | **57.0801%** | **0.676846** |
| **PRISM 14k（选中）** | **0.1** | **70.7714%** | **57.9590%** | **0.671447** |

公共选择规则是两个方法开发 BT NLL 的等权均值，最终共同选择 L2=0.1。PRISM 的训练集
拟合明显更高，但选中头的共同开发集只提高 0.879 个百分点。因此 V5 不把 Reward 开发
准确率单独作为下游成败判断，而是同时报告最终策略的 Luna、IFEval 和 KL。

### 8k 预算研究的全部 L2 候选

| 方法 | L2 | 训练准确率 | 开发准确率 | 开发 BT NLL |
|---|---:|---:|---:|---:|
| Empirical 8k | 0.001 | 75.0875% | 54.2480% | 0.931857 |
| PRISM 8k | 0.001 | 77.8250% | 57.2754% | 0.755147 |
| Empirical 8k | 0.01 | 73.1875% | 55.0781% | 0.772296 |
| PRISM 8k | 0.01 | 77.3750% | 57.4219% | 0.720717 |
| **Empirical 8k（选中）** | **0.1** | **69.1625%** | **56.6406%** | **0.693975** |
| **PRISM 8k（选中）** | **0.1** | **75.0625%** | **57.2754%** | **0.683294** |

8k 仍按相同的“两个方法开发 NLL 等权均值”规则选择 L2=0.1。选中头上 PRISM 比
Empirical 的开发准确率高 0.635 个百分点、NLL 低 0.01068；但下游 PPO 结果反向，说明
冻结线性 Reward 的同分布开发指标不足以判断策略优化后的真实表现。

## V5 主线 PPO 结果（16,384 episodes）

同源 1B/HH 下共有五次独立 PPO 训练：8k 与 14k 的 Empirical/PRISM 线性 Reward 各两次，
以及一次全池 LoRA Reward。四次线性 PPO 各只对最终 16k checkpoint 做完整外部评价；
LoRA PPO 对 16k、32k、65k 三个 checkpoint 做了完整外部评价。因此是五次 PPO 训练、
七个带 Winrate/KL/IFEval 的 Policy evaluation 对象。2k/8k 等中间闸门只有内部
Reward/KL/critic probe，不冒充完整外部 Policy evaluation。

| 策略 | Luna 对 SFT | IFEval prompt strict | 相对 SFT 差值 | sequence KL |
|---|---:|---:|---:|---:|
| SFT | — | 45.4713% | — | 0 |
| Empirical | 66.90% `[63.05,70.65]` | 41.9593% | -3.5120pp `[-7.3937,0.1848]` | 12.7732 |
| PRISM | 71.35% `[67.75,75.10]` | 40.4806% | -4.9908pp `[-9.0573,-1.2939]` | 10.5526 |

直接主比较 PRISM 对 Empirical 的 Luna 胜率为 **59.20%**，95% CI
`[55.35%,63.15%]`。结论是：在同一 14k 标签预算下，PRISM 的 HH 偏好收益显著高于
Empirical，且外部生成 KL 更低；但两者都降低了 IFEval，尤其 PRISM 的配对差值区间全负，
所以不能表述为“无条件优于 SFT”。

## 8k 预算 PPO 结果（16,384 episodes）

8k 与 14k 使用完全相同的 PPO 强度和学习率时钟，区别只在 Reward 的 8k 选样与对应
线性头。500题 Luna medium、AB/BA 共 3000 个有效判断；IFEval 和 sequence KL 口径与
14k 主线完全一致。

| 策略 | Luna 对 SFT | IFEval prompt strict | 相对 SFT 差值 | sequence KL |
|---|---:|---:|---:|---:|
| SFT | — | 45.4713% | — | 0 |
| Empirical 8k | 67.00% `[63.05,70.85]` | 40.1109% | -5.3604pp `[-9.2421,-1.4787]` | 12.8252 |
| PRISM 8k | 56.20% `[52.20,60.05]` | 31.7930% | -13.6784pp `[-17.7449,-9.7967]` | 14.4634 |

PRISM 8k 对 Empirical 8k 的直接胜率只有 **40.75%**，95% CI
`[36.65%,45.05%]`；Empirical 显著更好。分子集上 PRISM 仅在 harmless 略占优
（54.66%），在 helpful 上只有 35.66%。8k Empirical 的 HH 胜率几乎复现 14k Empirical
的 66.90%，但 PRISM 从 14k 时对 Empirical 的 59.20% 优势反转为 40.75%，且 KL 更高、
IFEval 损失更大。

这给出一个重要的低预算边界：结果与“固定 16k PPO 强度对 8k PRISM Reward 过强、沿
冻结线性头未能可靠外推的方向过度优化”这一机制相一致，但单个训练 seed 不能唯一证明
该因果解释。它不能支持“每个预算 PRISM 都更好”的主张；更稳妥的结论是 PRISM 的优势
在本实验中出现在约 14k，而 8k 需要预注册地检验更强正则、更高 KL 约束、较短 horizon
或基于独立策略 proxy 的早停，不能仅凭 Reward 开发 NLL 选定。

## 同源历史对照（不属于 V5 主结论）

为了避免只保留有利结果，registry 同时保存以下同源对照，但用
`historical_linear_diagnostic` 或 `out_of_scope_control` 标记，禁止提升为 V5 primary：

### 全池冻结线性 Reward

| L2 | 训练准确率 | 开发准确率 | 开发 BT NLL |
|---:|---:|---:|---:|
| 0.001 | 62.9871% | 60.2539% | 0.659842 |
| 0.01（开发 NLL 选中） | 62.7986% | 60.6934% | 0.658015 |
| 0.1 | 61.8060% | 59.7168% | 0.659348 |

### LoRA+线性 Reward（表征能力对照）

| 已见偏好对 | 训练准确率 | 开发准确率 | 开发 BT NLL |
|---:|---:|---:|---:|
| 10,000 | 61.0200% | 57.7148% | 0.669940 |
| 50,000 | 70.1480% | 66.8945% | 0.598041 |
| 100,000 | 71.1940% | 69.3359% | 0.572712 |
| 147,360（选中） | 71.0987% | 69.6289% | 0.566590 |
| 159,682（末次） | 71.0381% | 69.2871% | 0.567637 |

### LoRA Reward PPO 对照

| Episodes | Luna 对 SFT | IFEval strict | 相对 SFT 差值 | sequence KL |
|---:|---:|---:|---:|---:|
| 16,384 | 65.15% `[61.20,68.80]` | 42.5139% | -2.9575pp | 5.9122 |
| 32,768 | 64.65% `[60.75,68.55]` | 42.8835% | -2.5878pp | 6.4437 |
| 65,536 | 65.90% `[62.15,69.55]` | 41.2200% | -4.2514pp | 6.4843 |

这些结果说明“Reward 开发准确率更高”不自动意味着 Luna 胜率更高；最终仍需以策略评价
为准。V5 选择线性 Reward 是研究对象约束，而不是声称线性头表征更强。

### 为什么不纳入 V4 的 Empirical 多预算曲线

V4 确实在 HH 上保存了 10k/50k/100k/159,682 四个 Empirical 线性 Reward checkpoint，
但其冻结模型是 `allenai/Olmo-3-7B-Instruct-SFT` revision `e1452fc...e549`，不是 V5
固定的 1B SFT。它属于另一模型尺度的实验，不能混入“同一 1B SFT + HH”的 V5 registry。
V1/V2 则使用生成候选与模型标注协议，也不是原始 HH chosen/rejected Reward 训练对象。

## 运行和结果入口

```text
python -m prism.v5 validate
python -m prism.v5 plan
python -m prism.v5 results
```

安装包后也可使用 `prism-v5`。V5 CLI 会先验证固定协议，再把允许的线性阶段交给已经
审计过的 V3 执行引擎；LoRA Reward/PPO 阶段会被 V5 入口拒绝。统一机器可读结果位于
`results/v5/evaluation-registry.json`，其中每项都带本机原始证据路径和 SHA256；原始回答、
模型权重、API 密钥和账本不会提交到 Git。

## 8k 执行审计（2026-11-20）

8k 链从 11:55:15 到 19:03:45（Asia/Shanghai）完成。共 65 个实际分配的顶层/数组作业：
63 个 `COMPLETED/0`，2 个操作层失败均保留记录并被普通替代：首次 IFEval merge 暴露
目标运行目录缺少 SFT import receipt；首次补交 import 又因操作命令漏写 `--execute`
被 CLI 正确拒绝。随后 SFT 同 revision 回答的原 receipt、模型/协议、541题覆盖和独立
fixed-RNG 复算全部通过，只重跑 CPU import/merge，没有覆盖或重跑 GPU 回答。另有一个
零分配依赖作业被普通取消。累计 **51.020556 GPU 卡时**、**395.943333 CPU 核时**，
包含失败作业；结束后本人 GPU 队列为空。

Luna packet 固定为 `gpt-6-luna` medium、500题、三比较、AB/BA，共 3000/3000 有效判断；
实际估算费用 **$0.322823**，未知预留 $0，低于独立 $5 硬上限。最终 aggregate SHA256
为 `cb1a97054b101ee723965256d817c5f791bef3471119dfb5bbfc8893df639a55`，receipt SHA256
为 `a2393aa44ed98c5d2346d289718ccd7a7aedc79b6a0cba654eddc154481f8308`；本机证据位于
`local/v5-execution/8k-final`。
