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

## V5 主线 Reward 结果

| 方法 | 训练准确率 | 开发准确率 | 开发 BT NLL |
|---|---:|---:|---:|
| Empirical 14k | 66.5357% | 57.0801% | 0.676846 |
| PRISM 14k | 70.7714% | 57.9590% | 0.671447 |

PRISM 的训练集拟合明显更高，但共同开发集只提高 0.879 个百分点。因此 V5 不把 Reward
开发准确率单独作为下游成败判断，而是同时报告最终策略的 Luna、IFEval 和 KL。

## V5 主线 PPO 结果（16,384 episodes）

| 策略 | Luna 对 SFT | IFEval prompt strict | 相对 SFT 差值 | sequence KL |
|---|---:|---:|---:|---:|
| SFT | — | 45.4713% | — | 0 |
| Empirical | 66.90% `[63.05,70.65]` | 41.9593% | -3.5120pp `[-7.3937,0.1848]` | 12.7732 |
| PRISM | 71.35% `[67.75,75.10]` | 40.4806% | -4.9908pp `[-9.0573,-1.2939]` | 10.5526 |

直接主比较 PRISM 对 Empirical 的 Luna 胜率为 **59.20%**，95% CI
`[55.35%,63.15%]`。结论是：在同一 14k 标签预算下，PRISM 的 HH 偏好收益显著高于
Empirical，且外部生成 KL 更低；但两者都降低了 IFEval，尤其 PRISM 的配对差值区间全负，
所以不能表述为“无条件优于 SFT”。

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
