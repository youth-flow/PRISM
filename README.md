# PRISM — V5

Prospective-Regret-Aware Preference Sampling for Reward Modeling in RLHF.

V5 将实验收敛到一条固定主线：OLMo-2-0425-1B-SFT、HH 人工偏好、冻结 SFT
EOS-2048 特征上的无 bias 线性 Reward，以及已经验证的 14k Reward / 16k PPO 配方。
计算执行复用审计过的 V3 引擎，但 V5 入口会拒绝 LoRA Reward/PPO 和参数漂移。

| 环节 | V5 协议 |
|---|---|
| 数据 | 固定 revision 的 HH 四个正式 subset；训练/开发/PPO/最终评价隔离 |
| 特征/K | 冻结 OLMo-1B FP32 EOS-2048；全池 loglinear 代理 K |
| Design/采样 | λ=.001；Empirical/PRISM 各 14,000 对；PRISM 加权无放回 |
| Reward | 冻结骨干、无 bias 线性 BT、L2=.1、4096 条固定 SFT 定标 |
| PPO | 每方法 16,384 episodes；4×4090 同步；保留 65,536 学习率时钟 |
| 评价 | Reward dev、HH500 Luna medium AB/BA、官方 IFEval、sequence KL |
| 结果 | 主线与同源 LoRA 历史对照统一登记，但严格分层、不混作主结论 |

- [V5 协议与统一结果](docs/v5-protocol-and-results.md)
- [V5 配置](configs/experiment.v5.yaml)、[机器可读结果](results/v5/evaluation-registry.json)
- [V3 执行和恢复手册](docs/v3-runbook.md)、[V3 线性 PPO 最终结果](docs/v3-linear-ppo-results-2026-11-20.md)
- [V2正式结果与审计](docs/v2-results-2026-11-17.md)
- [数学定义](docs/protocol.md)、[数值协议](docs/numerical-protocol.md)

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:PYTHONUTF8 = "1"
python -m prism.v5 validate --config configs/experiment.v5.yaml
python -m prism.v5 plan --config configs/experiment.v5.yaml
python -m prism.v5 results --config configs/experiment.v5.yaml
python -m pytest -q
```

validate/plan不下载、不加载模型、不提交作业、不调用API。实际阶段显式传--execute。
正式 sample/Reward/PPO 仍要求 selection/admission receipt；Judge 要求独立账本和显式美元上限。
集群模型计算经 Slurm 在固定 SIF 内执行；特征/生成可独立分片，正式线性 PPO 固定单节点
4×4090 同步数据并行。
OpenAI key留本机；原始对话、账号、密钥、模型、容器、账本和运行产物不进Git。

`prism.v5` 负责固定协议、执行白名单和结果 registry；底层继续复用 `prism.v3` 的
data/features/geometry、study/reward、PPO、evaluation/audit。V5 配置默认关闭付费 API，
本分支整理过程没有启动 GPU、PPO 或 Judge。
