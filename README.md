# PRISM — V3

Prospective-Regret-Aware Preference Sampling for Reward Modeling in RLHF.

当前分支v3。使用HH人工偏好全池，先计算冻结SFT特征及全池loglinear代理K，按原目标求三档Design。
**分布求出后暂停，由用户根据稀疏性决定唯一偏好预算N或是否更换求解算法。**
随后比较Empirical/PRISM、seed11两个正式训练单元；V2配置、权重、账本和结果保留。

| 环节 | V3协议 |
|---|---|
| 数据 | HH官方train 160800原始pair，全池预检并报告有效M |
| 特征/K | 冻结OLMo1B FP32 EOS2048；已有回答前向；全池代理K |
| Design | 原simplex/联合L-BFGS-B；ridge .001/.01/.1；原FW gap<=1e-4 |
| 预算/采样 | 一档N待用户决定；按剩余权重不放回 |
| Reward | 冻结骨干线性BT；共用开发集选公共L2；固定SFT定标 |
| PPO | 独立LoRA critic预热；65536episodes/batch128/4epochs/4096真实更新 |
| 评价 | HH500，Luna medium/ABBA/TIE=.5，三比较3000有效判断 |
| 当前范围 | 全池准备、特征、三档Design、暂停；未授权付费Judge |

- [完整方案与当前状态](docs/v3-plan.md)
- [配置](configs/experiment.v3.yaml)、[执行和恢复手册](docs/v3-runbook.md)
- [V2正式结果与审计](docs/v2-results-2026-11-17.md)
- [V2 Reward/PPO定位](docs/v2-empirical-reward-ppo-diagnosis-2026-11-17.md)
- [数学定义](docs/protocol.md)、[数值协议](docs/numerical-protocol.md)

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:PYTHONUTF8 = "1"
python -m prism.v3 validate --config configs/experiment.v3.yaml
python -m prism.v3 plan --config configs/experiment.v3.yaml
python -m pytest -q
```

validate/plan不下载、不加载模型、不提交作业、不调用API。实际阶段显式传--execute。
未记录用户N时，正式sample/Reward/PPO拒绝开始；Judge要求独立账本和显式美元上限。
集群模型计算经Slurm在online1和固定SIF内执行，每任务1GPU，分片扩展，个人任务<=50。
OpenAI key留本机；原始对话、账号、密钥、模型、容器、账本和运行产物不进Git。

prism.v3独立分层：data/features/geometry、study/reward、ppo/ppo_trainer、evaluation/audit。
新PPO执行器保留概率闸门、实际裁剪、真实步数、独立critic及完整断点恢复，与V2隔离。
重要验收节点普通提交并推送origin v3，不改main或v2历史。
