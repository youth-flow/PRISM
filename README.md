# PRISM — V4

Prospective-Regret-Aware Preference Sampling for Reward Modeling in RLHF.

当前分支`v4`，使用固定的 **OLMo-3-7B-Instruct-SFT**、HH人工偏好及4096维FP32 EOS特征。V3的1B实验与结果保留。

V4提供独立配置、Transformers 4.57.1运行层、人工cohort锁定与重新编码、分片特征/完整回答logp、全池冻结线性Reward诊断、原PRISM Design与不放回采样、7B内存适配的PPO及独立Luna评价账本。actor/critic的LoRA独立，仅共享冻结基座存储。

本次为工程迭代。N/选中Design λ尚未确定；正式PPO和付费评价默认关闭。7B真实GPU验收与全量结果须由后续计算证明，不能用小模型测试代替。

- [V4工程与执行手册](docs/v4-runbook.md)
- [V4固定配置](configs/experiment.v4.yaml)
- [V3方案与已完成Reward结果](docs/v3-plan.md)
- [V2正式结果](docs/v2-results-2026-11-17.md)
- [数学定义](docs/protocol.md)、[FP32数值协议](docs/numerical-protocol.md)

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:PYTHONUTF8 = "1"
python -m prism.v4 validate --config configs/experiment.v4.yaml
python -m prism.v4 plan --config configs/experiment.v4.yaml
python -m pytest -q
```

validate/plan不下载、不加载模型、不提交任务、不调用API。模型相关检查使用独立V4环境；旧环境会跳过OLMo3专项，不能据此宣称7B验收完成。集群入口为`submit_v4.sh`，只在online1、Slurm分配内计算，默认提交关闭、单任务1GPU、个人展开任务不超过50。

密钥、原始人工对话、模型、容器、账本与运行产物不进Git。只普通推送当前任务分支，不改写V3/V2/main历史。
