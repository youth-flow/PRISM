# PRISM — V2

Prospective-Regret-Aware Preference Sampling for Reward Modeling in RLHF.

研究变量是标注前的比较边采样分布：固定SFT候选池 → Design选边 → 新偏好标签 →
冻结EOS特征上的BT线性Reward → 相同PPO/LoRA → 独立评价。

当前工作分支为`v2`。按用户要求，先完成代码/配置审阅，再开始运行；当前API
和正式集群实验保持暂停。已有V1结果和V2部分标签保留，不覆盖、不重新付费。

## 最终实验

| 项目 | 冻结设置 |
|---|---|
| SFT | OLMo-2-0425-1B-SFT，固定revision与原tokenizer模板 |
| Prompt | Design3000、PPO2000、Eval500，另有541条IFEval |
| 方法 / seed | Empirical、PRISM / 11、23 |
| 标签预算 | 2048、4096、8192，nested_prefix，32768个有效观察 |
| Design | 2048维，两ridge0.01，原q空间相对gap≤1e-4 |
| Reward | 冻结backbone，2048→1无bias线性头，BT＋固定L2=0.01 |
| PPO | FP32/禁TF32，Actor1e-5、Critic3e-5，4096 episodes/64 rollout/256 AdamW |
| 训练Judge | Sol high/Flex，4096输出上限、3次尝试、900秒 |
| 评价Judge | Luna medium/Flex，同上，AB/BA及TIE |
| API总额 | 含既有消费最多40美元，操作停止线39美元 |

研究配置为[experiment.v2.yaml](configs/experiment.v2.yaml)，最终Judge执行配置为
[judges.v2.yaml](configs/judges.v2.yaml)。两文件经过显式兼容验证：评价profile只覆盖
Flex/900秒，原experiment.json和已标注packet保持不变。V1原配置为
[experiment.yaml](configs/experiment.yaml)，历史结果见[报告](docs/final-experiment-report.md)。

## 阅读和执行入口

- [完整执行顺序、冻结配置、导入/恢复](docs/v2-runbook.md)
- [计算资源正确高效使用、资源档位、Slurm、计费](docs/v2-compute.md)
- [最终冻结说明及实测Judge费用](docs/v2-frozen.md)
- [概率数值协议](docs/numerical-protocol.md)、[PPO参数依据](docs/ppo-parameter-pilot.md)
- [数学定义](docs/protocol.md)、[算法审计](ALGORITHM_REVIEW.md)

```powershell
$env:PYTHONPATH = "$PWD\src"
python -X utf8 -m prism.v2 plan --config configs/experiment.v2.yaml --profile configs/judges.v2.yaml
python -X utf8 -m pytest -q
```

plan只报告参数、缺失输入、六阶段资源和任务数，不加载模型、调用API或提交Job。
基础依赖为Python3.11/NumPy/SciPy/PyYAML；完整测试需要`.[llm,benchmark,test]`
及显式官方IFEval/NLTK资产，在已验收SIF的CPU计算节点执行。

## 代码分层

| 模块/目录 | 作用 |
|---|---|
| `src/prism/design.py` | loglinear proxy、K、同目标FW/联合L-BFGS-B求解 |
| `src/prism/hf.py`、`ppo_numerics.py`、`ppo_optim.py` | OLMo/EOS、TRL0.15.2、概率闸门、真实AdamW步数 |
| `src/prism/reward.py` | 无重要性加权的BT＋显式L2 |
| `src/prism/v2.py`、`v2_audit.py` | 六阶段计划、配置冻结、前缀导入、Reward/PPO独立验收 |
| `src/prism/budgeted_judge.py`、`api_budget.py` | 本机限额执行、owner锁、账本、响应落盘、暂停/离线恢复 |
| `src/prism/local_judge.py` | 可校验packet、角色/顺序/协议、离线导出导入 |
| `src/prism/evaluation_v2.py`、`benchmarks.py` | KL、直接胜率、配对统计、官方IFEval |
| `scripts/yancheng/` | 不可变源码部署、CPU/GPU单任务调度、共享存储/容器 |
| `scripts/windows/run_v2_judge.ps1` | 本机OpenAI执行入口；默认仅检查，不付费 |

本机和集群运行源码均应固定到已提交版本。重要节点检查后普通推送`origin v2`，
main保留原状态。密钥、账号资料、模型、SIF、数据、缓存、账本和运行产物不进Git。
`execution.allow_paid_api_calls=false`及`PRISM_ALLOW_SUBMIT=0`默认保持关闭。
