# PRISM

Prospective-Regret-Aware Preference Sampling for Reward Modeling in RLHF.

研究变量是**标注前的比较边采样分布**：固定SFT候选池 → 选边并获取新标签 → 冻结EOS特征上的BT线性reward → 相同PPO/LoRA → 独立评价。

当前第一版采用 **Empirical / PRISM、OLMo 1B、每方法每seed 1024标签、seeds 11/23**。唯一执行配置是 [configs/experiment.yaml](configs/experiment.yaml)，完整约定见 [第一版实验说明](docs/first-study.md)。H4 Design/PPO/test为3000/2000/500 prompts，每题4候选；评价为同源KL＋Direct Win Rate，以及官方IFEval。

## 执行分层

```text
configs/experiment.yaml        实验定义；相对路径相对config解析
          ↓
scripts/run.py                通用薄入口
scripts/yancheng/             平台调度、容器和资源
          ↓
src/prism/                    配置/数据/设计/RM/PPO/Judge/benchmark/评价
```

改已支持的模型、方法组合、预算、seed或API后端参数只改config；新增算法或provider在src增加适配器，scripts不复制算法。A-optimal保留为可选方法，本版不启用。

## 无计算检查

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[llm,benchmark,test]"
.\.venv\Scripts\python.exe scripts/run.py validate
.\.venv\Scripts\python.exe scripts/run.py plan
.\.venv\Scripts\python.exe scripts/run.py plan --json
.\.venv\Scripts\python.exe -m pytest -q
```

`validate`和`plan`不加载模型、不收费、不提交作业。默认配置会明确报告未准备的数据/资源；不能把结构通过当成GPU验收。第一版计划为4个PPO训练单元、4096个有效训练标签、6000次双顺序API评审、2705次IFEval生成。

## 阶段入口

统一形式：

```text
python scripts/run.py <stage> --config /absolute/path/to/experiment.yaml
```

共享准备：`stage-assets → prepare-data → validate --ready → model-smoke → rollout → design`。

每个配置单元：`sample → label → reward → policy`，传入`--method`、`--budget`、`--seed`。例如本版单元为`--method prism --budget 1024 --seed 11`。

评价先共享`evaluate-reference`，然后先完成所有单元的`evaluate-generate`，再做所有`evaluate-judge`，最后`aggregate`。共享SFT回答和三组直接比较避免重复生成/混淆对手。

`python scripts/run.py resource --for-stage label`按config计算CPU/GPU需求。盐城脚本在 [scripts/yancheng](scripts/yancheng/)；其`--resource`参数接收Python给出的资源分类，平台配置见 [docs/yancheng.md](docs/yancheng.md)。

## 模块

| 模块 | 职责 |
|---|---|
| config.py / io.py / workflow.py | schema、准备状态、输入hash、通用任务矩阵 |
| data.py | prompt-only数据准备、互斥划分、固定版本与ID |
| design.py | loglinear proxy、K_hat、A-optimal/PRISM、Frank-Wolfe |
| reward.py | 不加权BT＋显式可配置固定L2 |
| hf.py | 冻结SFT EOS feature与TRL PPO接口 |
| judges.py | 独立training/evaluation角色；DeepSeek/OpenAI/HF适配 |
| benchmarks.py | 固定官方源码/资源的IFEval；兼容exact-match |
| evaluation_v2.py | SFT缓存、KL、AB/BA/TIE、跨方法比较与汇总 |

schema1本地HF旧接口仍可用，示例在 [configs/hf.example.yaml](configs/hf.example.yaml)。新默认schema2明确区分API Judge与HF checkpoint，使用API名称不再触发HF模型下载。官方IFEval不会退化成exact-match。

## 当前执行边界

`execution.allow_paid_api_calls: false`与`PRISM_ALLOW_SUBMIT=0`保持关闭。本次不提交盐城作业、不加载所选OLMo做训练、不调用真实Judge。模型权重、数据、NLTK资产、结果与密钥均不提交Git。

平台为是石科技盐城集群，代码和运行数据放在online1共享存储；原始[计算服务](计算服务/)资料含私人账号信息，仅保留本地并已忽略。Apptainer镜像仍需重新构建，真实OLMo/A100兼容性需未来验收。详见[网络说明](docs/network.md)与[数学契约](docs/protocol.md)。
