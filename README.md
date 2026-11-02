# PRISM

Prospective-Regret-Aware Preference Sampling for Reward Modeling in RLHF.

本工程按根目录《实验流程.pdf》搭建。研究变量是**标注前的比较边采样分布**：共享 SFT 候选池 → 三种 sampling → 固定 Judge 的新标签 → 冻结 EOS 表征上的线性 BT reward head → 相同 PPO/LoRA → 独立评测。

| sampling | 分布/目标 | 对照目的 |
|---|---|---|
| `empirical` | 经验 prompt 分布，每个 prompt 内均匀抽无序比较边 | 原始采样基线；固定候选数时等于所有边均匀分布 |
| `a_optimal` | `min tr((lambda I + Σ q_e v_e v_eᵀ)^(-1))` | 普通参数估计导向的数据设计，K=I |
| `prism` | `min tr(K_hat (lambda I + Σ q_e v_e v_eᵀ)^(-1))` | 检验 Prospective 几何权重的增益 |

不加入第四个 uncertainty 方法。三个方法的差异只在 q；都 iid 有放回抽边，每次 draw 都独立询问 Judge。选择发生在标签产生之前。

## 你需要确定的四组内容

复制 `configs/experiment.yaml` 为 `configs/local.yaml`，填 `choices`：

1. **prompts**：设计池、PPO prompts、heldout evaluation 三个 JSONL 路径。前两者可相同，评估与训练按内容检查不重叠。
2. **SFT**：HF 模型 ID 或本地完整 snapshot，加不可变的 40 位 revision。需要 Transformers 4.52.3 原生支持、提供 safetensors 权重的文本 chat causal LM；不自动执行远程模型代码。
3. **Judge**：固定文本生成式 HF Judge 及 revision。统一的 A/B 模板，随机交换顺序后还原标签，必须输出严格 A 或 B。模型需能遵守该接口；先做小规模格式检查。
4. **benchmark**：名称和转换后的 heldout JSONL。v1 已实现确定的短答案 normalized exact-match；自带复杂评分器的 benchmark 需要接其官方 evaluator，不能把这里的分数冒称为官方分数。

这些是**四类选择、九个必填配置字段**。配置中已有一套统一预算、seeds、PPO/LoRA 参数，作为首轮默认协议，不是已调优参数。HF 上的模型必须先在 CPU Slurm 作业下载，GPU 作业默认离线。

Prompt JSONL 每行：
```json
{"id":"design-0001","messages":[{"role":"user","content":"Your prompt"}]}
```
Benchmark 多一个 `answer` 字符串字段：
```json
{"id":"eval-0001","messages":[{"role":"user","content":"Answer with just one number: 1+1=?"}],"answer":"2"}
```

## 本地检查

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[llm,test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m prism.cli validate
.\.venv\Scripts\python.exe -m prism.cli plan
```

真正运行前：`python -m prism.cli validate --config configs/local.yaml --ready`。缺少选择时会明确退出，不会静默采用示例模型或生成假标签。

共享阶段：
```bash
python -m prism.cli stage-assets --config /path/to/frozen.yaml
python -m prism.cli model-smoke --config /path/to/frozen.yaml
python -m prism.cli rollout --config /path/to/frozen.yaml
python -m prism.cli design --config /path/to/frozen.yaml
```
每个 `(method, budget, seed)` 单元依次：
```bash
python -m prism.cli sample --config /path/to/frozen.yaml --method prism --budget 256 --seed 11
python -m prism.cli label --config /path/to/frozen.yaml --method prism --budget 256 --seed 11
python -m prism.cli reward --config /path/to/frozen.yaml --method prism --budget 256 --seed 11
python -m prism.cli policy --config /path/to/frozen.yaml --method prism --budget 256 --seed 11
python -m prism.cli evaluate-generate --config /path/to/frozen.yaml --method prism --budget 256 --seed 11
python -m prism.cli evaluate-judge --config /path/to/frozen.yaml --method prism --budget 256 --seed 11
python -m prism.cli aggregate --config /path/to/frozen.yaml
```

HPC4 上用 [作业说明](docs/hpc4.md) 包装以上阶段，**不要在登录节点直接执行这些计算命令**。

## 与数学方案对齐的位置

- [数学与实现契约](docs/protocol.md)：精确公式、参数区分、EOS、采样和预算定义。
- `src/prism/design.py`：loglinear surrogate、K_hat、普通/Prospective A-optimal、away-step Frank–Wolfe 和 gap。
- `src/prism/reward.py`：不加权、不加辅助 logit、不默认增加正则的 BT logistic loss。
- `src/prism/hf.py`：冻结表征、LLM Judge、现成 TRL PPO 的 EOS reward adapter。
- `src/prism/pipeline.py`：候选缓存、设计、抽样、标注、RM、PPO 和输入指纹。
- `src/prism/evaluation.py`：独立 prompt 胜率、相对 SFT 的 MC sequence KL、benchmark 和跨 seed 汇总。
- `tests/`：独立求解器对照、数值差分、采样/标签语义、真实微型 TRL 更新与 adapter 重载。

每次实验使用一个不变 config 和 output 目录；输入/配置变化必须开新目录。结果目录包含池、q、FW 轨迹、逐次 Judge 调用、BT 诊断、LoRA adapter、逐条评估及 `results.csv` / `summary.json`。Judge 只有对已成功 **draw ID** 的断点续跑，不按 edge ID 复用标签。中断发生在调用返回与落盘之间时需人工核对额外实际成本。

## 当前边界

这是可执行的第一版工程，而非已经完成的大模型实验。Windows CPU 测试不替代 HPC4 CUDA、真实模型和 Judge 的验收；容器工作流需首次构建验证。v1 每个训练作业一张 GPU、一个进程，可以并行跑不同单元，但多卡分布式训练尚未开放。

完整 EOS hidden dimension 默认不降维。dense optimal-design 的时间/内存随维度增长，应先测一轮设计成本，再决定是否另开投影消融；不要为了省计算默默改变 reward 表征。无正则 BT 在可分数据上可能没有有限 MLE，日志只报告数值拟合状态，不冒称存在性证明。评估中的优势尚待实验验证。

PPO 接口依据 [TRL 0.15.2 官方文档](https://huggingface.co/docs/trl/v0.15.2/en/ppo_trainer)；版本固定，升级时必须重跑接口测试。
