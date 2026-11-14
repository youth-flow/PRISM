# 第一版小规模实验：最终配置与执行分层

本版唯一执行配置是 [`configs/experiment.yaml`](../configs/experiment.yaml)。`study-plan.yaml` 只保留说明入口，不再复制参数。配置已经冻结为单一标签预算；用户于2026-11-14明确授权开始实验，并指定两个API Judge均在本机执行。集群仅为已检查的必要阶段单次开启提交。

## 1. 最终实验矩阵

| 项目 | 配置 |
|---|---|
| Sampling | Empirical、PRISM |
| Seeds | 11、23 |
| 每个方法每个 seed 的有效训练标签 | 1024 |
| 总训练标签 | 2×2×1024 = 4096，失败请求另计 |
| SFT/reference | allenai/OLMo-2-0425-1B-SFT，固定 HF revision |
| Design/PPO/test prompts | H4 train_gen 3000 / 另外2000；test_gen 500 |
| 每个 Design prompt 候选回答 | 4；共享12000个回答、18000条无序边 |
| Reward | 冻结 SFT 最后一层完整2048维EOS特征；单线性头；均值BT损失+0.01/2倍L2平方 |
| 训练 Judge | DeepSeek V4.1 Flash，API名 deepseek-flash，非thinking，A/B |
| 独立评价 Judge | GPT-6 Luna，API名 gpt-6-luna，低推理强度，A/B/TIE |
| 同源评价 | KL＋PRISM–Empirical、PRISM–SFT、Empirical–SFT |
| 展示顺序 | 每对回答AB、BA各一次，复用同一对回答 |
| 总同源评价调用 | 500×2 seeds×3比较×2顺序 = 6000，失败请求另计 |
| Benchmark | 全部541条IFEval，官方strict/loose与prompt/instruction四项指标 |
| PPO训练数量 | 4；加SFT共5个模型参加IFEval，2705次benchmark生成 |

不预设PRISM一定提高性能。第一版检验同样标签预算下的策略质量；标签效率曲线留待后续配置。两seeds为探索性起点，seed波动与prompt置信区间分开报告。

所有来源revision、输出位置、预算、seed、Judge协议、比较列表和PPO参数都在config。只取H4的prompt/prompt_id，忽略原有回复与偏好标签；先排除benchmark重叠、规范化去重、按固定SFT tokenizer过滤超长prompt，再精确选取并冻结3000/2000/500个ID。Design与PPO互斥，训练与test/benchmark互斥；不静默减少数量或截断prompt。IFEval保留全部541条原始指令及参数。

每个draw独立获取新标签。同一条edge再次抽中仍新调用Judge。当前只有一个预算，不需要预算前缀复用；框架仍保留可选nested_prefix供未来预算曲线使用。同一seed下两个方法的RM和PPO各自从头开始，不继承另一方法或小预算policy。

## 2. PPO起始参数

按此前核对的盐城A100 80GB资源，单GPU/单进程、BF16、TRL 0.15.2 PPOTrainer。所选OLMo真实GPU尚未验收，以下为统一起始配置而非调优结果。

| 参数 | 数值 |
|---|---:|
| policy prompt池 | 2000条 |
| 每个policy总rollouts | 4096 |
| prompt/response上限 | 1024/512 tokens |
| microbatch | 2 |
| gradient accumulation | 16 |
| 每epoch PPO minibatches | 2，每个minibatch32个样本 |
| rollout batch | 64 |
| rollout批数 | 64 |
| 每批PPO epochs | 2 |
| learning rate | 3e-6，线性衰减 |
| 固定KL系数 | 0.05 |
| policy/value clip | 0.2/0.2 |
| value loss系数 | 0.1 |
| gamma/GAE lambda | 1.0/0.95 |
| max_grad_norm | 1.0 |
| LoRA r/alpha/dropout | 8/16/0 |
| LoRA target | all-linear |
| rollout forward microbatch | 4 |

Actor更新LoRA，关闭adapter即reference；reward始终取独立冻结SFT的EOS特征，不能取变化中的actor hidden state。critic共用冻结SFT backbone但有独立可训练value head。不开reward whitening、不加额外missing-EOS惩罚；采样T=1、top_p=1、top_k=0。取固定最后checkpoint，最终test不用于挑选checkpoint。同一beta不保证同一实际KL。

正L2使均值BT目标强凸且强制增长，保证惩罚目标具有有限唯一最优解；数值拟合仍需检查梯度。它不是无正则MLE，旧的无正则估计器结论不能直接挪用。lambda_RM、设计矩阵damping、surrogate ridge、PPO beta是四个不同参数，碰巧数值相同也不代表等价。

## 3. config → scripts → src

| 层 | 职责 |
|---|---|
| configs/experiment.yaml | 定义实验数据、模型、方法、预算、seed和超参数；密钥只引用环境变量名称 |
| scripts/run.py | 薄执行入口，转交CLI，不含采样、loss或评价算法 |
| scripts/yancheng/ | 只负责调度、容器、资源和平台路径；不复制研究算法 |
| src/prism/config.py | schema、输入合同、路径规范化、实验规模计数 |
| src/prism/workflow.py | 从config生成阶段顺序与CPU/GPU需求 |
| src/prism/data.py | 通用的数据准备与冻结ID/manifest |
| src/prism/design.py、reward.py、hf.py | 采样设计、线性reward训练、模型/PPO接口 |
| src/prism/judges.py | training/evaluation角色与DeepSeek/OpenAI/HF后端适配 |
| src/prism/benchmarks.py | 官方IFEval或旧exact-match适配 |
| src/prism/evaluation_v2.py | 共享SFT缓存、同源三组比较、AB/BA、KL与汇总 |

改已支持的方法组合、seed、预算、模型或endpoint，只改config；新增算法或provider需在src添加适配器/注册项，scripts保持不变。可用方法注册表保留A-optimal，但当前配置不会求解或运行它。旧schema1本地HF接口保存在`configs/hf.example.yaml`，不作为本版默认。

所有数据/资产/output相对config文件解析，与启动命令的工作目录无关。HF/Git源码使用不可变revision；API模型名称是服务别名，日志保存实际返回model、request ID、时间和usage，不能宣称服务输出可完全复现。config与输入内容变化必须使用新output目录。

## 4. 只读检查入口

```powershell
.\.venv\Scripts\python.exe scripts/run.py validate
.\.venv\Scripts\python.exe scripts/run.py plan
.\.venv\Scripts\python.exe scripts/run.py plan --json
.\.venv\Scripts\python.exe scripts/run.py resource --for-stage label
```

这些命令只检查/打印计划，不加载模型、不调用Judge、不提交作业。`validate`区分结构验证与准备文件是否齐全，不把已填满YAML等同于真实GPU验收。`validate --ready`还会核对数据数量、ID、内容互斥、官方checker源码hash与依赖资源，不需要API密钥。

## 5. 未来准备和执行顺序

下面是未来显式执行的阶段，不是本次已运行的任务：

1. 安装锁定依赖并构建运行镜像，包括benchmark extra。
2. `stage-assets`：下载固定SFT资产、官方IFEval源码子集与NLTK punkt/punkt_tab资源。
3. `prepare-data`：只加载tokenizer准备固定数据文件、样本ID和manifest；不加载模型权重做计算。
4. `validate --ready`与`model-smoke`；后者验证真实模型EOS/reward接口，但schema2不自动调用收费Judge。
5. 共享`rollout → design`，然后各单元`sample → label → reward → policy`。
6. `evaluate-reference`只生成一次SFT的同源回答和IFEval回答，保留不可变receipt。
7. 先完成全部`evaluate-generate`，再做`evaluate-judge`，保证跨方法比较双方回答已存在；最后`aggregate`。

Benchmark另用greedy、最多2048新tokens，不受同源KL的512上限约束。IFEval调用原始Google checker，不能用exact-match替代。官方源码/资源在明确的准备阶段下载，验证和计分过程离线。

集群config保持`execution.allow_paid_api_calls: false`，避免集群直接请求API。`prism.local_judge export`将已核验依赖的请求导出为带哈希的数据包；本机`run --execute`只在调用进程中授权付费，使用同一个API适配器；回传后`import`校验配置、依赖哈希和请求内容，并让原有label/evaluate-judge阶段重放真实结果、生成集群路径receipt。调用日志记录usage/失败/重试，目前不宣称有跨进程总费用硬上限或Batch实现。

`resource`命令从Judge后端计算API标注/评价为CPU、本地HF Judge为GPU。当前API阶段使用本地调用路线，集群只运行`judge-export`和`judge-import`两个短CPU操作。数据包包含prompt、回答和固定调用顺序，不包含模型权重或密钥；成功调用断点复用，重复抽中的不同draw仍独立标注。当前receipt包含绝对路径，不能直接搬动已冻结的实验目录；导入通过原阶段生成正确路径的receipt。

Apptainer使用cleanenv；wrapper先读取已验证config中的当前Judge角色变量名，再将任务环境中的对应密钥安全转为`APPTAINERENV_<api_key_env>`，也接受预先设置的该前缀变量。只在进程环境传递值，不放入`--env KEY=value`命令参数或日志。CPU-only请求在当前GPU分区能否被调度接受，仍待未来实际验收。

## 6. 结果定义和证据边界

同源KL始终相对固定SFT，在policy自己的T=1样本上计算完整生成序列log-ratio均值/标准误，包括实际EOS而不包括prompt或人工补EOS；报告回答长度、截断率，不裁掉有限样本可能出现的负估计。它对应明确停止/长度上限下的策略分布。

每对回答的AB/BA判断还原到具体方法，胜/平/负为1/0.5/0再取平均；记录位置不一致。置信区间按prompt配对bootstrap，不能把同一题两次评审当成两个独立prompt。训练标签与评价调用费用分开，失败请求仍计入成本。

K_hat仍是loglinear proxy几何，不冒称真实PPO Hessian。单元测试和微型PPO仅验证实现合同；真实OLMo CUDA、Judge质量、实验收益与省标签结论均需后续实测。
