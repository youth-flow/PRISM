# V4：OLMo3 7B 工程与执行手册

2026-11-18，`v4`从已验收的`v3`提交9239736建立。本次用户要求完成7B工程并push；正式PPO仍待Reward/Design汇报后的决定。

## 模型与比较口径

固定[allenai/Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT/tree/e1452fc572d51966ff4aaeb25118b891eb93e549)，revision `e1452fc572d51966ff4aaeb25118b891eb93e549`。使用SFT阶段，不能换成已进一步对齐的Instruct或Think。架构Olmo3ForCausalLM，32层、hidden4096。官方要求Transformers≥4.57，V4固定4.57.1；不修改V3的4.52.3环境。

模型FP32、TF32关闭、不量化，几何FP64。冻结EOS Reward头4096→1、无bias，共4096个训练参数。保留固定tokenizer的原生chat template；批处理使用既有工程的词表外padding sentinel，避免把模型可能生成的原生pad误作无效动作。没有扩大模型embedding/输出词表或屏蔽原词表概率。

此次比较整个SFT模型及表征，涉及代际、规模、SFT数据和模板差异，不能解释为只改变参数量。冻结7B线性头也与1B LoRA Reward属于不同模型类别。

## 数据和Reward诊断

`prepare-data --reference-run "$V3_RUN"`校验原数据receipt和逐文件SHA，锁定V3的159682训练比较、2048 Reward开发比较、32768策略上下文、512 PPO开发和500最终评价上下文的原身份、顺序与人工标签。所有回答用7B tokenizer完整重新编码；超长直接报错，不能静默截断、过滤或重采样。`cohort-audit`独立比对身份和标签，忽略模型特有token长度。

1B token ID、特征、logp、K、Design、Reward、定标及SFT回答不能复用为7B资产。EOS特征取最后一层；规范人工回答logp包含EOS。每次只投影32个token，仍计算完整词表log-softmax；训练时分块重计算，避免保留整段词表张量。

全池冻结头诊断可直接从`phi_A-phi_B`拟合，不依赖K/Design。L2=.001/.01/.1，同一2048开发集按BT NLL选择；报告训练/开发准确率、NLL、Brier、ECE和来源子集，独立复算梯度和指标。500题不选参，诊断不覆盖正式Reward。

### 单次训练的线性 Reward 学习曲线

2026-11-18 用户授权先做全池冻结线性头，完成 Reward 后汇报。用 seed 11 打乱有效 HH 训练对，同一个无偏置线性头从零开始，FP64 Adam、学习率 0.001、显式 L2=0.01、每步最多 256 对，完整走一遍。批次在 10000、50000、100000 和全量边界切开，以该步实际对数归一。每个 checkpoint 保存头、Adam 一阶和二阶矩、步数及已见前缀 SHA。中断时从最后一个已验证 checkpoint 继续。

`linear-test-data` 从固定 HH raw manifest 重建官方测试集的留出部分，按上下文排除整个训练池、2048 Reward 开发对、PPO 开发及最终 500 题。重复比较折叠，标签冲突剔除，使用同一 7B tokenizer 完整编码。可与训练特征提取并行运行 `linear-test-features`。特征归并后执行 `linear-reward-train`，独立 `linear-reward-report` 加载四个已保存的头，核对真实步数和前缀，分别重算已见训练对、共用开发集和留出测试集的准确率、BT NLL、Brier 与 ECE。测试集只进入评分。产物在 `diagnostics/linear-reward-curve`，包含四个权重和学习曲线 CSV。

若计算节点无法访问公开模型服务，从本机获取固定版本并核验官方 LFS SHA256 或 Git blob ID。把文件和 SHA256 manifest 上传到独立 online1 目录，CPU 节点用 `download-model --source-manifest /absolute/path/to/verified-model.json` 校验并安装离线快照。

## 六阶段与新预算

| 阶段 | V4实现 |
|---|---|
| 数据/共享SFT | 原人工cohort重新编码、32片7B特征；重新生成共享SFT定标资产 |
| Design | 全池双回答loglinear代理K、surrogate ridge=.01；原trace目标/simplex/联合L-BFGS-B，matrix ridge=.001/.01/.1，原gap≤1e-4 |
| 偏好/Reward | Empirical/PRISM按剩余权重顺序不放回；公共开发规则选L2；固定7B SFT均值/标准差定标 |
| PPO | seed11，两方法各65536 episodes、batch128/minibatch64/4epochs，4096真实Adam更新；独立critic预热/裁剪/概率检查/恢复 |
| 评价生成/KL | 固定500题、三模型，分片覆盖审计，序列MC KL保留原始负值 |
| Luna/汇总 | medium、AB/BA、TIE=.5，三比较3000有效判断，按prompt bootstrap；独立V4账本 |

N和选中λ保持null；新Design报告后由用户重新决定。V3的14000/.001不自动转移。求解目标/可行域/solver不改，不加floor、entropy或均匀混合。付费Judge金额也尚未授权，不能挪用V3的$5。

PPO统计参数沿用V3，显存参数改为micro1、accumulation64、rollout forward batch1。actor LR1e-5，critic LoRA3e-5/head1e-4，r16/alpha32/dropout0；其余逐项见[配置](../configs/experiment.v4.yaml)。

actor、critic和Reward有独立模块树，只共享不可训练的基座Parameter存储。actor/critic LoRA与critic头完全独立，Reward视图没有LoRA；禁止merge或修改共享基座。非重入gradient checkpoint和分块词表投影节省激活内存。真实tiny OLMo3与三份独立基座的PPO更新数值对照通过，不能代替7B GPU验收。

`policy-smoke`在真实7B上做2条工程episode、actor/critic Adam更新、最长2048+512序列反向、Reward不变检查和checkpoint恢复。工程数据不计入正式PPO。实际显存、吞吐和全量ETA须测定，目前不能填入未经测量的值。

## 隔离运行层和集群入口

保留原Singularity SIF，在online1新目录安装[固定wheel层](../containers/requirements.v4-overlay.lock)，不修改旧SIF。新运行层用`runtime.sha256`逐文件封存；提交与计算节点核对其seal以及原SIF SHA。只在Slurm计算节点安装/计算。

1. 本机下载Linux CPython3.11 wheels，上传到online1独立V4输入目录；生成逐wheel SHA及`wheels.sha256`，核对上传字节。
   清单必须使用LF换行；Windows用`Path.write_bytes(...encode("utf-8"))`写清单，避免把CR当成Linux文件名的一部分。失败输入应保留原SHA，不覆盖后冒称首轮通过。

   ```powershell
   python -m pip download --no-deps --only-binary=:all: --platform manylinux2014_x86_64 --python-version 311 --implementation cp --abi cp311 -r containers/requirements.v4-overlay.lock -d local/v4-execution/linux-wheels
   ```

2. 复制[环境模板](../scripts/yancheng/env.v4.example.sh)到Git外，填写已提交V4源码checkout、独立目录、wheels manifest SHA和新的runtime路径。配置复制到独立incoming/configs，设只读并保留相对runs布局。
3. 检查`submit_v4.sh --dry-run runtime-bootstrap --config "$CFG"`，单次启用提交，CPU安装层。`build_v4_runtime.py`拒绝覆盖旧层。取得seal SHA写回私有环境，再提交`cpu-smoke`、`runtime-check`。
4. `download-model`在CPU分配内匿名下载固定模型并逐文件记录SHA；`model-smoke`在真实GPU验证4096维FP32特征、完整词表/分块logp和padding。

```bash
# 每个必要阶段先检查dry-run；全局PRISM_ALLOW_SUBMIT保持0。
bash scripts/yancheng/submit_v4.sh --dry-run model-smoke --config "$CFG" --execute
PRISM_ALLOW_SUBMIT=1 bash scripts/yancheng/submit_v4.sh model-smoke --config "$CFG" --execute
PRISM_ALLOW_SUBMIT=1 bash scripts/yancheng/submit_v4.sh prepare-data --config "$CFG" --reference-run "$V3_RUN" --execute
PRISM_ALLOW_SUBMIT=1 bash scripts/yancheng/submit_v4.sh cohort-audit --config "$CFG" --reference-run "$V3_RUN" --execute

# 其余片分给已验收空闲GPU，保持全局32片index，不重复领取。
PRISM_ARRAY=0-6%7 PRISM_ALLOW_SUBMIT=1 bash scripts/yancheng/submit_v4.sh extract-features --config "$CFG" --shards 32 --execute
# 32片全部通过后merge-features；三档full-pool-reward CPU拟合；full-pool-reward-report。
```

不可变源码release、普通账号/分区/QOS、单任务1GPU、最多50个个人running+pending展开任务、afterok和文件SHA约束均保留。不hold/release、提权或绕过管理员限制，不空占GPU，不在登录节点运行模型/矩阵。数组只允许独立特征/共享生成/评价生成。OpenAI key始终不入集群。

| 任务 | 初始申请 |
|---|---|
| Runtime/bootstrap/CPU验收 | 4CPU/6GB |
| 7B特征/共享/评价生成 | 1GPU/4CPU/64GB主存 |
| model/policy/shared smoke | 1GPU/8CPU/64GB主存 |
| 4096维几何/Design/Reward | 32CPU/48GB，符合q_amd_share每核1.5GiB |
| 特征归并 | 16CPU/24GB |
| 正式PPO（当前关闭） | 1GPU/8CPU/64GB主存，先通过该GPU真实7B验收 |

优先A10080GB；4090另做7B验收，不能用旧1B凭证代替。稳定后30分钟检查，异常/关键节点立即处理。

## 放行与审计

工程默认不启动全量V4/PPO/API。将来用户明确批准PPO后，私有`RUN/control/policy-admission.json`需包含`authority=explicit_user_after_v4_reward_review`、`user_authorized=true`、`config_sha256`和`receipts`，后者绑定cohort-audit/model-smoke/policy-smoke/study-selection/design-report/reward/training-audit/normalization各receipt的SHA。Python重新核验且要求真实policy-smoke passed；提交还要求`PRISM_V4_POLICY_ADMISSION`指向该文件，resume绑定其SHA。工程不自动生成放行文件。

本机`python -m prism.v4.evaluation run-local ...`默认预览。另获新金额上限后显式`--execute --budget-usd ... --ledger .../v4-eval.sqlite`；拒绝复用V3/V2账本。唯一锁、结算和未知费用预留使用已有实现。本次API调用0。

V3已经完成的结果与权重、Design、14k选样、定标和SFT资产保留。V4复用通用HH/优化/采样/审计模块，但schema、配置指纹、输出目录及费用身份独立。不把V3完成状态记为V4完成。

## 本次工程验收记录

2026-11-18：主工程`f2589ff`已普通推送origin/v4并核对一致。生产SIF保持原SHA，V4隔离层seal为`b7c0a1828b1f8812113c3e6e102501f7fddcf10757b788ee2eabce622d2aabfa`。

- CPU运行层安装1989080：COMPLETED/0，24秒，离线安装四固定wheel，`pip check`与Olmo3/PEFT导入通过。
- 生产SIF＋V4层全库1989082：COMPLETED/0，163秒；**430 passed、1 skipped**，实际Torch2.7.1+cu126、Python3.11，包含真实tiny OLMo3/PEFT PPO及数值对照。
- 生产运行时检查1989083：COMPLETED/0，22秒；Olmo3注册及全部固定版本通过，没有加载模型或使用GPU。
- 本机原V3依赖环境全库：**407 passed、15 skipped**。本机V4专项及受影响模块59 passed。生产验收后仅将V4报告中的历史N14000字段改为通用`formal_reward_reference`，两版本冻结头及V4链专项另行**5 passed**；训练/推理算法未再修改。
- 三作业共**0.220000 CPU核时、0 GPU卡时、0 API调用**，均已结束，实查本人Slurm队列为空。日志、输入换行失败的原清单、运行层manifest与SHA在本机忽略目录`local/v4-execution`保留；V3/2/main分支指针未动。

以上只验收工程与CPU环境；完整7B权重尚未下载，7B真实model/policy GPU smoke、全池Reward和正式V4实验尚未运行。不得将小模型通过写成7B全量通过。

## 冻结线性 Reward 学习曲线结果

2026-11-18 用户授权在完成7B特征后，先用全部人工偏好数据测试冻结线性头的表现力，并在1万、5万、10万和全量处保存同一条连续训练轨迹的checkpoint。执行源码`acfebe1`已普通推送origin/v4。

模型固定为OLMo3 7B SFT revision `e1452fc572d51966ff4aaeb25118b891eb93e549`。三类GPU的真实7B FP32 smoke均通过。4090和AMD A100短验收峰值显存均为27.247 GiB，分块与完整词表logp误差分别为1.91331e-5和7.39098e-6。159682个训练对的FP32 EOS4096特征由32个全局分片覆盖，实际使用AMD A100和RTX 4090提取。Intel A100完成同一7B验收。

测试集从固定HH官方test重建，按上下文排除训练池、2048 Reward开发对、512 PPO开发题和最终500题。折叠重复比较并排除标签冲突后保留5362对、5361个独特上下文。各来源为helpful-base 1485、helpful-online 716、helpful-rejection-sampled 1719和harmless-base 1442，多数类准确率为50.1305%。测试标签只用于最终评分。

训练使用seed 11固定乱序、无bias线性头、FP64 Adam、batch 256、学习率0.001和L2 0.01，完整训练数据恰好走一遍。四个边界的真实Adam步数为40、197、393和627。边界小批次按实际对数归一，并保存头权重、两阶Adam矩和已见训练前缀SHA。

| 已见训练对 | 训练准确率 | 开发准确率 | 测试准确率 | 测试BT NLL | 测试Brier | 测试ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 61.1800% | 59.2285% | 59.2503% | 0.66151569 | 0.23493984 | 0.02802167 |
| 50,000 | 63.0960% | 60.1074% | 60.8542% | 0.64830877 | 0.22910159 | 0.01311274 |
| 100,000 | 63.3230% | 60.7910% | 61.8053% | 0.64455795 | 0.22738809 | 0.01084811 |
| 159,682 | 63.0941% | 60.7422% | 62.0850% | 0.64407864 | 0.22709493 | 0.01058696 |

从1万对到全量，独立测试准确率提高2.8348个百分点，测试NLL下降0.01743705。全量相对多数类提高11.9545个百分点。10万到全量的测试准确率只增加0.2797个百分点，测试NLL下降0.00047930。本次没有保存逐题配对差异的置信区间，因此该末段小增益只作描述。

32个训练特征分片、5362对测试特征、归并1990148、线性头训练1990149和独立报告1990150均为COMPLETED/0。归并、训练和报告墙钟分别为1分44秒、1分21秒和1分47秒。含首轮计算节点无法直连Hugging Face的失败下载、零分配替代任务、三卡型验收与全部计算，共17.013889 GPU卡时和71.935556 CPU核时，API调用为0。结束时本人Slurm队列为空。

最终summary SHA256为`67180b828a3d501319c9985531f2badd1e4d93c80070be61c04753b768b8674e`，报告receipt SHA256为`a5e23c1e45a8de67597f52946453ff50a76bec6260eeae440cfaaa79ed7af56f`。集群权重保留在`diagnostics/linear-reward-curve/checkpoints`，本机忽略目录副本在`local/v4-execution/linear-reward-final`。

四个结果是同一条一遍训练轨迹的中间快照，各节点评估其实际已见训练前缀。它们不能解释为四个分别收敛的预算实验。该诊断没有替换正式Empirical或PRISM Reward，也没有启动Design、PPO或付费评价。V4的N和选中Design lambda仍为空。
