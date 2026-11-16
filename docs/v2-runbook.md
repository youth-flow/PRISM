# V2工程入口与执行顺序

研究实现与全部资产沿用原工程，`v2`分支整理最终执行入口。用户已于2026-11-16
完成审阅并明确确认开始，六阶段执行状态及固定源码见[v2-progress.md](v2-progress.md)。
继续已有run时使用既有配置、账本和packet，不重复执行已经完成的准备命令。

## 配置来源

| 文件 | 职责 |
|---|---|
| `configs/experiment.v2.yaml` | 数据/model revision、预算、Design、Reward、PPO与评价定义 |
| `configs/judges.v2.yaml` | 最终有效Judge执行设置与40/39美元费用约束 |
| 仓库外`prism/env.sh` | 当前账号分区、共享存储、SIF哈希；无OpenAI key |
| 仓库外绝对路径config | 将固定数据/资产指向共享存储，保护immutable release不含数据 |

研究config中的原评价传输档位保持字节，保护已有experiment.json/packet。
有效profile将评价覆盖为Flex/900秒；只允许这两项执行覆盖，模型、推理强度
和研究参数不能被profile改掉。导出/导入收据记录实际设置、profile哈希。

本机查看最终配置和六阶段任务数，无模型加载、无API、无作业：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -X utf8 -m prism.v2 plan --config configs/experiment.v2.yaml --profile configs/judges.v2.yaml
```

`missing_inputs`明确列出本机没有的集群资产，不能把配置结构通过称为GPU验收。
最终矩阵为12组Reward/PPO、32768有效训练观察、49152 PPO episodes、18000
有效评价判断、7033条IFEval生成（含共享SFT），不重复给嵌套预算计费。

完整容器代码回归使用一次4CPU/6G/15分钟的CPU分配：

```bash
PRISM_CPUS=4 PRISM_MEMORY=6G PRISM_TIME=00:15:00 \
  bash scripts/yancheng/submit_v2.sh --dry-run cpu-smoke --offline --tests
```

实际执行需单次开启提交开关。它测试tiny模型/模拟Judge和官方checker，不调用
真实API、不训练选定的1B正式实验。不可变source通过PRISM_ASSET_ROOT引用共享
官方checker/NLTK资产，不要求把资产放进Git release。

## 冻结源码与配置

本机确认`v2`已提交且干净，运行`git bundle create local/v2.bundle v2`。将bundle、
`scripts/yancheng/prepare_release.sh`及同目录`common.sh`一起上传到仓库外
`prism/incoming/v2/`，因此不要求原main checkout已经包含新部署脚本。登录节点执行：

```bash
source "$HOME/online1/prism/env.sh"
BUNDLE="$PRISM_ROOT/incoming/v2/v2.bundle"
COMMIT="$(git bundle list-heads "$BUNDLE" refs/heads/v2 | cut -d ' ' -f 1)"
bash "$PRISM_ROOT/incoming/v2/prepare_release.sh" \
  --bundle "$BUNDLE" --commit "$COMMIT" \
  --target "$PRISM_ROOT/source-releases/$COMMIT"
export PRISM_REPO="$PRISM_ROOT/source-releases/$COMMIT"
cd "$PRISM_REPO"
```

配置路径使用canonical `/online1`。`prism.v2 freeze-config`只生成路径映射和哈希：

```bash
python -m prism.v2 freeze-config --config configs/experiment.v2.yaml \
  --asset-root /online1/<用户路径>/PRISM \
  --run-root /online1/<用户路径>/PRISM/runs/olmo1b-small-v2-openai \
  --output-config /online1/<用户路径>/prism/configs/v2.experiment.yaml
```

以上Python命令需要已有的Python3.11与基础依赖环境；如果登录节点未提供相同
依赖，在已有CPU验证作业环境中执行，不在登录节点安装或跑模型。
既有run继续使用相同输入和配置内容；不改原manifest或复制其他run的标签冒充。

## 候选、Design和抽样

现有FP32几何有独立证书，可通过CPU阶段`fixed-candidates`显式导入：
`--original <v1run> --pool-checks <四份检查目录> --geometry-check <几何目录>
--independent-check <LU检查目录>`。该实现再次核对原tokens来源、b/K、新q哈希
和原q空间gap，拒绝覆盖已有pool/Design。已导入的当前run不重复执行此阶段。

随后在CPU计算节点执行：

```bash
bash scripts/yancheng/submit_v2.sh --dry-run v2-sample-export \
  --config "$CFG" --profile "$PRISM_REPO/configs/judges.v2.yaml" --execute
```

通过后产物为`local_judge_packets/`四份最大预算packet、12份queries及
`v2.prepared.receipt.json`。已存在的相同训练packet保留原ID；程序重新核对
任务、Judge和各预算前缀，兼容已经标注的原始结果。

## 本机有预算的OpenAI执行

下载packet，原始结果留原目录。为它们生成manifest：

```powershell
python -X utf8 -m prism.budgeted_judge prepare `
  --packet local\<empirical11>.packet.json --packet local\<empirical23>.packet.json `
  --packet local\<prism11>.packet.json --packet local\<prism23>.packet.json `
  --profile configs/judges.v2.yaml --ledger local/v2-api-budget.sqlite `
  --output local/v2-training.json --workers 8
python -X utf8 -m prism.budgeted_judge inspect --manifest local/v2-training.json
```

当前已有账本必须继续使用，包含先前校准和标注。`init-ledger`仅用于完全没有
历史消费的新实验，会拒绝覆盖文件并默认暂停；当前不调用它。profile额度与
历史账本合同不同不意味着重建账本，实时约束必须仍在39美元内。

当前这份已标注run已经生成并离线核对的manifest是
`local/v2-review/training.manifest.json`，指向原四份packet、原JSONL和原账本。
继续该run时使用它，不另行复制一套packet ID或创建新费用账本。

若代理高并发不稳定，可保留原manifest，并使用只降低workers的执行副本；
packet/结果/账本/profile和status路径保持相同，必须先确认原worker已退出并核对。
本次执行副本`training.workers4.manifest.json`为每流4、总16并发。受限恢复可单次传
`-NetworkRetries 2 -MaxNetworkResumes 6`，两项默认均0。后者仅编排执行器自身
网络暂停后的离线核对、30秒退避和无推理费用GET健康检查，不增加每题原3语义
槽位或最后网络重传最多2次。SQL条件恢复保护显式用户暂停；预算、鉴权、格式或
最终槽位耗尽、内部故障、健康检查失败均停止。旧停止状态保存在manifest同目录
`network_pause_history/`，原始失败调用和未知费用全部保留，不得通过重启清除。

停止/离线恢复命令：

```powershell
python -X utf8 -m prism.budgeted_judge pause --ledger local/v2-api-budget.sqlite --reason '配置审阅'
python -X utf8 -m prism.budgeted_judge recover --manifest local/v2-training.json
```

recover不读真实key、不发HTTP，只补记已经派发的缓存响应或未知正文状态；
恢复后仍暂停。之后用户明确开始，才执行resume并用新Windows入口：

```powershell
python -X utf8 -m prism.budgeted_judge resume --ledger local/v2-api-budget.sqlite --reason '用户确认开始V2'
.\scripts\windows\run_v2_judge.ps1 -Manifest local/v2-training.json -Execute
```

不加`-Execute`只检查。key从已有进程环境或忽略的私人文件读取，恢复环境后
不保留到日志。一个账本只允许一个新版worker；内部最多4组×8并发。执行器
有独立owner锁、费用/状态文件、HTTP响应缓存及JSONL，代码已经进入版本控制。

## 离线导入与Reward

上传已经冻结、带SHA256的原始结果快照到online1，CPU阶段
`v2-materialize --results-dir ... --execute`默认要求四条最大流完整。可显式加
`--budgets 2048`，先逐条核对四条已完成2048前缀，再启动该预算的四组Reward/PPO，
同时继续原四条8192标注流；随后同理发布4096、8192。只接受原冻结预算，不修改
config或重抽样、不对前缀额外标注。未齐备前缀不发布，缺失结果不替代。
每个单元使用不可变`labels.materialize.audit.receipt.json`，后续更大前缀不改旧单元
标签或训练审计依赖；后续快照必须与旧前缀逐项相符。总体状态写
`v2.labels.audit.receipt.json`并明确本批实际预算/单元数，不将部分预算声称全量。

12个单元分别运行`reward`，随后`v2-audit-reward --method ... --budget ...
--seed ... --execute`。审计独立重放抽样、Judge seed/顺序、原始A/B方向、BT
目标和梯度。优化器目标gtol=1e-6，沿用原fit的10倍数值验收边界并报告实际值，
不把小梯度等同于Reward泛化良好；泛化属于独立评价问题，不能据最终500题调参。

## PPO、生成和评价

每个`policy`依赖本单元Reward审计成功；每个`v2-audit-policy`依赖其PPO完成。
V2 wrapper还会在模型加载前检查对应审计收据，缺少验收时阻止训练或评价生成。
PPO报告直接记录Actor/Critic AdamW state真实步数（预期均256），每轮概率闸门
和学习率日程；CPU审计再核对完整history、FP32 adapter有限性与已更新LoRA B。

共享`evaluate-reference`按FP32生成500+541条；`evaluate-generate`依赖本单元
PPO验收和共同SFT。相同预算/seed两方法产物齐备后，CPU`judge-export`：

```bash
bash scripts/yancheng/submit_v2.sh --dry-run judge-export \
  --config "$CFG" --stage evaluate-judge --method prism --budget 8192 --seed 11 \
  --packet /online1/<用户路径>/prism/incoming/<packet>.json \
  --execution-profile "$PRISM_REPO/configs/judges.v2.yaml"
```

Empirical负责Empirical对SFT，PRISM负责另外两个比较；各budget/seed均导出。
下载后用同一费用账本和新版执行器执行Luna；再上传并CPU`judge-import`，
最后`aggregate`。不在计算节点调用OpenAI，不占GPU等API。

源数据、输出、checkpoint、packet、cache、密钥都不进Git。重要检查通过后
提交本任务代码到`v2`并普通push；main保留原状态，不force push。
