# V2计算资源：正确性、效率与计费

本指南适用于盐城集群及`v2`分支。研究参数见`configs/experiment.v2.yaml`，
最终有效Judge配置见`configs/judges.v2.yaml`。旧`docs/yancheng.md`保留历史记录；
V2执行入口是`scripts/yancheng/submit_v2.sh`。用户已完成审阅并明确确认开始，
最新六阶段状态与固定源码见[v2-progress.md](v2-progress.md)。既有env的提交开关
仍保持0，仅对已检查的必要阶段单次授权；不重复执行已经完成的准备工作。

## 本机、登录节点、计算节点

| 位置 | 执行内容 | 资源约束 |
|---|---|---|
| Windows本机 | Git、SSH/SFTP、两个OpenAI Judge、费用账本 | 无需GPU；key、账本和响应缓存留本机 |
| 登录节点psn001–003 | 编辑、Git、文件传输、配置/哈希元数据、查询和提交Slurm | 不训练、不生成、不做Design矩阵计算 |
| CPU计算节点 | FP32几何/Design、抽样、导入、Reward拟合、验收、汇总 | 经q_amd_share分配，线程跟随实际核数 |
| GPU计算节点 | SFT基线、模型生成/前向、PPO、评价生成 | 经GPU分区分配；每个任务一张卡 |

分区是调度队列，节点是实际机器，Job是一次资源分配。账号能访问某分区，
不等于独占其中全部GPU或CPU；空闲资源仍由Slurm分配。候选8卡节点不是8张
专属卡。登录节点不因SSH可达而成为可运行训练的计算节点。

## 已验证资源和默认申请

- Slurm account：`sc001`；CPU分区：`q_amd_share`。
- 已完成一致FP32 PPO短试的GPU分区：`q_intel_gpu_nvidia_4090_48g_share`，GRES=`gpu:tesla:1`。
  真实设备验收为RTX4090，可见显存约44.5GiB；FP32 PPO短试峰值约34.9GiB。
- AMD A100分区：`q_amd_gpu_nvidia_A100_share`，对应`PRISM_GPU_MODEL=A100`。
  1981930已通过一致FP32 PPO两轮/8 AdamW实机短验收。
  单独提交一个分区；站点此前拒绝过组合分区请求，V2 wrapper会阻止此用法。
- Intel A100分区：`q_intel_gpu_nvidia_A100_share`，同样对应`PRISM_GPU_MODEL=A100`。
  1981931已使用Singularity通过相同FP32短验收，原rootless失败记录保留。
- CPU关联曾显示`GrpTRES=cpu=2560`，CPU节点显示128 CPU/193288MB。
  此为授权/配置快照，使用前查询当前状态，按实际所需核心提交。
- 排除曾发生starter/Lustre停滞的`bn002,bn009`，日志和故障记录保留。

| 阶段 | 每任务CPU | 主存 | GPU | 并行方式 |
|---|---:|---:|---:|---|
| FP32几何/Design、固定候选导入 | 16 | 24G | 0 | 一个完整数值问题；不盲目加到32核 |
| 抽样/导出、API结果导入 | 2 | 3G | 0 | 处理4条观察流和12个预算快照 |
| Reward、独立Reward验收 | 8 | 12G | 0 | 12个独立单元可并行 |
| PPO | 8 | 64G | 1 | 12个独立单卡单元利用空闲卡 |
| SFT/策略/IFEval生成 | 2 | 16G | 1 | 独立模型单元；候选可用固定全局索引分片 |
| PPO产物CPU验收 | 2 | 3G | 0 | 每单元验收后放行评价 |
| 本机OpenAI | 本机线程 | 本机 | 0 | 每packet 8并发，最多4组，即32请求并发 |
| 最终导入/汇总 | 1–2 | 1500M–3G | 0 | 同预算、同seed的产物配对 |

资源默认值以`common.sh`为准。CPU-only内存按约1500MB/核档位配置；通过
`sacct`记录实际AllocCPUS，不能仅用请求CPU数估算核时。GPU节点上的8CPU/64G
是已验证PPO配置，不按CPU队列的内存比例推断其可分配性。

2026-11-16 11:57–12:02通过真实SSH完成只读复核，另做`sbatch --test-only`，
没有创建运行/排队作业或恢复API：

- 私有`env.sh`实际默认仍为AMD A100，`PRISM_ALLOW_SUBMIT=0`。不能假定它已改为4090；
  每次GPU提交明确配套设置`PRISM_GPU_PARTITION`及`PRISM_GPU_MODEL`，按实时调度选择。
- 4090节点gpu045共8卡、当时8卡均已分配。AMD A100节点gpu056共8卡、0卡已分配，
  但有其他用户的1卡reservation；因此是7张未分配且未被该预留占用的卡，不能宣称8卡
  全部可用，也不能把整个节点误判为不可用。单卡PPO/生成调度试算均被接受。
- Intel A100节点gpu003共8卡、5卡已分配；该分区单卡PPO调度试算被接受，但本项目
  尚未用该节点验证Singularity与一致FP32 PPO，不因提交权限通过便直接放行正式计算。
- CPU分区当时约20261核idle，用户CPU关联`GrpTRES=cpu=2560`；节点空闲总量不等于
  本账号可同时使用量。AMD A100关联上限为8卡、提交任务上限50；按阶段窗口提交，
  不一次堆积全部依赖任务。所有分区仍由Slurm执行账号及父级限制。
- Reward 8CPU/12G、Design 16CPU/24G、导入/审计2CPU/3G、PPO单卡8CPU/64G和
  生成单卡2CPU/16G均通过wrapper dry-run及调度试算。试算中的2030年等预测日期
  显然不可靠，不能用作完成时间承诺。
- 个人Lustre配额为1TiB/600000文件，实际约11.67GiB/58755文件；共享文件系统整体
  剩余容量不替代个人配额。SIF哈希、48个活动依赖/packet哈希及本地/集群研究配置
  核对通过；已部署6833940业务代码与当前v2一致，后续差异仅为文档。
- 本账号最终`squeue`为空。212项完整回归1981224仍为COMPLETED/ExitCode0；本轮
  没有新的GPU运行时验收，不把调度试算当作实机FP32验收。

用户已授权合理利用尽可能多的计算资源。4–8卡只是耗时估算场景，不是并行硬上限；
12个独立PPO单元可按实际账号限制、空闲GPU和阶段准备状态并行，不等待凑齐整批卡。
每任务仍保持单进程单卡、已验收CPU/主存档位；不同分区分别提交，不占用他人预留，
不增加重复实验。两类A100随后已通过1981930/31实机验收：各128诊断episodes、
8次真实AdamW、224个有限FP32 LoRA张量/112 B更新，最大零更新approxKL约2.54e-11，
显存34.885GiB。它们目前可与4090共同承接新Reward通过后的正式任务；短试实际
分配约12.5分钟，完整4096 episodes暂按每组6–7小时估算，不能沿用4090耗时。

所有计算任务设置OMP/OpenBLAS/MKL线程数为`SLURM_CPUS_PER_TASK`，关闭tokenizer
额外并行；不让每个Python进程自行用满节点全部核心。16线程优于32线程来自
本项目真实数值基准，不把“CPU够多”当作单任务无限加核的依据。

## 存储与不可变源码

集群源码、配置、数据、模型、SIF、输出与临时文件全部位于物理`/online1/...`。
`$HOME/online1`是符号链接，站点会检查提交命令的实际工作目录；只给sbatch加
`--chdir`不足以修复home侧提交。wrapper会先进入规范化仓库路径再调用sbatch。

```text
/online1/<用户路径>/PRISM/                    原始固定数据/资产与既有run
/online1/<用户路径>/prism/env.sh              私有平台配置，默认PRISM_ALLOW_SUBMIT=0
/online1/<用户路径>/prism/source-releases/SHA/ 不可变源码checkout
/online1/<用户路径>/prism/configs/             绝对路径的冻结配置
/online1/<用户路径>/prism/images/ hf-cache/    已验证镜像与单份模型缓存
/online1/<用户路径>/prism/work/jobs/JOB/       独立源码快照/tmp/容器home
/online1/<用户路径>/prism/logs/ incoming/      日志、传输包及校验
```

使用现有SIF，SHA256为
`c1b4e80d731cbb3124825ce81c3dcd5226865478cf051db165ec27a0856d6896`。
使用站点Singularity3.10.4 setuid运行时及对应`SINGULARITYENV_`前缀；GPU节点
rootless Apptainer此前失败。没有为V2重新下载容器或模型的必要。

源码通过Git bundle只传已跟踪文件。`prepare_release.sh`要求目标在
`/online1/.../prism/source-releases/`，拒绝覆盖已有release；Git1.8.3.1操作
使用`cd`而非`git -C`。已排队Job固定源码SHA，后续分支提交不改变它使用的源码。

模型/缓存只存一份，多任务只读共享；临时目录按Job隔离。定期用`quota -s`
核对个人容量/文件数，`df -h`显示的是整个共享文件系统，不是个人余额。
不要把权重、SIF、运行输出、账号资料、API key或本地虚拟环境提交Git。

## 依赖与资源复用

1. 验证固定输入和镜像；既有12000条候选token可以显式复用，保留其BF16生成
   来源，使用已验收FP32特征/几何。SFT正式评价基线按FP32单独生成，不能用旧
   BF16回答代替；它可与CPU/本机标注并行。
2. Design必须满足原q空间gap并有独立复核，随后生成12份抽样、4份最大预算
   packet。API阶段只处理四条8192流，2048/4096严格复用前缀。
3. 本机Sol调用不保留GPU。原始结果上传后，CPU离线导入并拟合12个Reward；
   独立损失/梯度和标签方向验收通过，再放行各自PPO。
4. PPO一任务一卡，不用未经适配的DDP、不申请整节点独占。空闲卡通过多个
   独立训练单元利用；4卡空闲就运行4组，不等待一次拿齐全部卡。
5. 每个PPO验收通过即可运行其评价生成。每个预算/seed的OpenAI比较包依赖
   同预算同seed两方法及共同SFT产物，不必等待其他预算全部生成完成。
6. 最后按prompt聚合AB/BA、seed和bootstrap，KL保留有限样本负值，IFEval用
   官方分母。不能把两次展示顺序当成两道独立评价题。

候选分片必须保留全局prompt索引、候选索引和原seed；合并核对完整覆盖、顺序
及每片哈希。调度拆分不改变研究数据或随机数定义。当前主候选池已有可复用
产物，计划中不能重复生成后再不说明其来源差异。

## 作业命令与检查

先在登录节点加载私有env，固定`PRISM_REPO`到本次不可变release，再构造命令：

```bash
source /online1/<用户路径>/prism/env.sh
export PRISM_REPO=/online1/<用户路径>/prism/source-releases/<完整SHA>
# 每次显式选定已验收卡型；下例是4090，不假定私有env的默认分区。
export PRISM_GPU_PARTITION=q_intel_gpu_nvidia_4090_48g_share
export PRISM_GPU_MODEL=4090
cd "$PRISM_REPO"
CFG=/online1/<用户路径>/prism/configs/v2.experiment.yaml
bash scripts/yancheng/submit_v2.sh --dry-run policy \
  --config "$CFG" --method prism --budget 8192 --seed 11
```

正式提交时仅对已检查的必要阶段单次设置`PRISM_ALLOW_SUBMIT=1`，环境文件中
仍保持0。后序任务设置`PRISM_DEPENDENCY=前置JobID`；多个前置用冒号连接，
wrapper提交`afterok`，不能绕过失败闸门。CPU阶段不加`--gres`；API不走Slurm。

```bash
squeue -u "$USER" -o '%.18i %.24j %.10T %.10M %R'
sacct -j <JobID> --parsable2 --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,MaxRSS
sinfo -o '%P %a %D %t %C %m %G'
```

稳定作业每15分钟检查一次，长时间无变化可30分钟；失败、OOM、阶段完成和
关键验收立即检查。PENDING/Dependency只是等待，RUNNING才是分配运行，
COMPLETED还需ExitCode=0和产物验收通过。站点曾给出2031/2282年的预测，
不能把这些`--test-only`日期当成可信等待时长。

CPU费用按实际`AllocCPUS×ElapsedRaw/3600×0.04元`计；GPU记录实际分配卡数
乘时长，GPU单价另以平台账单为准。数组按真实task分配统计，不用整个排队
窗口冒充GPU卡时，也不把Job根行和batch/extern重复相加。

## 本机费用与故障恢复

OpenAI key只在本机进程环境，两个Judge共用同一既有费用账本；不另开账本
绕过40美元总额。实际使用`prism.budgeted_judge`，不用旧无预算Windows入口。
默认检查/恢复不调用API，付费运行需要显式resume和`--execute`两步。

费用账本、journal/响应缓存放本机本地盘，不放Lustre/NFS网络目录。
本实现用进程内互斥串行写入、事务和数据库触发器约束跨进程额度。
仅SQLite3.51.3+或官方3.44.6/3.50.7回补版本启用WAL；其他运行时使用DELETE
rollback journal，避开官方已公布的WAL-reset问题。本机SQLite3.50.0走DELETE，
不需要安装新依赖。WAL本身仍只有一个写入者，且不适用于网络文件系统。
依据：[SQLite WAL官方文档](https://sqlite.org/wal.html)。

HTTP响应先原子落盘，再结算SQLite；失败/未知费用保留预留。恢复只重放既有
响应，不重复派发成功请求；确实没有响应正文的调用记录为response_unavailable，
不生成替代标签。暂停时停止新派发、收完在途，不强杀付费HTTP。

API可断点恢复；PPO部分输出不等于完成的checkpoint。当前保存的是最终actor，
不能拿部分adapter猜测恢复optimizer/critic/RNG日程。若PPO失败，保留日志/输出，
检查原因、归档后按完整冻结日程重新运行，不能覆盖旧结果后冒称原作业成功。
