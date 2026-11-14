# 是石科技盐城集群：PRISM 平台

## 已核对的信息（2026-11-14）

已阅读本地计算服务资料中的上机手册、快速入门和账户说明，并通过已登录环境独立建立 SSH 连接查询元数据。SSH 主机公钥与 Xshell 已保存的公钥核对一致。没有提交作业、申请计算节点、加载模型或执行 GPU 运算。

| 项目 | 实际结果 |
|---|---|
| 登录入口 | `174.0.250.88:22`，先连接 SSLVPN |
| 登录节点 | `psn001`–`psn003`；本次查询落在 `psn002` |
| 当前账户共享根目录 | `/online1/yangyanxin/yangyanxin`；`$HOME/online1` 是指向它的符号链接 |
| Slurm account | `sc001` |
| 当前授权 partition | CPU `q_amd_share`；GPU `q_intel_gpu_nvidia_A100_share` |
| CPU资源 | 243节点、每节点128 CPU/193288 MB；用户关联并发上限`cpu=2560`；1500 MB/CPU |
| GPU资源 | A100 80GB PCIe；单节点8 GPU、64 CPU、1031000 MB；默认8 CPU/GPU |
| 可选 QOS | `normal,standby`；脚本不擅自覆盖平台默认 |
| Apptainer 模块 | `amd/apptainer/1.2.5` |
| 其他模块 | `amd/python/3.11.2`、多个 CUDA/Anaconda/Singularity 模块 |
| 登录系统 | CentOS 7，x86_64 |
| 存储配额 | `/online1`个人1 TiB、600000文件；2026-11-14使用约3.3 GiB/159文件 |
| 本机工具 | Git 1.8.3.1、Bash 4.2；平台脚本避免 `git -C` |

2026-11-14的关联、分区和`--test-only`检查确认本账户属于`amd_share`与
`gpu_nvidia_A100`组。CPU关联`GrpTRES=cpu=2560`，20节点×128核的测试请求
被接受；GPU分区的1卡和8卡请求均被接受。测试请求没有入队或计费，且不能
视为资源立即可用。查询时8张A100全部已分配；CPU分区有21777个空闲核。
站点`--test-only`对CPU给出2031年的不可信预测，因此仍需真实短CPU smoke
确认调度与容器。个人配额由`quota`确认为1 TiB，而不是`df`展示的全文件系统容量。

## 目录

所有持久文件与临时文件都放在 `online1`。不要使用 home 的小容量文件系统存放模型，也不沿用其他平台的 project/scratch 路径。

```text
$HOME/online1/PRISM/                # main 分支源码，origin 为 GitHub PRISM
$HOME/online1/prism/env.sh          # 本账户环境配置，仓库之外
$HOME/online1/prism/configs/        # 冻结实验配置，路径使用绝对路径
$HOME/online1/prism/images/         # 校验后的 runtime SIF
$HOME/online1/prism/hf-cache/       # 模型 snapshot 和 tokenizer
$HOME/online1/prism/work/           # 逐作业源码快照与临时目录
$HOME/online1/prism/runs/           # 实验输出
$HOME/online1/prism/logs/           # Slurm stdout / stderr
$HOME/online1/prism/apptainer-cache/
$HOME/online1/prism/apptainer-tmp/
$HOME/online1/prism/incoming/       # 离线上传包与校验记录
```

源码可通过 GitHub clone，或用本地 `git bundle` 经 SFTP 上传后在共享目录 clone；后者不要求集群能访问 GitHub。仅上传 Git 已跟踪的源码，避免把 `计算服务/`、本地 `.venv`、口令和大文件传上去。

## 环境与运行时

复制 `scripts/yancheng/env.example.sh` 到仓库外的 `prism/env.sh`，然后 `source`。默认 `PRISM_ALLOW_SUBMIT=0`：真实 `submit.sh` 会退出，`--dry-run` 仅打印命令。

CentOS 7 的主机系统不能据此认定能原生安装新版 PyTorch wheel，因此采用已提供的 Apptainer 模块承载用户态依赖。平台迁移保留已验证的 Python 3.11 / PyTorch 2.7.1 / TRL 0.15.2 版本组合；GPU 型号检查改为 A100，不假定新驱动兼容性已验收。

`containers/prism.def` 与 `containers/requirements.lock` 是统一运行时定义。`.github/workflows/build-runtime-image.yml` 仅手动触发构建，发布到 `ghcr.io/youth-flow/prism-runtime`，记录 manifest digest、SIF SHA256 和 pip freeze。二者的 digest 不能混用。

运行时SIF已放到`prism/images/prism.sif`并以声明的SHA256复核通过。文件存在和
哈希一致不等于运行时可用；仍需`cpu-smoke`与`model-smoke`分别验收CPU依赖、
代理、PyTorch/CUDA、真实OLMo 1B和A100路径。

`job.sbatch` 加载 `amd/apptainer/1.2.5`，挂载完整共享根目录，隔离容器 home，将 HF/cache/tmp 都落在共享目录；不设置其他站点专用 mount workaround。CUDA visibility 由 Slurm 提供；只有 GPU 阶段使用 `--nv`。不通过 `mpirun` 启动当前单进程 PyTorch 程序。

## 当前可用的准备与检查命令

```bash
cd /online1/yangyanxin/yangyanxin/PRISM
source /online1/yangyanxin/yangyanxin/prism/env.sh
bash scripts/yancheng/preflight.sh
bash scripts/yancheng/submit.sh --dry-run cpu-smoke
bash scripts/yancheng/submit.sh --dry-run smoke
```

`preflight.sh` 只查询目录、Git、磁盘、队列、账户和模块定义，不调用 `sbatch`、`srun`、容器或 Torch。

站点提交钩子会检查调用 `sbatch` 的当前物理目录，不能只依赖
`--chdir`。`$HOME/online1` 虽指向共享存储，但从 home 侧符号链接直接
提交会被拒绝；wrapper 在真实提交前会切换到规范化后的
`$PRISM_REPO`（`/online1/.../PRISM`）。用户仍可从任意目录调用
`submit.sh`，不要绕开 wrapper 手工复制长提交命令。

当前实验定义已收束到 `configs/experiment.yaml`。后续准备资产和数据、镜像验收完毕并决定开始实验后，才将 `PRISM_ALLOW_SUBMIT=1`，按配置生成的计划执行：

```text
cpu-smoke -> stage-assets -> prepare-data -> validate --ready -> model-smoke
rollout -> design
每个 method/budget/seed: sample -> label -> reward -> policy
evaluate-reference（共享SFT一次）
全部单元 evaluate-generate -> 全部单元 evaluate-judge
aggregate
```

使用 `bash scripts/yancheng/submit.sh STAGE --config /online1/.../frozen.yaml ...`。
GPU阶段申请`--gres=gpu:1`、1 node、1 task；CPU阶段不申请GPU。wrapper按阶段
设置经审计的CPU、内存和时间默认值：CPU内存遵循1500 MB/CPU；GPU任务使用
8 CPU/64 GB，rollout默认24小时，PPO/评价默认12小时。环境变量覆盖只用于
看过pilot日志后的单次显式调整。不会申请独占节点，当前配置只支持单进程单GPU。

资源由Python配置层计算：`python scripts/run.py resource --for-stage label --config ...`。API Judge返回cpu、本地HF Judge返回gpu；将结果通过wrapper的`--resource cpu|gpu`传入，例如`submit.sh --resource cpu label --config ...`。这样Bash不复制provider判断或解析YAML。默认提交开关仍关闭，`--dry-run`可以检查命令。

API调用还需显式开启实验config里的`execution.allow_paid_api_calls`。密钥只引用环境变量，不进入config或日志。Apptainer cleanenv模式下，wrapper通过只读metadata命令获取当前角色配置的变量名，将任务环境中的普通密钥变量自动转发为`APPTAINERENV_<api_key_env>`，也支持预先设置前缀变量；不会把值放在`--env KEY=value`命令参数中。密钥应在仓库外的私有环境中提供。镜像增加官方IFEval依赖后需要重新构建，不能沿用旧依赖镜像冒称已验收。

提交前要求源码 commit 已固定且工作区干净，配置在共享目录内，镜像 SHA256 正确。执行时再次核验，并把源码快照放到该作业目录。需要依赖时设置 `PRISM_DEPENDENCY=<job_id>`，下一次提交使用 `afterok`，用完清除。每次只提交已准备好的必要阶段。

## 完成与待验收

源码、共享存储、CPU/GPU权限、裸GPU、宿主机驱动、Apptainer模块、SIF文件
哈希和本机两个真实API账户已经验收。仍待真实CPU容器smoke、正式数据/模型
资产准备、`validate --ready`、真实OLMo/A100 model smoke以及本地Judge产物同步
流程。不要把Slurm权限、文件哈希或最小API请求解释为完整实验已通过。

## 本机 API Judge 与集群结果交换（2026-11-14）

用户已授权开始实验，并指定DeepSeek与Luna两种API调用均在本机执行。
集群config仍关闭直接API请求，私有密钥不上传。逻辑阶段顺序不变，
label/evaluate-judge各拆成集群导出、本机调用、集群导入三个执行动作。

计算节点通过wrapper导出一个单元（PATH需替换为实际绝对路径）：

```bash
PRISM_ALLOW_SUBMIT=1 bash scripts/yancheng/submit.sh judge-export \
  --config /online1/.../frozen.yaml --stage label \
  --method prism --budget 1024 --seed 11 --packet /online1/.../prism11.packet.json
```

用scp下载packet，在Windows执行：

```powershell
.\scripts\windows\run_local_judge.ps1 -Packet local\prism11.packet.json -Results local\prism11.results.jsonl -Execute
```

脚本从被忽略的`计算资源/api.md`读取两个key，默认4个并发请求，
只在子进程环境中传入，结束后恢复环境；日志含usage而不含key。
重新运行同一packet/results会跳过成功调用。失败重试也会保留并计入调用量。

将结果上传回online1，再运行：

```bash
PRISM_ALLOW_SUBMIT=1 bash scripts/yancheng/submit.sh judge-import \
  --config /online1/.../frozen.yaml --packet /online1/.../prism11.packet.json \
  --results /online1/.../prism11.results.jsonl
```

导入核对packet哈希、冻结config、输入内容、调用seed、AB/BA和原始判断，
不进行网络调用；已有成功/失败记录与receipt由原阶段代码处理。
judge-export/import只申请1 CPU与1500M内存，不占GPU。

实际CPU短作业1976605于2026-11-14在bn030立即启动，容器和Python正常启动；
计算节点经官方代理访问Hugging Face得到CONNECT 503，作业因此失败。
固定资产改为本机下载、核验并上传，数据准备与模型计算仍在计算节点执行。

离线数据源通过`python -m prism.source_assets --config configs/experiment.yaml --cache local/hf-cache`
在本机下载固定revision的原始train_gen/test_gen parquet和IFEval JSONL，
生成相对路径及SHA256清单`sources.manifest.json`。将hub与清单上传到集群HF cache后：

```bash
PRISM_ALLOW_SUBMIT=1 bash scripts/yancheng/submit.sh prepare-data \
  --config /online1/.../frozen.yaml --source-manifest /online1/.../hf-cache/sources.manifest.json
```

计算节点核对revision/分区/文件哈希，只读取prompt与ID列，并用同一个
prepare_data算法和SFT tokenizer生成固定划分。原始来源哈希写入最终数据manifest，
不依赖计算节点公网，也不在本机运行正式数据筛选。

已确认网络阻断后，可显式使用`cpu-smoke --offline --tests`验收离线容器
及CPU兼容测试；网络结果会明确记录为`skipped_offline`，不能当作联网成功。
完整测试单次申请4 CPU/6G/15分钟，测试中不调用真实API，不加载真实模型权重。

## 当前执行分工与空闲 GPU 并行（2026-11-14 更新）

用户更新分工：DeepSeek训练Judge在集群CPU上调用，OpenAI/Luna评价Judge在本机调用。
DeepSeek探针1976711在bn030使用`http://174.0.250.13:3128`代理，真实返回A，
输入109/输出1 token、请求约1.962秒，作业正常结束。仅DeepSeek key存放于
仓库外`prism/secrets/deepseek-api.key`（目录700、文件600）；OpenAI key不上传。

集群使用`submit.sh cluster-label`（2 CPU/3G、默认4并发）在同一节点导出请求、
调用DeepSeek、导入真实结果。配置及实验合同不变，付费许可仅在执行器进程中启用；
失败/重试和断点记录沿用原阶段。该执行器拒绝OpenAI后端。

共享rollout可拆为8个独立单GPU任务，使用`PRISM_ARRAY=0-7%8`与`--shards 8`。
每任务仍为1张A100/8 CPU/64G；有几张空闲就可并行几片，不要求同时拿齐8卡。
每条回答的seed由原全局prompt索引计算；合并按原prompt/candidate顺序，
核验全部分片与完整覆盖，缺失或改动拒绝发布最终pool。
合并任务用4 CPU/6G，随后Design用16 CPU/24G。
独立PPO保持单GPU，通过四个训练单元并行利用多卡；不更改batch或episodes。

旧排队作业会校验Git HEAD，因此新代码部署为独立源码版本目录；保留原model-smoke
排队位置，替换尚未启动的旧rollout/Design依赖链。SFT基线生成也可在验收成功后
提前与rollout并行，其输出仅作固定评价，不用于挑选训练参数。

## gpu003 容器启动兼容（2026-11-14 13:20）

原GPU验收1976660拿到卡后，在启动Apptainer时因`user namespace disabled`失败。
CPU节点可运行同一SIF，不等于GPU节点的内核隔离设置相同。站点已提供
`amd/singularity/3.10.4`（带管理员安装的setuid starter），可显式选择：

```bash
PRISM_CONTAINER_RUNTIME=singularity PRISM_ALLOW_SUBMIT=1 \
  bash scripts/yancheng/submit.sh model-smoke --config /online1/.../frozen.yaml
```

脚本沿用原SIF和全部研究参数，只切换运行时命令、模块与环境变量前缀。
CPU探针1976891的110项测试与完整输入校验通过；真实GPU仍需短作业验收。
只有该验收成功，才放行rollout和基线生成。

2026-11-14 17:43更新：GPU验收1977631已在gpu056以Singularity3.10.4通过，
实测卡型NVIDIA A100-SXM4-80GB；原SIF与全部研究超参数保持不变。
私有配置与示例默认GPU分区已更新为`q_amd_gpu_nvidia_A100_share`，
GRES为`gpu:tesla:1`，运行时为Singularity。旧Intel A100分区与新4090 48G分区
为备用；4090仅确认提交权限，使用该机型前仍需实机验收。
