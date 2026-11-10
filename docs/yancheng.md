# 是石科技盐城集群：PRISM 平台

## 已核对的信息（2026-11-09）

已阅读本地计算服务资料中的上机手册、快速入门和账户说明，并通过已登录环境独立建立 SSH 连接查询元数据。SSH 主机公钥与 Xshell 已保存的公钥核对一致。没有提交作业、申请计算节点、加载模型或执行 GPU 运算。

| 项目 | 实际结果 |
|---|---|
| 登录入口 | `174.0.250.88:22`，先连接 SSLVPN |
| 登录节点 | `psn001`–`psn003`；本次查询落在 `psn002` |
| 当前账户共享根目录 | `/online1/yangyanxin/yangyanxin`；`$HOME/online1` 是指向它的符号链接 |
| Slurm account | `sc001` |
| 当前可见 partition | `q_intel_gpu_nvidia_A100_share` |
| 资源描述 | A100 80GB PCIe；单节点 8 GPU、64 CPU；GRES 显示 `gpu:tesla:8` |
| 可选 QOS | `normal,standby`；脚本不擅自覆盖平台默认 |
| Apptainer 模块 | `amd/apptainer/1.2.5` |
| 其他模块 | `amd/python/3.11.2`、多个 CUDA/Anaconda/Singularity 模块 |
| 登录系统 | CentOS 7，x86_64 |
| 本机工具 | Git 1.8.3.1、Bash 4.2；平台脚本避免 `git -C` |

手册中 GPU 示例使用 `q_intel_gpu_nvidia`，CPU 示例使用 `q_amd_share` / `q_intel_share`。它们不等于本账户当前的授权队列，不能照抄。CPU-only 阶段暂指向当前可见的 A100 partition，但不请求 GRES；其调度/计费规则仍需平台确认或未来无执行预检，不冒称已有独立 CPU 队列。

手册不同章节给出的存储额度不一致，`df` 也只反映整个并行文件系统容量；账户实际额度以服务方确认/配额查询为准。

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

后续完成构建后，把 SIF 通过 Xftp/SFTP 放到 `prism/images/prism.sif` 并填写 `PRISM_IMAGE_SHA256`。集群若能拉取 GHCR，也可在允许的软件准备环境中按 manifest digest 拉取。此次不启动容器构建、容器测试或 GPU 验收。

`job.sbatch` 加载 `amd/apptainer/1.2.5`，挂载完整共享根目录，隔离容器 home，将 HF/cache/tmp 都落在共享目录；不设置其他站点专用 mount workaround。CUDA visibility 由 Slurm 提供；只有 GPU 阶段使用 `--nv`。不通过 `mpirun` 启动当前单进程 PyTorch 程序。

## 当前可用的准备与检查命令

```bash
cd "$HOME/online1/PRISM"
source "$HOME/online1/prism/env.sh"
bash scripts/yancheng/preflight.sh
bash scripts/yancheng/submit.sh --dry-run smoke
```

`preflight.sh` 只查询目录、Git、磁盘、队列、账户和模块定义，不调用 `sbatch`、`srun`、容器或 Torch。

当前实验定义已收束到 `configs/experiment.yaml`。后续准备资产和数据、镜像验收完毕并决定开始实验后，才将 `PRISM_ALLOW_SUBMIT=1`，按配置生成的计划执行：

```text
smoke -> stage-assets -> prepare-data -> validate --ready -> model-smoke
rollout -> design
每个 method/budget/seed: sample -> label -> reward -> policy
evaluate-reference（共享SFT一次）
全部单元 evaluate-generate -> 全部单元 evaluate-judge
aggregate
```

使用 `bash scripts/yancheng/submit.sh STAGE --config /online1/.../frozen.yaml ...`。GPU 阶段申请 `--gres=gpu:1`、1 node、1 task；CPU 阶段不申请 GPU。所有阶段都显式设置 account、partition、CPU、内存、时间、共享 working directory 和分开的日志。不会申请独占节点。当前配置只支持单进程单 GPU。

资源由Python配置层计算：`python scripts/run.py resource --for-stage label --config ...`。API Judge返回cpu、本地HF Judge返回gpu；将结果通过wrapper的`--resource cpu|gpu`传入，例如`submit.sh --resource cpu label --config ...`。这样Bash不复制provider判断或解析YAML。默认提交开关仍关闭，`--dry-run`可以检查命令。

API调用还需显式开启实验config里的`execution.allow_paid_api_calls`。密钥只引用环境变量，不进入config或日志。Apptainer cleanenv模式下，wrapper通过只读metadata命令获取当前角色配置的变量名，将任务环境中的普通密钥变量自动转发为`APPTAINERENV_<api_key_env>`，也支持预先设置前缀变量；不会把值放在`--env KEY=value`命令参数中。密钥应在仓库外的私有环境中提供。镜像增加官方IFEval依赖后需要重新构建，不能沿用旧依赖镜像冒称已验收。

提交前要求源码 commit 已固定且工作区干净，配置在共享目录内，镜像 SHA256 正确。执行时再次核验，并把源码快照放到该作业目录。需要依赖时设置 `PRISM_DEPENDENCY=<job_id>`，下一次提交使用 `afterok`，用完清除。每次只提交已准备好的必要阶段。

## 完成与待验收

此次迁移完成的是接口、存储布局、源码部署和不执行计算的检查。运行时 SIF、真实驱动兼容性、CPU-only partition 接受规则以及最终模型/Judge 的运行验收，必须在后续对应阶段核实。不要把 Slurm 能查询或 dry-run 正确解释为 GPU 训练已通过。
