# HPC4 操作手册

由原 `Prospective-Reward-Model/docs/hpc4.md` 与脚本迁移；这里仅保留新 PRISM 流程。已知账户入口 `yyangjo@hpc4.ust.hk`、account `sigroup`、GPU partition `gpu-l20`。资源限制可能变化，提交前以平台查询为准。

## 1. 目录与计算边界

登录节点只做 Git、文件传输、队列查询、sbatch。**不要在登录节点启动模型、Apptainer 计算、CUDA 检查、训练或评估**。下载/验证模型放 CPU Slurm 作业，训练放 GPU 作业。

建议布局：
```
/project/sigroup/$USER/PRISM                     # git checkout
/project/sigroup/$USER/prism/images             # immutable SIF
/project/sigroup/$USER/prism/hf-cache            # shared downloaded snapshots
/project/sigroup/$USER/prism/configs             # frozen run config + input JSONL
/project/sigroup/$USER/prism/logs                # job logs
/project/sigroup/$USER/prism/archives            # final reports/adapters
/scratch/$USER/prism/runs                       # intermediate experiment output
/scratch/$USER/prism/jobs                       # per-job source snapshots/tmp
```
scratch 不作长期唯一备份；旧文档记录过 inactive scratch 约 60 天清理规则，运行时核对现行政策。

```bash
ssh yyangjo@hpc4.ust.hk
squota -A sigroup
squota
sinfo -p gpu-l20
squeue -u "$USER"
git clone https://github.com/youth-flow/PRISM.git /project/sigroup/$USER/PRISM
cd /project/sigroup/$USER/PRISM
# 若初始代码尚在 codex/prism-scaffold 分支，切到该分支。
```

GitHub 远端必须已有代码后才能 clone；本地设置 origin 不等于已 push。

## 2. 容器首次构建

仓库的 `build-hpc4-image` GitHub Actions **只手动触发**，不因普通 push 自动发布镜像。基于旧工程验证过的 PyTorch 2.7.1/CUDA 12.6 基础镜像 digest，增加固定版 TRL、SciPy 等依赖；完整版本表为 `containers/requirements-hpc4.lock`。新 PRISM 镜像仍须实际构建及验收。

首次推送代码后，在 GitHub Actions 手动运行该 workflow。它执行容器内 CPU 测试、`pip check`，发布 `ghcr.io/youth-flow/prism-hpc4:git-<commit>`，上传 build evidence。保存：

- 镜像 manifest digest 与 SIF SHA256（两者不是同一个 hash）；
- `pip-freeze.txt`、构建日志、构建 commit；
- 需要时在 GHCR 将新 package 设为 public，或使用账户自己的 registry 登录。代码不保存 token。

在**分配的 CPU Slurm 节点**按 evidence 中的 digest 拉取镜像，不在登录节点构建/运行：
```bash
apptainer pull /project/sigroup/$USER/prism/images/prism.sif \
  oras://ghcr.io/youth-flow/prism-hpc4@sha256:MANIFEST_DIGEST
sha256sum /project/sigroup/$USER/prism/images/prism.sif
```
此 digest 由首次构建产生，仓库不能预填一个尚未存在的镜像 digest。

## 3. 固定环境与实验配置

复制 `scripts/hpc4/env.example.sh` 到仓库外的 `env.local.sh`，填写真实 image SHA256。复制实验 YAML 到 `/project/sigroup/$USER/prism/configs/run-v1.yaml`，填四组选项，并将 `output` 设为绝对 scratch 路径。输入数据路径也推荐绝对路径。

```bash
source /project/sigroup/$USER/prism/env.local.sh
CONFIG=/project/sigroup/$USER/prism/configs/run-v1.yaml
bash scripts/hpc4/submit.sh --dry-run validate --config "$CONFIG" --ready
```

所有提交要求源码已有 commit、checkout 干净。每个作业验证 commit 与 image hash，并把该 commit 的源代码快照到自己的 scratch，避免运行途中共享 checkout 变化。**开始后不要修改 YAML、输入文件或本地模型目录**；配置和输入的 hash 会进入 experiment manifest，变化需新 output。生产优先使用带 immutable revision 的 HF snapshot，不使用会被覆盖的本地模型目录。

脚本统一带 `--no-mount /opt/knem-1.1.4.90mlnx3`，这是旧 HPC4 运行记录确认的缺失 KNEM bind 修复。保留 Slurm 分配的 CUDA visibility，GPU 作业只允许一张卡，HF 网络默认关闭。

## 4. 分阶段执行

每次只提交一个必要阶段，查看 exit status 和日志后再推进。旧记录中的 L20 QOS 是每用户最多 2 个 running / 4 个 submitted GPU jobs；数组的每个 task 也可能计入 submitted 限额，所以这里不一次性投递大数组。当前配额请查平台。

```bash
# 镜像/硬件 smoke；不依赖你选择的模型
bash scripts/hpc4/submit.sh smoke

# 检查四组配置与数据格式
bash scripts/hpc4/submit.sh validate --config "$CONFIG" --ready

# CPU 下载两个固定 revision；共享 cache 串行写入
bash scripts/hpc4/submit.sh stage-assets --config "$CONFIG"
bash scripts/hpc4/submit.sh model-smoke --config "$CONFIG"

# 先在小规模、独立 output 的配置上验收完整单元，再开正式预算。
bash scripts/hpc4/submit.sh rollout --config "$CONFIG"
bash scripts/hpc4/submit.sh design --config "$CONFIG"

METHOD=prism
BUDGET=256
SEED=11
bash scripts/hpc4/submit.sh sample --config "$CONFIG" --method "$METHOD" --budget "$BUDGET" --seed "$SEED"
bash scripts/hpc4/submit.sh label --config "$CONFIG" --method "$METHOD" --budget "$BUDGET" --seed "$SEED"
bash scripts/hpc4/submit.sh reward --config "$CONFIG" --method "$METHOD" --budget "$BUDGET" --seed "$SEED"
bash scripts/hpc4/submit.sh policy --config "$CONFIG" --method "$METHOD" --budget "$BUDGET" --seed "$SEED"
bash scripts/hpc4/submit.sh evaluate-generate --config "$CONFIG" --method "$METHOD" --budget "$BUDGET" --seed "$SEED"
bash scripts/hpc4/submit.sh evaluate-judge --config "$CONFIG" --method "$METHOD" --budget "$BUDGET" --seed "$SEED"
bash scripts/hpc4/submit.sh aggregate --config "$CONFIG"
```

这些命令是**顺序操作清单，不要一次性粘贴异步提交**；前一步完成后才发下一步。若需要串联一个已提交作业，设置 `PRISM_DEPENDENCY=<job_id>`，下一次提交会使用 `afterok:<job_id>`；用完 `unset PRISM_DEPENDENCY`。`submit.sh` 输出纯 sbatch job ID，方便留档。

## 5. 放行标准与恢复

- `smoke`：硬件/版本断言、全部 CPU 数学和微型 PPO 接口测试通过。
- 真实模型小配置：EOS 和 context 检查、Judge A/B 合规、BT 梯度、PPO 更新与 adapter 重载、独立评测都通过。
- `design.report.json`：两个优化设计都达到 gap 阈值。未收敛不得先花标签；用新 output 调整统一 FW 求解预算。
- `labels.receipt.json`：有效标签数等于预算，实际 Judge calls 和 invalid calls 都要报告。
- `reward.report.json`：梯度/权重范数无异常；可分数据需明确解释，不静默改变训练目标。
- `summary.json`：只有所有方法/预算/seed 完成才标记 complete。

已完成阶段拒绝覆盖。Judge 中断可按原命令续跑；其他阶段中断后保留失败日志，把该 stage 不完整产物移到单独备份目录再重新运行，或使用新 output，不能把半成品当完成 checkpoint。PPO 自动 checkpoint resume 不在 v1 范围内。

完成后将结果 CSV/JSON、调用日志、receipt、LoRA adapter、配置和源码 commit 归档到 project 路径。候选池可保留在 scratch 并另做受控备份。主仓库不会跟踪模型权重、cache、运行结果或账户密钥。
