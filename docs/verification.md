# 盐城平台迁移验证 — 2026-11-09

本次实际验证：

- 本地全套测试：**13 passed, 1 skipped**。跳过项为 Windows 没有符号链接创建权限的路径逃逸测试；Linux CI 保留该测试。
- 平台脚本测试覆盖 GPU/CPU 路由、account/GRES、依赖参数、无提交 dry-run、目录边界、脏工作区拒绝和测试模式禁止真实提交。
- 原有数学和微型 PPO 产物链测试保持通过，实验采样和奖励算法未因平台迁移改变。
- Python 依赖检查、默认实验配置结构检查及 Bash 语法检查通过。
- 通过核对 Xshell 已保存的主机公钥建立原生 SSH 连接，读取了真实共享目录、账户、队列、资源描述、module 和工具版本。当前 Slurm 队列中本账户没有运行作业。
- 调度网页：本机 direct 超时、现有代理返回 502、登录节点 direct 超时；SSH/SFTP 正常。没有改变系统代理、VPN 或路由。
- 新平台配置默认关闭提交。没有提交 sbatch/srun 作业，没有在集群执行容器、模型、Torch 或 GPU smoke。

运行时 SIF 尚待首次构建/上传和驱动兼容验收；CPU-only 阶段在当前授权 GPU partition 中能否被接受也尚待平台确认。上述检查不等于正式实验已可放行。

源资料包含私人凭证，`计算服务/`、`local/` 与 `tmp/` 已排除 Git。部署记录与本次本机网络诊断保存在本地 `local/`，不随公开代码发布。

## 小规模配置与通用执行接口 — 2026-11-10

- 默认schema2配置：Empirical/PRISM、seeds11/23、单预算1024、OLMo1B；plan实际计算出4个训练单元、4096个有效标签、6000次AB/BA评价、2705次IFEval生成。
- 全套本地测试：**75 passed, 1 skipped**。跳过项仍为Windows符号链接权限；微型CPU PPO与adapter重载保持通过。
- DeepSeek/OpenAI协议、付费开关、错误与usage、TIE/位置还原、断点恢复使用模拟HTTP验证，没有真实API调用。新增评价测试使用保存的合成产物与模拟生成，不冒称真实OLMo效果。容器密钥按config指定变量名安全转发，专项测试仅使用虚构测试值。
- 真正的官方IFEval源码（e6890f85757dd84e27ca6df2dd30651dafad28e0）与punkt/punkt_tab已在隔离tmp目录用于checker验收，测试得到四项正确分母和strict/loose差异；正式data/assets目录没有因此被冒充准备完成。
- 官方checker测试依赖安装在隔离tmp/ifeval-test-deps，不写入研究结果或公开仓库。容器lock同步增加准确版本；langdetect在Python3下显式从固定版本sdist构建。
- 本地pip check、Bash语法检查、config validate、跨工作目录scripts入口与计划计数通过。
- 默认配置准确报告尚缺正式数据与资产；没有下载所选OLMo权重，没有调用真实Judge，没有提交盐城作业。运行镜像需重新构建，真实OLMo/A100、API账户可用性和CPU-only调度尚未验收。

## PRISM核心算法复核 — 2026-11-10

- 逐式核对原始流程A2–A5与当前已确认的固定L2调整，未发现核心数学定义、梯度符号、edge方向或冻结链路错误。详细证据见根目录`ALGORITHM_REVIEW.md`。
- 两处三元einsum启用优化contraction，修正线搜索每次O(d²)注释；目标、特征、超参数和实验配置不变。小规模单线程kernel复测的数值差异约1e-15，不以kernel加速推断正式完整流程耗时。
- 新增增广最小二乘、pairwise covariance恒等式、两边解析最优、away/drop、zero/rank-one K、独立SLSQP与未收敛阻断标注的回归检查。
- 新增tiny随机OLMo2架构的EOS/logp独立对照和真实TRL更新/adapter重载，确认更新前后FrozenReward全部权重逐元素不变；另以n1024/d2048可分rank-one设计对照L2 BT精确一维最优条件。
- 最新全套本地测试为 **82 passed, 1 skipped**；跳过项仍是Windows符号链接权限。没有加载真实1B checkpoint、调用收费API或运行GPU/集群作业。真实EOS数据的条件数和500次FW收敛性仍需后续预检。

## 盐城调度与裸 GPU 验收 — 2026-11-13

- 使用独立OpenSSH公钥建立严格主机指纹校验的非交互连接；没有把登录密码、私钥或VPN凭证写入仓库、命令参数或日志。
- 集群源码通过已验证的Git bundle从`0ea0526`快进到`d7ab54c`；bundle只包含Git已跟踪历史，不包含本地私有资料。
- `sbatch --test-only`仍返回2282年的不可信预计时间；真实最小作业在约半分钟`Priority`等待后正常运行，因此不再把该站点的test-only时间作为可用性依据。
- 最小探针作业`1972446`申请1 GPU、2 CPU、4 GiB、5分钟上限，在`gpu003`实际运行1秒并以`COMPLETED / 0:0`结束；Slurm记账确认分配1张`gpu:tesla`，`nvidia-smi`确认A100 80GB，stderr为空。
- 站点提交钩子要求调用`sbatch`时的物理当前目录位于`/online1`；仅设置`--chdir`不能通过检查。wrapper现从规范化后的`$PRISM_REPO`执行真实提交，并新增回归测试。
- 平台专项测试为 **4 passed, 1 skipped**，Bash语法检查通过。完整本地套件在既有SciPy`cho_solve`路径触发Anaconda原生扩展进程中止，未记录为全套通过；本次Shell提交修复未进入该数值路径。

该探针只验收Slurm账户、GPU分配与宿主机驱动工具；运行时SIF、PyTorch/CUDA组合、真实模型和付费Judge仍未执行，`PRISM_ALLOW_SUBMIT=0`与付费API开关保持关闭。
