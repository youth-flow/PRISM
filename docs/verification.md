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
- 集群源码通过已验证的Git bundle从`0ea0526`持续快进到当前`main`；bundle只包含Git已跟踪历史，不包含本地私有资料。
- `sbatch --test-only`仍返回2282年的不可信预计时间；真实最小作业在约半分钟`Priority`等待后正常运行，因此不再把该站点的test-only时间作为可用性依据。
- 最小探针作业`1972446`申请1 GPU、2 CPU、4 GiB、5分钟上限，在`gpu003`实际运行1秒并以`COMPLETED / 0:0`结束；Slurm记账确认分配1张`gpu:tesla`，`nvidia-smi`确认A100 80GB，stderr为空。
- 增强探针作业`1972523`以同样的单GPU小资源配置运行3秒并正常完成，确认驱动`550.54.15`、Apptainer`1.2.5`、A100 80GB和`/online1`共享挂载在计算节点同时可用。
- CPU-only探针未生成作业：站点提交钩子明确拒绝在当前A100 partition中不申请GPU。账户当前看不到`q_intel_share`或`q_amd_share`，因此独立CPU partition是正式执行前的外部前置条件；wrapper已对占位符和复用GPU partition fail closed，禁止用A100空跑CPU阶段。
- 站点提交钩子要求调用`sbatch`时的物理当前目录位于`/online1`；仅设置`--chdir`不能通过检查。wrapper现从规范化后的`$PRISM_REPO`执行真实提交，并新增回归测试。
- 平台专项测试为 **6 passed, 1 skipped**，Bash语法检查通过。当前`main`的GitHub Python 3.11锁定环境完整CI通过；本机全局Anaconda仍在既有SciPy`cho_solve`路径触发原生扩展进程中止，不把该本机环境用于正式实验。

该探针只验收Slurm账户、GPU分配与宿主机驱动工具；运行时SIF、PyTorch/CUDA组合、真实模型和付费Judge仍未执行，`PRISM_ALLOW_SUBMIT=0`与付费API开关保持关闭。

## CPU权限、本地API与启动前审计 — 2026-11-14

- `sacctmgr`确认`sc001/yangyanxin`对`q_amd_share`具有`GrpTRES=cpu=2560`的关联，分区为243个128核节点、193288 MB/节点、1500 MB/CPU；8核与20节点×128核的`--test-only`请求均被接受且没有入队。CPU预测日期异常地落在2031年，不能作为真实排队依据，必须以短真实作业验收。
- GPU分区单节点提供8张A100 80GB、64 CPU、1031000 MB，默认8 CPU/GPU；1卡和8卡测试请求均被接受。查询时`AllocTRES`显示8卡全部占用，因此不把测试开始时间解释为资源保证。
- `quota`确认个人`/online1`配额1 TiB、600000文件，审计时使用约3.3 GiB/159文件。站点提交必须从字面`/online1/...`物理目录调用；审计日志已移出Git仓库，远程源码恢复干净。
- 本机经Sakuracat代理用真实密钥各执行一次最小Judge请求：`deepseek-flash`返回A（135输入/1输出token），`gpt-6-luna`返回A（154输入/16输出token，其中9 reasoning），解析、usage和模型权限均通过。密钥位于Git忽略目录，正式付费开关仍关闭。
- 集群SIF大小3540611072字节，声明与实际SHA256均为`c1b4e80d731cbb3124825ce81c3dcd5226865478cf051db165ec27a0856d6896`；尚未在CPU节点执行，也未加载真实1B模型。
- 启动审计确认正式prompt、IFEval源码/NLTK资源、数据manifest和HF cache仍未准备；结构校验通过但`validate --ready`未通过。正式顺序因此是CPU smoke→资产/数据→ready校验→GPU model smoke，而不是直接提交rollout/PPO。
- 平台wrapper新增`cpu-smoke`、cleanenv代理转发和按阶段资源默认值，避免以统一8核/64GB/8小时错误套用所有阶段；提交总开关保持fail closed。

## 实验启动、离线资产与本机 Judge — 2026-11-14

- 用户已明确授权执行冻结的小规模实验，并指定DeepSeek训练Judge、Luna评价Judge均在本机调用。新增哈希绑定的导出/本地调用/集群导入流程；复用原始阶段的seed、顺序、失败重试和计费记录。模拟传输测试覆盖AB/BA、独立重复draw、并发、断点续跑、部分结果导入和内容篡改拒绝。真实密钥不上传到集群，集群config保持关闭直接API调用。
- CPU短作业1976605证明调度与容器启动正常，但官方代理访问Hugging Face返回CONNECT 503。资产改为本机下载、核验并上传；正式数据准备及模型运算仍在计算节点执行。
- 固定OLMo快照只有`pytorch_model.bin`；原stage-assets无条件排除bin导致漏权重，现优先safetensors、无该格式时保留bin。权重2969891414字节，本机SHA256为`338aeb8b3049cc9b19ba1f3e6df4b5280f3e002b5ae9e38328a353d8f072d534`，与仓库LFS哈希一致。
- 离线CPU验收作业1976644在bn037申请4 CPU/6G，55秒完成；**104 passed**，含真实微型OLMo/TRL PPO与完整31阶段CLI、官方IFEval及本地Judge交换测试。峰值RSS619224 KiB。网络明确标记`skipped_offline`。首次全套检查发现NLTK即使资源已暂存仍查询在线索引，修复为校验manifest后离线复用；不改研究算法或超参数。
- 数据准备作业1976650在bn037以4 CPU/6G运行62秒完成。按固定来源、SFT tokenizer和NumPy1.26.4/seed20261109生成3000 Design、2000 policy、500 evaluation、541完整IFEval。训练原始61135条，去重30、超长988，合格60117；测试原始1000条，排除训练重叠2、超长18，合格980。原始来源清单SHA256为`c96eecd982267feda9abf975bf7275d5b643f3cc81ad3982666bea80c07dcab4`。
- `validate --ready`作业1976655以1 CPU运行12秒完成，报告`prepared_inputs_present=true`、`full_input_validation=true`、无剩余缺项；GPU验证仍为`not_performed`。上述作业源码commit为`0eccc375cf202c58bbe207cf4879e2d30853c3d5`。

| 固定数据文件 | SHA256 |
|---|---|
| design.jsonl | `983d6c64c3a9d48226b0ef0f96b4b7c6e6811fc85fd00c7b2a5f081244398217` |
| policy.jsonl | `d4a19d226e6978ff356b7442184e0c1929dd3dda51b323dba3e265c1a59e5fc5` |
| evaluation.jsonl | `e8cb91086dcf83c5c980204bf6494db8140bc6b8e54e7b2ac905343eea12a13f` |
| ifeval.jsonl | `5e180e38c95f64e28bd0796e0469509b884aff83973990b2683b12be062dbf1c` |
| data.manifest.json | `8216275620b7c64b1520f68eedbb3aa95905a7a47a65b12173ee94e71f046801` |

本节点未完成真实1B/A100验收、正式rollout/PPO或批量Judge调用。模型大包上传、
集群端权重哈希及真实GPU model-smoke为接续步骤，不能把CPU测试替代为真实模型效果证据。

## 集群 DeepSeek 与机会式多 GPU 并行 — 2026-11-14

- 用户更新执行位置：DeepSeek训练Judge在集群经官方代理调用，OpenAI/Luna评价Judge保留本机。真实CPU探针1976711在bn030经`http://174.0.250.13:3128`完成DeepSeek请求，返回有效A，109输入/1输出token，请求1.962秒，作业COMPLETED/0:0。只部署DeepSeek key到仓库外私有文件，目录700/文件600；OpenAI key未上传。
- 新增cluster_judge在集群CPU执行请求包、4并发调用及原阶段结果导入；只允许DeepSeek training后端。冻结config、API模型、参数及标签预算均不变，付费许可只在显式执行器进程中开启。
- 共享rollout新增独立prompt分片与CPU合并。每条候选仍使用原全局索引生成seed；全部分片完备、哈希及顺序匹配后才发布pool。真实tiny OLMo的顺序/分片输出文本、token、EOS特征、logp逐项一致。缺失/损坏分片拒绝合并。
- 计划使用8个单GPU数组任务（最多8并发），每个8 CPU/64G；不要求8张同时可用。CPU合并4核/6G，Design16核/24G，BLAS线程数绑定实际分配CPU。四个独立PPO仍各单GPU，未更改batch或episodes。
- release `ed2791ddc0e26072e1638584a94ac176cd232d8e`在bn047的CPU作业1976721完成完整测试：**110 passed**，实际用时67秒，峰值RSS634176 KiB，含真实微型PPO、官方IFEval、本地与集群Judge、分片一致性和调度资源校验。测试未调用真实API或加载1B权重。
- 保留1976660真实GPU验收的原始源码17b1791与排队位置；旧依赖作业1976662、1976663仍PENDING时撤下，没有中断运行或删除产物。新代码位于独立source-releases/ed2791d目录，避免更新原HEAD使已排队作业失效。

## GPU 节点容器启动问题与现有运行时兼容 — 2026-11-14

- GPU验收1976660实际于12:15:34在gpu003启动，12:15:43以FAILED/1:0结束。错误为`Failed to create user namespace: user namespace disabled`，发生在容器启动阶段，尚未执行Torch或加载1B模型。其后序依赖链因此失效，不能继续描述为正常GPU排队。
- 13:18重新登录查询仍只有amd_share、gpu_nvidia_A100组及原A100分区。截图中运维同意开通公共队列，不作为新权限已生效的证据。AMD A100 share、A100_gb（使用要求的gpu:tesla:1语法）及4090 48G的test-only检查均被用户组权限拒绝。
- 当前Apptainer安装只有非setuid starter；站点Singularity3.10.4/3.9.6有root拥有的setuid starter，配置允许setuid。没有更改系统内核参数或安装系统软件，使用站点已提供的正常运行方式兼容原SIF。
- CPU探针1976891在bn076用Singularity3.10.4运行同一SHA256的SIF，63秒COMPLETED/0:0，**110 passed**，随后`validate --ready`通过。此结果证明该运行时与既有用户态依赖、数据和微型研究流程兼容，不代替真实GPU验收。
- 平台支持显式选择apptainer或singularity，按运行时转发APPTAINERENV_/SINGULARITYENV_及缓存、代理。镜像、实验config、模型权重、数据划分及研究超参数不变；GPU短验收仍是后续作业的成功依赖。

## 最终准备条件验收 — 2026-11-14 17:43

- 新权限已生效：用户组新增gpu_nvidia_A100_share、gpu_nvidia_4090_48g_share，AMD A100和4090 48G分区的单卡test-only请求均接受。三个GPU节点各8卡，共24张共享池卡；不等于个人独占24卡。新增GPU关联显示gpu=8限额、CPU关联cpu=2560，normal/standby的已查询QoS限额字段为空。
- 快照：gpu003的8卡全部分配；gpu045的8卡中3张分配、5张未分配（平台标注4090 48GB，未实机验收）；gpu056的8卡中1张分配、7张未分配，另有1卡预留记录，实际可分配数以调度为准。CPU分区243节点/31104核，17:43报告20348空闲核。
- Lustre个人配额复核为1 TiB/600000文件，已用10074800 KiB（约9.61 GiB）/8316文件；不是把整个online1的剩余空间当作个人配额。本机C盘剩余约42 GiB。
- 本地、GitHub、集群主目录源码在检查时均为0d76656；工作区干净。config SHA256仍为98f6b8102fb6ab53d949b9b72b2769a8f89867386e5e676dda3f083fc933816e；SIF、模型权重及全部固定JSONL/manifest再次复核与前述哈希一致。
- 本机OpenAI只读模型元数据请求HTTP200，配置的gpt-6-luna在可见模型列表中；此前真实最小Responses判断通过。DeepSeek重新在正式计算环境验证：1977630在bn030使用Singularity/官方代理，109输入/1输出token，返回A，请求1.854秒，COMPLETED/0:0。登录节点curl超时不被误当作计算节点API故障。
- 真实GPU验收1977631在新AMD队列gpu056运行33秒COMPLETED/0:0：确认NVIDIA A100-SXM4-80GB、CUDA前后向、真实1B加载生成和EOS/Reward适配通过，适配误差约6.31e-8。此前失效的旧队列等待链已撤下，没有正式训练产物被删除。
- 对验收中接近0的logp进一步做内容检查：1977636在gpu056对3个短prompt比较固定tokenizer模板与模型卡无BOS格式；原模板分别生成13/17/28 tokens，回答非空，列表题正确抽出前6项。两次相同的短答案属于高置信输出，不是只生成EOS。保持固定版本自带模板，不改科研格式或参数。额外探针1977632因诊断脚本错误假设GPU节点有/etc/profile.d/modules.sh而在0秒退出；复用正式模块入口后重试成功，该失败未产生研究产物。
- 私有env已备份并切到验证过的Singularity3.10.4、q_amd_gpu_nvidia_A100_share和gpu:tesla:1；提交默认仍为0，阶段执行单次启用。正式rollout、Design、4096训练标签、四份PPO和6000次评价判断均尚未开始，不能把准备验收当作完整实验完成。

## 正式生成启动与并行调度 — 2026-11-14

- 源码faffcbcf78b43096ad63ce7a8124fb930dc01f03增加2–32片、最多8卡并发调度与生成吞吐日志；不改变研究配置。新增32片逆序执行/不等长分片合并和多前置依赖检查。本机专项20 passed、2 skipped；集群1977805为110 passed、2 skipped，53秒完成。两个跳过项是未绑定当前release的官方checker测试资产，官方checker此前已验收；真实微型PPO与分片一致性测试通过。
- 1977811完整输入校验通过；1977812共享SFT基线与1977813数组第0片（实际JobID1977818）已在gpu056运行，其余分片等待资源。1977814合并、1977815 Design为成功依赖。首次候选生成约58 token/s/卡，只是早期吞吐，不据此承诺整个PPO工期。
- 4090独立探针1977820在gpu045以COMPLETED/0:0结束，26秒；GPU名NVIDIA GeForce RTX 4090、显存47800057856字节、计算能力8.9。原SIF、CUDA前后向、真实OLMo生成与EOS reward通过；探针只写独立诊断目录，不覆盖A100验收或正式产物。完整1B PPO显存和吞吐仍未实测。
- 四单元后序按依赖安排在CPU与4090队列，OpenAI调用仍待本机数据包就绪后执行；PENDING不计为已完成阶段。冻结config SHA256仍为98f6b8102fb6ab53d949b9b72b2769a8f89867386e5e676dda3f083fc933816e。
