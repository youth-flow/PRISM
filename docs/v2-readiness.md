# V2工程验收与暂停状态（2026-11-16）

本页记录用户要求审阅期间的工程验收与暂停快照。用户随后已明确确认开始，
最新执行状态见[v2-progress.md](v2-progress.md)。以下费用/标签计数均为启动前
快照，不覆盖后续实际进展；工程验收本身没有调用真实OpenAI或正式模型/Design。

## 已完成的工程变化

- 公共`prism.budgeted_judge`取代私有目录worker：有预算执行、owner锁、32并发
  上限、暂停/恢复、只读检查和不发HTTP的缓存恢复，统一由源码版本管理。
- 修复32并发写入账本时的SQLite锁超时与二次ROLLBACK错误。相同本地账本的
  进程内写入串行，跨进程由SQLite事务/触发器控制；仅修复版SQLite启用WAL，
  本机3.50.0使用DELETE。HTTP响应先fsync/原子落盘再结算，结算失败可离线重放。
- `prism.v2`给出六阶段参数/资源计划、冻结绝对路径配置、4条最大预算流与
  12个前缀的导出/离线导入、独立Reward损失/梯度检查。
- PPO报告直接读取实际AdamW state计数；`v2_audit`检查Actor/Critic步数、64轮
  概率闸门、LR日程和FP32 LoRA产物。训练目标与PPO更新循环未改。
- V2集群wrapper阻止在集群等OpenAI/占GPU调用API；GPU任务使用一个已验证
  分区、单卡，模型计算前检查对应Reward/PPO审计收据。新增不可变bundle部署。
- 更新README、执行手册、资源使用指南。研究参数和两个Judge配置文件保持
  原字节；Design、Reward数学实现及PPO train loop没有修改。

## 现有标签与费用

旧worker在本次整理前已因SQLite写锁停止。检查到11776有效标签、11780次派发。
先以SQLite backup API备份原账本并复制原始JSONL，再用新版入口离线核对。
一条派发没有可用正文，补记response_unavailable并保留费用预留，没有替代标签。

核对后仍11776有效标签，所有11780派发与日志对齐，无遗漏缓存/未知记录；
`PRAGMA quick_check=ok`。既有成功日志前缀、packet、Judge设置和输入均保留。
本次恢复新增HTTP请求为0。

返回usage对应校准$1.694933、训练$12.233377，合计$13.928310；另有未知费用
预留$0.345652，不作为已知实际消费。全局39美元、校准2/训练35.5/评价1.8
限额不变，未知预留不释放。账本明确paused=true。

私有核对文件为`local/v2-review/offline-recovery.audit.json`及其备份目录，
新版manifest为`local/v2-review/training.manifest.json`；不提交这些运行产物。
集群查询当前无运行/排队作业。新Reward/PPO与正式评价尚未开始。

## 验证

本机已通过费用、32并发/12000旧行压力、响应结算故障恢复、暂停/owner锁、
前缀复用、独立BT检查、调度资源与不可变release等专项测试。Windows符号链接
权限测试有跳过，Linux完整容器回归将覆盖相应路径。Bash及PowerShell语法、
文档相对链接和六阶段计数检查通过。

完整容器回归已在固定源码6833940707bf412b5da4309043ab68979facf53c实际通过：
CPU作业1981224，bn012，4CPU/6G，Elapsed82秒，ExitCode0。结果为212 passed，
无跳过，8条已有tiny单样本方差警告，pytest71.54秒。覆盖真实Tiny OLMo/TRL、
Actor/Critic AdamW计数、官方IFEval、API缓存故障恢复及调度/不可变部署。
本轮验证约0.091111 CPU核时、0 GPU卡时、0真实API调用。

首次部署662a589在站点Bash4.2的空数组nounset兼容处于提交前退出，无Job/费用；
修复为显式分支后，以新不可变source通过公共submit_v2.sh/cpu-smoke入口验证。
后续仅更新验收文档，业务代码与上述验证版本一致；正式实验继续保持暂停。
