# PRISM 核心算法审计 — 2026-11-10

## 结论

对当前已冻结的有限候选 PRISM 定义，未发现 surrogate、K_hat、最优设计目标、梯度、FW/away-step、抽样方向或 reward/PPO 冻结链路的数学实现错误。审计发现并修复两处 contraction 性能问题，同时纠正线搜索复杂度注释；未改变目标、特征、采样约束、超参数或实验配置。

本审计由主代理和两位继承当前模型设置的同系列子代理完成。`review_independence: same-family`，`acceptance_status: provisional`；它不是跨模型盲审或形式化验证。原始review trace保存在本地`.aris/traces/research-review/2026-11-10_run01/`，不提交公开仓库。

## 1. 数学与实现逐项核对

来源为根目录`实验流程.pdf` A2–A5及已确认的后续调整：本版两方法、单预算，以及固定L2 reward head。原始PDF B2为无正则MLE；增加L2是用户已明确同意的研究设定，不是本次审计偷偷更改。

| 部分 | 实现与检查 |
|---|---|
| EOS特征 | 冻结SFT最后一层EOS hidden state；截断回答额外EOS仅用于特征 |
| log-probability | 全部实际响应token的条件log-prob之和，包含采样EOS，不包含prompt或人工补EOS |
| b拟合 | 只中心化目标logp，不中心化feature；求解sum平方损失+alpha倍L2 |
| 候选概率 | 在每个prompt内对b^T phi做softmax |
| K_hat | 每个prompt的概率加权协方差再平均；没有误加跨prompt均值项 |
| 比较边 | 无序a<b，v=phi_a-phi_b；canonical标签1表示a胜出 |
| Empirical | 均匀prompt、再均匀该prompt的边；J固定时等价于全边均匀 |
| 设计目标 | A=lambda_design I+sum q_e v_e v_e^T；min trace(K_hat A^-1)，q在simplex |
| 梯度 | -v_e^T A^-1 K_hat A^-1 v_e |
| FW gap | q^T g-min g；有ridge时不能误写为max sensitivity-objective |
| Away步 | 负gamma区间[-q_e/(1-q_e),0]对应可行的去质量方向；允许边权降到零 |
| 线搜索 | A(gamma)=(1-gamma)A+gamma lambda I+gamma v_e v_e^T，ridge没有丢失 |
| 收敛 | 末尾重新计算真实gap；未达容差则converged=false，采样阶段阻止生成标注查询 |
| Reward | 均值BT+lambda_RM/2倍平方范数；不加importance weights或辅助logit |
| PPO | actor更新LoRA；reward使用独立冻结SFT特征；reference为关闭LoRA的SFT |

两个独立恒等式用于交叉核对：

1. K_i = sum_(a<b) p_ia p_ib (phi_ia-phi_ib)(phi_ia-phi_ib)^T。这与weighted covariance的实现一致。
2. d trace(K A^-1) = -trace(A^-1 K A^-1 dA)，因此上述梯度符号和矩阵顺序正确。

## 2. 实际修复

`src/prism/design.py`中K_hat和sensitivity的三元`np.einsum`原先使用默认未优化路径，现加`optimize=True`，使其采用等价的优化contraction。结果定义和归一化不变，仅浮点求和顺序可能产生正常舍入差异。

线搜索仍含dense二次型`a @ T @ a`，每次评价为O(d^2)。原先O(d)注释不准确，已修正。Sherman–Morrison省去的是每个试探步重新求逆，并没有把dense二次型降成线性复杂度。

主代理单BLAS线程、合成数据的kernel复测（sensitivity取三次中位数，K为单次计时）：

| kernel / 规模 | 原耗时 | 优化后 | 比值 | 最大相对数值差异 |
|---|---:|---:|---:|---:|
| sensitivity：2000边×128维 | 0.02062s | 0.000969s | 21.27 | 3.58e-15 |
| sensitivity：2000边×256维 | 0.07596s | 0.002745s | 27.67 | 2.72e-15 |
| sensitivity：2000边×512维 | 0.30401s | 0.009405s | 32.32 | 1.84e-15 |
| K：200 prompts×4候选×256维 | 0.05682s | 0.001400s | 40.58 | 2.00e-15 |

这是对应kernel的本机小规模计时，不是完整设计阶段或正式2048维数据的加速承诺。计时原始数值保存在本地`local/prism-algorithm-audit.json`，不作为实验性能结果发布。

## 3. 独立数值验证与回归测试

- Surrogate与增广矩阵最小二乘解对照，不复用normal equation作为参考。
- K_hat与上述pairwise covariance恒等式对照。
- 梯度做方向/坐标有限差分；FW目标与独立构造A、直接solve并由SLSQP求解的结果对照。
- 两条正交边的解析最优、rank-one/zero K、重复边和会触发away/drop的边界案例。
- 刻意令FW仅迭代一次且容差严格，确认不能进入标注查询阶段。
- tiny随机初始化OLMo2架构对照EOS reward及逐token logp，不仅使用GPT2替代架构。
- n=1024、d=2048、可分rank-one样本的L2 BT，对照一维精确最优条件根。
- 真实TRL对tiny OLMo2做一次CPU PPO更新：LoRA_B确实变化，adapter可重载，FrozenReward全部state_dict逐元素不变。

子代理附加35个小型随机实例中，b/K_hat/梯度最大误差分别为2.22e-15、1.11e-15、5.15e-11；FW目标较SLSQP最多高4.90e-12。另一个主代理30实例压力检查（含不可观测方向的rank-deficient v）中，均通过设定FW容差；相对参考目标差异最大9.30e-7。

持久回归测试位于`tests/test_prism_design_contract.py`与`tests/test_prism_model_contract.py`，连同原`test_math.py`已通过11项针对性检查；全套测试为82 passed、1 skipped（Windows符号链接权限），详见`docs/verification.md`。

## 4. 明确边界与后续顺序

| 可以据此确认 | 尚不能据此声称 |
|---|---|
| 已冻结公式与实现一致，且有独立数值参考 | K_hat等于真实PPO几何或精确policy regret |
| 固定L2惩罚BT存在唯一有限最优解 | 原无正则MLE理论自动适用于L2估计器 |
| tiny OLMo2的模型接口/冻结链路通过 | 真实1B BF16/A100全部路径已经验收 |
| 当前单预算iid抽边、每draw新标注语义正确 | 真实Judge满足BT或必然产生人类偏好改进 |
| 未收敛设计会阻止标注 | 正式2048维特征必然在默认500次内收敛 |

后续仍按既定顺序准备真实资产、运行模型预检、检查真实pool上的FW gap与数值条件，再开始收费标注；本次没有执行这些集群/真实模型任务，也没有消耗API额度。

未来多预算时，当前`budget_coupling=independent`表示各预算重新独立标注；抽边RNG仍使用同一seed，因此query序列会共享前缀。它不影响本轮单预算，亦不违反当前“独立标注”约定；若后续要求跨预算独立抽边，届时应明确变更随机化协议，而不是现在默默换掉它。

## 5. 正式数据的稠密初始表示修复 — 2026-11-14

真实Design1978113完成500次但未收敛：目标8.105633→1.137937，最终gap
1.060549，高于原阈值1.137937e-4。未收敛闸门阻止全部四份query和收费标注。
CPU诊断1978256复算原目标/梯度，发现17971条活跃边中17746条仍有相同的
2.1786777e-5初始质量。away方向选中零特征边，单步可删除质量仅2.1787252e-5，
改进上界2.3143172e-5，与实际末尾迭代逐步停滞相符。

修复只改变away-step FW的内部原子表示：令q=alpha0*q_initial+sum alpha_e*e_e，
alpha在simplex上。初始alpha0=1，原全部边顶点仍保留；
conv({q_initial,e_1,...,e_E})等于原simplex，因此可行分布、最优值和全局FW gap
完全不变。alpha0允许降到0，没有固定coverage floor、降维或截断特征。
求解器能整体减少初始混合质量，避免逐个删除一万多条极小初始权重。

对于混合原子的线搜索，A(gamma)=A+gamma*(M_initial-M_q)。Cholesky A=LL^T，
令D=L^-1*(M_initial-M_q)*L^-T=U diag(d) U^T，
则原目标精确等于sum diag(U^T*L^-1*K*L^-T*U)/(1+gamma*d)。
顶点方向继续使用原Sherman–Morrison公式；drop端点将对应系数设为精确零，
避免浮点残余被当作活跃原子。最终证书仍使用原全部边的真实目标/梯度和FW gap。

500次上限、relative gap=1e-4、两项ridge=0.01、完整2048维与初始Empirical
均未更改。新增独立直接求逆的双向线搜索对照，以及18002边的解析最优用例：
原实现500步不收敛，新表示9次检查收敛，目标0.0008999982000036、
gap6.509e-12、无用边质量为0。原有解析解、SLSQP、重复/秩亏边及
未收敛禁止标注检查继续通过；本机数学专项10 passed。真实2048维重算仍需
完整集群检查和原gap验收，不能用小规模结果代替。

源563c5e3的完整集群检查1978284已通过115项，无跳过，包含真实微型PPO与官方IFEval。
真实Design1978299已启动。原失败产物原字节归档、前后SHA256一致。
此处改变了away原子与线搜索策略，属于同一目标的FW变体调整；
不声称迭代轨迹、最终边权或求解器实现与原次完全相同。真实2048维收敛证书仍待完成。

第二次真实1978299也在500步后未收敛（F=1.1264833151，gap=0.9810535348）。
因此上述表示修复不足以解决本次真实数据，未将其记为成功。独立1978385核对
b的正规方程残差3.482e-14，K的pairwise公式相对误差4.149e-16，几何保持原值。

## 6. 同一优化问题的联合权重求解候选

用户已授权改进数值求解，研究问题仍为：

F(q)=trace(K A(q)^-1)，A(q)=lambda I+sum_e q_e v_e v_e^T，
q>=0、sum(q)=1。保持所有边、完整2048维、原ridge与1e-4验收。

`joint_design`以w>=0、sum(w)>0和q=w/sum(w)调用L-BFGS-B。
任意原可行q可取w=q表示，包括零坐标；不设置正下限、熵惩罚或额外正则。
链式梯度为dF/dw_e=(g_e-q^T g)/sum(w)，其中g_e仍为原始梯度。
正的常数目标缩放用于数值条件，不改变原目标的极小点，证书使用未缩放F。

F关于原q是凸函数；归一化w参数化不被声称为凸优化器。因此L-BFGS-B的
success、投影梯度和目标停滞均不是验收证据。最终重新计算原q的完整梯度，
G=q^T g-min_e g_e。凸性保证F(q)-F*<=G；只有G<=1e-4*F(q)才能通过。
若数值优化早停或500步不够，明确converged=false，sample继续拒绝。

显式CLI选项记录算法与warm start；默认旧接口保留。可从同一run的归档
design.npz复用b/K/v/index/概率/Empirical，逐个验证原receipt哈希与当前
pool.receipt哈希；新receipt绑定原归档。新的迭代报告单独保存，禁止改写
旧失败产物。复用warm start只改变数值起点，不改变可行域、目标或研究预算。

本机数学与缓存专项27 passed，包含10个不复用归一化目标/内核的独立SLSQP
参考和原q空间直接solve的gap复核。真实只读1978500正在验证；
尚未以候选求解器成功标志宣称真实收敛或开始正式标注。

真实试算1978500结果：214次联合迭代、234次完整目标/梯度评价，用时348.906秒，
F=0.8930050456508944，G=8.1430078963e-5，G/F约9.11865e-5，
满足原1e-4标准。固定源码18d1ba9的集群检查1978533为132 passed，无跳过。
正式1978547与独立复核1978548已接入依赖链；独立复核使用原q空间的
分块A累加、LU solve和A^-1 v敏感度，成功前不生成训练query。
