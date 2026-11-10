# PRISM v1：数学与代码契约

依据 `实验流程.pdf`；2026-11-10 第一版配置收缩为 empirical、PRISM 两种、单预算1024、两个seed。A-optimal保留为可选实现但本轮不运行。当前执行参数以 `configs/experiment.yaml` 为准，完整执行说明见 `docs/first-study.md`。实现不引入NLHF、DPO或额外辅助logit模型。

## A. 先用无标签池设计分布

固定 reference/SFT `pi_0`，prompt 经验分布 `rho_M = M^{-1} Σ_i δ_xi`，每个 prompt 生成同样的 J 个候选，J 默认 4。所有方法共用一次 rollout。候选按温度 1、top_p=1、top_k=0 采样。

`phi_ij` 是冻结 SFT 最后一层 EOS hidden state，完整维度不自动投影。`ell_ij` 是完整生成响应的所有 token 条件 log-probability 之和，不是平均 token log-prob，也不是 EOS 单 token log-prob。包括实际生成的 EOS；不包括 prompt。

长度截断而没有 EOS 时，为**提取 feature**额外附加一次 EOS，`ell` 仍只计算真正生成的响应 token。这样 phi 对任何截断响应都有定义，RM 与 PPO reward adapter 使用完全相同约定。池里记录 `forced_feature_eos`。每次计算检查额外 EOS 是否超出 context，不静默截 prompt。

按 PDF 原式：

\[
\hat b=\arg\min_b\sum_{i,j}[b^T\phi_{ij}-(\ell_{ij}-\bar\ell_i)]^2+\alpha\|b\|^2.
\]

只对拟合目标中心化，不擅自同时中心化 feature 或添加 prompt intercept。解 normal equation `(XᵀX+alpha I)b=Xᵀt`。

\[
\tilde\pi_{ij}=\operatorname{softmax}_j(\hat b^T\phi_{ij}),\qquad
\hat\mu_i=\sum_j\tilde\pi_{ij}\phi_{ij},
\]
\[
\hat K=M^{-1}\sum_{i,j}\tilde\pi_{ij}(\phi_{ij}-\hat\mu_i)(\phi_{ij}-\hat\mu_i)^T.
\]

每个 prompt 的无序 pair 是一条 edge `e=(i,a,b), a<b`，`v_e=phi_ia-phi_ib`。标签 a=1 始终表示 canonical response a 胜过 b，不随 Judge 展示顺序改变。

\[
A_\lambda(q)=\lambda I+\sum_e q_e v_ev_e^T,\quad
f_K(q)=\operatorname{tr}(K A_\lambda(q)^{-1}),\quad q\in\Delta.
\]

empirical 是先均匀抽 empirical prompt、再均匀抽该 prompt 的 pair。因为固定 J，等于所有 edge 均匀分布。A-optimal 用 K=I；PRISM 用 K_hat；lambda 在两者间完全相同。

梯度：`g_e = -v_eᵀ A^{-1} K A^{-1} v_e`。FW 最优性间隙 `qᵀg-min_e g_e`；**有阻尼时不能误写成 max sensitivity-f(q)**。实现 away-step Frank–Wolfe，在相同 simplex 上优化，并用 Cholesky / 特征分解与 rank-one 公式做线搜索。没有覆盖约束、uniform mixture、强制每条边至少一次等额外机制。所有 q 设计完成并达 gap 阈值后才允许标注。

四个不同参数：alpha 是拟合 b 的 ridge；lambda_design 是设计矩阵阻尼；lambda_RM 是 reward 头固定 L2；beta 是下游 KL 系数。它们不是同一个正则参数。

## B. 分布决定获取哪些新标签

当前每个方法/seed在单预算下抽 `e_k iid ~ q_method, k=1,...,n`，有放回。未来多预算时可配置独立标注或同一观测序列的nested_prefix，不能把前缀复用冒称新标签。重复 edge 是不同 draw，重新调用 Judge；绝不把一次标签复制多份当作新证据。随机交换 A/B 展示顺序并还原到 canonical 标签。

人工oracle由固定训练Judge代替。本版DeepSeek非thinking、T=1，每个draw新调用；API客户端seed控制抽样和展示顺序，不保证供应商采样完全复现。LLM Judge不被假定等于真实人类oracle，其输出也没有被实现证明符合BT；BT是reward阶段的建模假设。评价使用独立Judge角色。

每条有效二元标签计入 n；无效格式会记录并有限次重试。因此同时报告 `valid_labels` 与 `actual_judge_calls`；比较真实标注成本应以实际 calls 为横轴。没有标签的失败请求不伪造 a=0。评价的额外 calls 与训练标签预算分开报告。

\[
\hat w\approx\arg\min_w \frac1n\sum_k
\left[\log(1+\exp(v_{e_k}^T w))-a_k v_{e_k}^T w\right]+\frac{\lambda_{\rm RM}}2\|w\|^2.
\]

用 float64 L-BFGS，数值稳定 `logaddexp`，零初始化，固定求解设置。无伪logit、辅助概率模型、重要性权重或逐方法reward归一化。用户已明确接受固定L2，本版lambda_RM=0.01，对均值损失加惩罚，所有方法/预算一致。正L2保证惩罚目标存在有限唯一最优解，但不是无正则MLE；理论argmin与有限步数值拟合仍区分，梯度/范数/迭代数记入报告。旧无正则入口仍可配置为0；此时可分数据未必存在有限MLE。`max_weight_norm` 只用于报错，不作为约束优化或 clipping。

## C. 一致的下游优化与评估

目标仍为 `E_pi[R_w(x,y)] - beta KL(pi||pi_0)`；PPO 是现成数值求解器。固定 KL 系数、PPO episode 数、LoRA 配置、初始化 seed、policy prompts。TRL temperature 明确设为 1，不加 missing-EOS reward penalty，不 whiten rewards，不用 RM 自己打最终胜率。

三个参数互不混淆：b 仅定义采样几何；w 定义冻结 RM；theta 是 PPO 更新的 actor/LoRA。PPO reference 是关闭 LoRA 的相同 SFT。奖励表征始终从独立冻结 SFT backbone 计算，不使用 actor 的动态 hidden state。critic 使用同一冻结 SFT backbone 加一个独立可训练 value head；此默认在所有方法间一致。v1 单 GPU，episode 数必须整除 rollout batch，避免 TRL 向上取整导致预算不一致。

TRL 0.15.2 的兼容细节：rollout 内部先生成到统一长度，再用保留的 EOS stop ID 截断奖励并屏蔽 EOS 后的 token。这避免 variable-length padding ID 在该版本的 log-prob gather 中越界；EOS 之前生成分布保持不变。不会把 EOS 后的隐藏状态当作 EOS feature。reserved padding 位于模型词表之外且从不用于真实模型输入，因此不 resize SFT embedding、不改变 reference 分布。

同源heldout上各policy各生成一次回答，SFT reference回答共享缓存一次；独立评价Judge对配置的跨方法/SFT比较按AB/BA两种顺序评审，允许TIE并按半胜计分，两个顺序还原后取均值。KL 是 actor 温度 1 样本上的 `log pi(y|x)-log pi_0(y|x)` 序列均值及标准误，不截断负估计值，不把 token 平均冒称序列 KL。报告长度以排查长度偏差。

schema2的IFEval使用固定官方Google checker源码与显式准备的NLTK资源，独立greedy生成上限2048，报告四项官方指标。schema1/可选exact-match适配器仍保留，但不能冒称IFEval。新增benchmark需增加对应适配器，不仅更改名称。

## 解释边界

K_hat 是 loglinear proxy 几何，不是实际 PPO/LoRA Hessian 的精确测量。测试证明的是实现与这些公式/接口的一致性，不证明 LLM Judge 满足 BT、不证明全局 policy regret 或标签节约。各方法共享设置是公平对照的起点，最终结论由独立评测决定。
