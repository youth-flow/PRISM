# PPO学习率短程对照

用户于2026-11-15授权探索PPO参数。先比较三个学习率设置，再根据实际证据决定
是否扩大搜索。当前不是新的完整采样研究，也不更新原正式实验的结果。

| 组 | Actor LR | Critic LR |
| --- | ---: | ---: |
| A | 3e-6 | 3e-6 |
| B | 3e-6 | 3e-5 |
| C | 1e-5 | 3e-5 |

分别使用原Empirical和PRISM、seed11的固定Reward权重，共六个独立单卡单元。
Reward backbone冻结，Reward线性头不重训；critic保持冻结SFT加线性value头。
既有Reward权重来自v1特征拟合，本轮在一致FP32特征上使用它们，目的是隔离PPO
优化设置，不作为新Judge/FP32 Reward实验或最终方法优劣结论。

每个单元执行512 episodes、8轮rollout、32次真实AdamW更新，保留完整4096
episodes对应的64轮线性学习率日程。batch64、2 PPO epochs、KL系数0.05、
clip0.2、value系数0.1、gamma1、lambda0.95、actor LoRA r8/alpha16均不变。
只新增可选critic_learning_rate；未配置时保持原优化器。独立学习率沿用相同
衰减乘数，保留上游weight decay分组；未加入原循环未执行的梯度范数裁剪。

从2000条policy提示按固定内容哈希留出64条开发、64条确认，1872条训练。
三组共享训练顺序/seed/提示划分，开发与确认提示不进入本轮PPO训练，最终500题
和IFEval不用于选参。256、512 episodes在开发集按同一逐prompt seed、T=1和
512 token上限生成，评价前后完整恢复Python/NumPy/CPU/CUDA RNG。确认集留待
候选选定后使用，不对其反复调参。原PPO可能用过这些policy提示，但所有本轮
策略均从原SFT开始，不从旧PPO权重继续，因此本轮没有沿用其训练记忆。

开发集逐条保存回答、EOS/长度、两套Reward分数、序列采样log-ratio和完整词表
条件KL之和。后者沿当前策略生成的前缀计算条件分布KL，降低chosen-token估计
在近似相同策略时的方差，但仍是对前缀分布的Monte Carlo估计。
每轮保留未更新策略概率闸门、value目标MSE/方差/解释方差及真实AdamW梯度/LR。

参数选择先要求数值/更新审计通过，再比较相对于共享SFT回答的配对开发集
Reward及Reward减0.05×KL。两套Reward量纲不同，跨Reward汇总先除以各自
SFT分数标准差，不直接相加原始分数。检查是否只靠增长回答长度或牺牲另一套
Reward取得改善。短程无清楚差异时保留不确定性，按曲线决定延长或加入更大
actor LR；不以训练loss下降或单个随机样本宣称参数最优。

这一轮不调用Judge/API，不重建Design，不改采样分布，也不覆盖原产物。
所有执行在online1下新diagnostics目录；每个试跑单GPU、8 CPU/64G。
基线生成独立单GPU/2 CPU/16G，完整测试用CPU节点。

执行入口为`python -m prism.ppo_pilot`。本机相关36项通过、3项因本机无TRL跳过；
集群完整检查与真实试跑仍待完成，不能将准备好记为试跑已验收。
