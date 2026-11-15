# PPO学习率短程对照

本轮已完成。当前推荐下一轮统一采用C：actor LR=1e-5、critic LR=3e-5、
GAE lambda=0.95，保留KL=0.05、batch64、2 PPO epochs及原LoRA设置。
这是短程筛选后的工程起点，确认集没有证明它在两套Reward上显著优于A。
未启动新版完整实验，原冻结配置及结果不变。

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
后续真实执行结果见下文。

## 首轮执行与扩展

源2cd8862的集群完整检查1980618为182 passed；SFT开发/确认基线1980622完成。
A/B/C的Empirical作业1980623/24/25、PRISM作业1980626/27/28均已提交，后续
独立CPU核对1980635依赖全部六组完成。入队时各自固定源码，不随HEAD更新。

第一批512结果：Empirical A/B在64开发题的完整回答相同；C的自身Reward均值
13.8020，相对SFT13.8331无清楚增益，完整条件KL和约0.00230。A/B约0.000202。
这支持再探索较大actor步长，不把1e-5直接定为最优。为识别稀疏末端奖励的
信用分配影响，追加D/E，每组仍在两套固定Reward上各512 episodes：

| 组 | Actor LR | Critic LR | GAE lambda |
| --- | ---: | ---: | ---: |
| D | 3e-5 | 3e-5 | 0.95 |
| E | 3e-5 | 3e-5 | 1.0 |

C→D只提高actor学习率；D→E只改变lambda。gamma仍1，Reward、KL系数、batch、
PPO epochs、clip和日程均保持。lambda1使用完整Monte Carlo回报减value基线，
减少value尚不准确时由lambda<1引入的bootstrap偏差，同时可能增加方差，需要
开发验证。尚未证明该设置更好。总搜索限定10个512-episode单元；不做大网格，
也不使用确认集挑设置。新四组在其固定源的集群检查通过后启动。

## 完成结果

D/E固定源a9b80d4，完整集群检查1980648为183 passed。四个试跑
1980650/51/52/53均COMPLETED。A/B/C六个试跑也全部COMPLETED。
每个512 episodes/8 rollout/32次真实AdamW步，合计5120 episodes/320步。
开发检查共10×2×64=1280条策略回答。以下为512检查点相对共同SFT的自身Reward
变化；两列来自不同Reward模型，不直接相加原始分数，也不是Judge胜率。

| 组 | Actor/Critic LR | lambda | Empirical ΔReward | PRISM ΔReward | Empirical / PRISM 序列KL |
| --- | --- | ---: | ---: | ---: | --- |
| A | 3e-6 / 3e-6 | .95 | -0.16065 | +0.15828 | 0.000202 / 0.000164 |
| B | 3e-6 / 3e-5 | .95 | -0.16065 | +0.15828 | 0.000202 / 0.000164 |
| C | 1e-5 / 3e-5 | .95 | -0.03111 | +0.15279 | 0.002298 / 0.001838 |
| D | 3e-5 / 3e-5 | .95 | -0.42077 | -0.05821 | 0.021434 / 0.020525 |
| E | 3e-5 / 3e-5 | 1 | -0.11618 | -0.44388 | 0.017295 / 0.028072 |

按预先约定的标准化开发目标排序C>B≈A>D>E。C的均值排序支持把actor提高至1e-5
作为下一轮起点；D/E的更大策略偏移没有相应开发收益，不推荐直接提高到3e-5
或改lambda1。各组均数值稳定，没有把D/E记为发散或算法错误。

A/B在两套Reward的64题开发集上最终完整回答均相同；独立critic学习率并非没有
生效。固定同一批SFT完整EOS回答（51条）的终点目标核对为：

| 固定终点目标MSE | 初始化 | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| Empirical | 269.832 | 267.088 | 243.343 | 243.298 |
| PRISM | 64.479 | 63.319 | 53.734 | 53.723 |

这说明critic拟合确实加快，但512 episodes仍未充分拟合。该终点核对与全序列GAE
误差不是同一指标；lambda不同也改变value targets，不能用E较大的原始value loss
直接判定其critic更差。

## 独立确认

先把C及开发分析SHA256写入selection.json并上传，再提交确认作业，未查看C/A
在确认集的策略回答后重选。四个确认1980680/81/82/83均COMPLETED，分别对A/C
和两套Reward在另64条提示上生成，共256条策略回答。

| 确认集C减A | Δ自身Reward | prompt bootstrap 95%区间 |
| --- | ---: | --- |
| Empirical | +0.48689 | [-0.08448, 1.23360] |
| PRISM | -0.07273 | [-0.49030, 0.29045] |

确认结果没有证明C同时改善两套Reward，两个区间均包含0。C仍是均值筛选后的
后续起点，不能称已经找到最优参数、证明PPO质量问题解决或证明胜率提升。
本次只有训练seed11；prompt区间不包含训练seed之间的变异。参数选择未使用
最终500题、IFEval或付费Judge；Reward沿用旧v1权重，新Judge/预算实验仍需
在新的Reward上做统一开发检查。

## 下一轮参数建议

建议统一设置actor=1e-5、critic=3e-5、lambda=.95、gamma=1、KL系数=.05、
policy/value clip=.2、vf_coef=.1、rollout batch64（micro2×accum16×mini2）、
2 PPO epochs、LoRA r8/alpha16/all-linear。保留response cap512和一致FP32。
梯度范数裁剪仍未在此训练循环执行，不能把配置中的max_grad_norm=1当成已生效。

KL和policy clip在短程里均未成为主要限制：C的序列KL约0.002，最大平均policy
clipfrac约0.34%；降低KL约束或放宽clip没有当前证据支持。batch64在单卡的
实测CUDA峰值34.887 GiB，符合现有4090资源。未为选参改LoRA秩或response cap。

正式训练量先保留4096 episodes（64 rollout/256 AdamW步），在1024/2048/4096
独立开发检查；512条短程不能证明4096条已充分训练。只有共同开发曲线仍上升
时再考虑所有研究单元统一8192。下一轮也应重新检查value拟合，不能只靠更大
actor LR解决其不足。当前没有启动4096/8192的新正式训练，也未改旧冻结配置。

GAE的偏差/方差依据见[原论文](https://arxiv.org/abs/1506.02438)，实际优化循环
仍固定[TRL 0.15.2](https://github.com/huggingface/trl/blob/v0.15.2/trl/trainer/ppo_trainer.py)。

## 验收与资源

独立CPU验收1980635核对六组、1980657核对十组，最终1980686连同四个确认全部
通过：receipt哈希、每组224个有限FP32 LoRA张量/6029312参数、112个已更新B、
2049个有限critic参数、保存特征的独立Reward点积和提示/seed顺序均通过。
本机再次核对实际32次step及各角色的完整64轮LR日程、全部开发与确认的配对
统计和复制文件哈希。原Design与最终正式审计SHA256均未改变。

所有20个分配均COMPLETED，执行窗口2026-11-15 22:36:21–23:57:41，约81分钟，
合计5.8611 GPU卡时、45.3778分配CPU核时。CPU按0.04元/核时约1.82元，GPU费用
需按平台GPU单价另计；API费用为0。GPU及CPU作业均已结束，无占用或排队。
私有原始记录在local/ppo-pilot和online1下diagnostics/ppo-pilot-*；公共仓库只
提交实现与报告，不提交权重、回答产物或私有配置。
本机final.local.audit.json SHA256为
12983787d716ff464e3951383b076fec5321441cda9618828fed9ea756645001，
分析/确认/账本及该审计已复制到online1仓库外的incoming/ppo-pilot-local-12983787d716，
服务器再次核对该审计SHA一致。
