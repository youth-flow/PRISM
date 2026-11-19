# V3 全池 LoRA Reward PPO 因果诊断

本路径只回答一个问题：在 V3 的 OLMo-2-1B-SFT 上，已经独立验收的强 LoRA Reward 能否让 PPO 在固定训练／开发探针上可靠优于 SFT 参考。它不是原 Empirical／PRISM 两方法比较，不替换原14k Reward、Design、抽样、归一化或策略产物。

## 冻结输入

- Reward：`diagnostics/full-pool-lora-reward-fresh-head/checkpoints/step-002304.pt`。
- 该 checkpoint 由预定开发 BT NLL 规则从完整全池学习日程选择：已见147,360对、训练准确率71.0987%、开发准确率69.6289%、开发 NLL .56658961。
- Reward base、LoRA adapter和无bias scalar head在PPO期间全部冻结；运行契约绑定checkpoint、训练／汇总receipt及SHA256。
- Actor仍从原SFT＋零增量LoRA开始，绝不复制Reward adapter。

## PPO配方

保持原V3配方，隔离Reward质量这一变量：FP32且禁TF32；actor LoRA r16/alpha32/all-linear/dropout0；65536总episodes；rollout128、minibatch64×2、4 PPO epochs；actor LR 1e-5；critic adapter/head LR 3e-5/1e-4；5% warmup、cosine降至10%；KL .05；policy/value clip .2；gamma1、GAE lambda .95；vf coefficient .1；独立norm1梯度裁剪；不白化Reward、白化advantage；T1、top-p1、top-k0、回答cap512。

正式执行使用单节点同型号GPU的4卡或8卡同步数据并行。每个rank持有同一全局rollout的连续不重叠切片；advantage矩、token分母、指标和梯度均做全局归并，梯度按SUM而不是默认平均。全局rollout128、minibatch64、epoch数、优化步数、学习率日程和prompt采样顺序因此不随卡数改变。4090最坏形状实测将物理microbatch从2降为1、累积从32增为64，仅改变每次前反向的放置粒度，不改变64样本minibatch或梯度数学。checkpoint绑定world size、各rank RNG和训练参数hash，只允许相同卡数精确恢复。禁止把多个独立PPO段训练后平均，也禁止混合4090/A100做同步组。

总horizon保持65536，使用绝对`--stop-after-episodes`分段，不能把配置临时缩成8192，否则会改变学习率日程。允许节点为2048、8192、16384、32768、65536。

## Reward定标与Critic

`lora-ppo-normalize`用同一冻结LoRA Reward重新评分已有4096条固定SFT回答，以FP64、`ddof=0`计算均值和标准差；旧线性Reward的定标禁止复用。同时保存128条固定训练probe和512条开发probe的SFT基线分数及身份。

Critic是独立SFT feature backbone＋独立LoRA＋带bias value head。初始时严格复制Reward adapter的数值，但不共享参数对象或存储；价值头初始化为`reward_head/std`，bias为`-mean/std`。因此初始EOS value精确对应归一化Reward。随后用2048条SFT轨迹、4 epochs、64 minibatch完成128个真实Critic预热更新，held-out explained variance必须大于0且MSE优于常数基线。PPO中Critic继续学习“终点Reward＋逐token KL惩罚”的未来回报；Reward保持冻结。

## 分阶段闸门

1. `lora-ppo-smoke`：真实4卡或8卡一次最坏形状同步更新，每卡选择4个长policy prompt，生成chunk=4、强制每条512 token，并以micro=1完成actor／critic反向；要求各rank参数一致、两者真更新、Reward base/adapter/head版本不变、Critic/Reward初始EOS误差小于2e-5、显存峰值和分布式checkpoint可绑定恢复。这不是正式episodes。
2. `lora-ppo-policy --stop-after-episodes 2048`：强制首停。Critic预热、actor/critic各128个PPO optimizer步、固定train/dev probe、有限性和Reward运行时不可变检查全部通过，才发布工程验收。
3. `lora-ppo-policy --resume --stop-after-episodes 8192`：强制第二停。训练probe平均Reward增益必须为正；512开发prompt的Reward增益和含KL RLHF目标的95%下界必须为正；Critic必须继续优于常数MC基线。另保存KL、长度差、EOS、重复率及Reward增益与长度差相关性。
4. 8192自动性能闸门通过仍不自动继续。检查probe后，只有显式执行`lora-ppo-admit --user-lora-ppo-confirmed`才允许后续16k／32k／65k。
5. 自身Reward增益只证明PPO会优化该代理。若要声称“RLHF有利于SFT”，还必须在未用于训练／选参的500题上做PPO对SFT的独立盲评；本路径不自动调用付费API。

checkpoint先于probe提交。若worker在checkpoint后退出，以相同stop点和`--resume`恢复会先重放／核对该checkpoint的probe和闸门；包括8192回调中断及65536最终发布中断，均不允许跳过或重做策略更新。

## 入口

```bash
# 所有命令还需 --config "$cfg" --execute；以下先做dry-run。
bash scripts/yancheng/submit_v3.sh --dry-run lora-ppo-normalize --config "$cfg" --execute
bash scripts/yancheng/submit_v3.sh --dry-run lora-ppo-smoke --config "$cfg" --execute
bash scripts/yancheng/submit_v3.sh --dry-run lora-ppo-policy --config "$cfg" --stop-after-episodes 2048 --execute
bash scripts/yancheng/submit_v3.sh --dry-run lora-ppo-policy --config "$cfg" --resume --stop-after-episodes 8192 --execute
# 只在用户审阅8192结果后：
bash scripts/yancheng/submit_v3.sh --dry-run lora-ppo-admit --config "$cfg" --user-lora-ppo-confirmed --execute
```

normalize为1GPU/4CPU/24G。smoke和policy默认4张同型GPU、每卡至少8CPU/64G主存；设置`PRISM_DISTRIBUTED_GPUS=8`可在整节点空闲且同型时改为8卡。policy默认24小时只是单次作业时限，不代表完整horizon必然在24小时内结束。配置文件保持原字节，新增路径复用旧run的hash绑定资产并写入独立`diagnostics/full-pool-lora-ppo`、`policy/full-pool-lora`和`lora-ppo-*` receipts。

## 独立盲评

阶段性checkpoint不会伪装成完整65k策略目录。先用CPU入口把已审阅的绝对episode、分布式checkpoint、report和train/dev probe哈希冻结，并保存当时`latest.json`的不可变快照；后续PPO更新live pointer不会破坏该选择证据。再由普通`evaluate-generate --method full-pool-lora`重建SFT+LoRA Actor并只恢复该checkpoint的Actor训练张量。最终500题、prompt级seed、T=1、cap512以及actor−SFT序列log-ratio语义与V3正式评价一致；已有SFT回答必须通过原receipt逐哈希复用。

```bash
bash scripts/yancheng/submit_v3.sh --dry-run lora-ppo-eval-freeze \
  --config "$cfg" --stop-after-episodes 32768 --execute
bash scripts/yancheng/submit_v3.sh --dry-run evaluate-generate \
  --config "$cfg" --method full-pool-lora --shards 8 --execute
bash scripts/yancheng/submit_v3.sh --dry-run merge-evaluation \
  --config "$cfg" --method full-pool-lora --shards 8 --execute
bash scripts/yancheng/submit_v3.sh --dry-run lora-ppo-judge-export --config "$cfg" --execute
```

盲评包独立写入`judge/lora-ppo`，只包含`full_pool_lora_vs_sft`的500题×AB/BA=1000个有效判断，不覆盖原Empirical/PRISM包、结果或账本。集群只生成和导出，明确金额上限前不得运行本机付费Judge；结果使用`lora-ppo-judge-import`和`lora-ppo-aggregate`进入独立汇总。
