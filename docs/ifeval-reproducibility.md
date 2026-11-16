# V2 IFEval 评分复现协议

2026-11-16，前两组正式生成的 CPU 审计发现保存的官方 IFEval 分数与独立复算不一致，付费评价未放行。只读诊断 1984709（16 秒、2 CPU、零 API）在同一份 PRISM2048/11 的 541 条原回答上重复评分，定位到 `keywords:letter_frequency`（key1129）和 `language:response_language`（key3305）的标记会变化。仅固定 langdetect 的 seed 仍不能固定字母规则。

固定版本的 Google 官方检查器会为缺省参数调用 Python `random`；langdetect 1.0.9 的检测过程本身也具有随机性。没有重新生成回答或改变数据参数来适配评分。

显式后处理协议 `ifeval_fixed_rng_v1` 在每次完整评分前同时设置 Python random seed=0 与 DetectorFactory.seed=0，结束后恢复调用者状态。原官方源码、数据、严格/宽松定义、541 prompt/834 instruction 分母均保持。种子统一预先固定，不根据分数挑选。共享 SFT 和八组 actor 全部采用同一协议。

新分数写入 `benchmark.fixed.metrics.json` 与 `benchmark.fixed.receipt.json`，记录原分数、逐条标记差异、依赖版本、协议和原文件哈希；原 `benchmark.metrics.json`、生成回答、生成 receipt、SFT 历史审计均保留。每份新分数须两次完整评分逐条完全相等，再由独立 CPU 审计核对标记、token 解码、EOS、长度和分母。Judge packet 显式绑定新评分文件及 receipt，最终汇总引用新分数，不能把原随机结果冒称为固定种子结果。

此修订只影响确定性评分后处理。训练、生成和 Judge 请求内容不变。代码提交时完整集群回归及三份真实全量重复评分仍待验收，不能据此宣称第五阶段完成。

依据：[Google 固定版本检查器](https://github.com/google-research/google-research/blob/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval/instructions.py)、[langdetect 可复现性说明](https://github.com/Mimino666/langdetect#basic-usage)。
