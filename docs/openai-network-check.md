# OpenAI网络复核与到期额度安排（2026-11-16）

不能断言集群完全没有连接OpenAI的方法。本次核实的是当前直连和站点代理路径，
以及已工作的本机路径，没有向集群发送OpenAI key。

CPU作业1980960在bn012完成19秒：OpenAI IPv4直连超时，官方代理
174.0.250.13:3128访问OpenAI也超时；同节点经该代理访问DeepSeek得到未授权401，
证明代理到DeepSeek可达。集群对api.openai.com的DNS结果与本机公共DNS结果不同。

1980962在同节点用现有Singularity中Python再次测试，19秒完成。域名CONNECT超时；
对本次公共DNS返回的两个地址建立CONNECT得到200，但保留api.openai.com SNI并
验证证书的TLS握手随后被断开，尚未到OpenAI HTTP/API层。因此不能归因为缺少key，
也没有因为CONNECT 200就报告API调用成功。本次没有关闭TLS验证。

本机经现用代理对Luna/Sol模型元数据均HTTP200。随后各一次真实控制题调用也成功：
Luna medium/2048用141输入、17输出token，4.438秒；Sol high/4096用141输入、
5输出token，2.515秒；均返回正确方向。它们仅验证可调用性，不代替Judge质量校准。

要从集群直接调用，需要运维修复OpenAI域名解析及HTTPS CONNECT/TLS出口；也可
设计计算任务到本机API执行端的转发。现有导出packet→本机调用→集群导入已实测，
无需等待新网络或转发服务就能采集标签，OpenAI key继续留本机。

用户说明可用OpenAI约56美元，页面仅显示11月18日到期，并选择优先用于V2训练
标注。按北京时间11月16–17日完成主要调用，不依赖18日可用。V2训练Judge现以
OpenAI Sol为候选，先比较high/medium的质量、实际token费用与速度；评价候选Luna
high/medium，Sol high作为强参考。DeepSeek不再默认承担V2主训练标注，v1不变。

用户同时希望扩大标签预算。优先评估2048/4096/8192、两方法两seed、nested_prefix，
最多32768有效标签及12个PPO单元。按实测费用选能装入额度的上限，必要时6144；
不能直接把旧80次Sol抽检的均价当作新训练标签成本，也不为增加数量混用不同Judge。

新增本机共享USD账本支持逐请求预留、实际usage结算、失败/未知请求保留预留，以及
请求响应落盘后断点重放。正式采用前须完成相关测试和小批实测。拟限制新增总费用
54美元、校准3/训练46/评价5，给56美元余额留余量；额度和单价仍以官方账单为准。
当前小批测试之外的V2批量API任务尚未开始。

官方说明：[额度到期](https://help.openai.com/en/articles/8264644-setting-up-and-managing-prepaid-api-billing)、
[Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)、
[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)。
