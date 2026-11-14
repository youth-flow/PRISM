# PRISM 工作约定

- 默认使用 `main` 分支；用户另有明确要求时再调整。
- 用户已授权：每个重要、可验收的工作节点，在完成相关检查后，主动将本任务的变更 `git add`、`git commit` 并 `git push origin main`，无需再次询问。
- 提交前检查 diff 和工作区状态，只纳入本任务的相关变更；不提交密钥、模型权重、运行产物、虚拟环境或无关修改。
- 使用普通 push，不擅自 force push 或覆盖远端历史。推送后核对本地与远端提交；失败时如实说明未同步状态。

## 当前计算平台

- 平台为是石科技盐城集群，入口在 `scripts/yancheng/`、说明在 `docs/yancheng.md`。所有文件放在 `online1` 共享存储下。
- 用户已明确授权开始冻结配置中的实验。模型生成、Design、Reward、PPO 和计分在集群计算节点执行；DeepSeek 训练 Judge 在集群 CPU 节点经官方代理调用，OpenAI/Luna 评价 Judge 在本机调用。只将 DeepSeek 密钥部署到 online1 下仓库外的私有文件；OpenAI 密钥保持本机。默认保持 `PRISM_ALLOW_SUBMIT=0`，仅为已检查的必要阶段单次设置为 1。
- 集群 DeepSeek 使用 `prism.cluster_judge`，本机 OpenAI 使用 `prism.local_judge` 数据包。冻结实验配置保持关闭直接 API 调用，各执行器在显式执行时仅在进程中授权对应角色。
- CPU 可充分用于并行和数值计算，线程数跟随分配核心；GPU 每任务默认 1 张，通过独立分片/训练单元利用更多空闲卡，不空占卡。Rollout 分片保持全局 prompt 索引和原始 seed，合并前校验哈希、覆盖及顺序。不得未经适配就将单进程 PPO 改为多卡。
- 已入队作业固定源码 commit。升级平台代码时使用 online1 下独立源码版本目录，保留原 GPU 验收作业的排队位置与原始源码，不因更新当前 HEAD 导致旧作业失效。
- 登录节点 Git 为 1.8.3.1，平台脚本使用 `prism_git` 兼容函数，避免 `git -C`。
- gpu003 的 rootless Apptainer 启动已实际失败（user namespace disabled）；站点提供 Singularity 3.10.4 的管理员安装 setuid starter。原 SIF 在该运行时的 CPU 全套测试和 ready 校验通过；用 `PRISM_CONTAINER_RUNTIME=singularity` 显式选择，并完成 GPU 验收后再放行后续计算。两种运行时的环境变量前缀须匹配。
- `计算服务/` 与 `计算资源/` 包含私人账号与口令，保持 Git 忽略；平台配置、访问令牌和登录资料均不提交到公开仓库。
