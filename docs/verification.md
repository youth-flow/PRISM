# Scaffold verification — 2026-11-02

本次交付实际完成的验证：

- Windows / Python 3.11，PyTorch **2.7.1+cpu**；其余实验依赖按 `containers/requirements-hpc4.lock` 安装。
- `python -m pytest -q --tb=short`：**11 passed**。
- `python -m pip check`：通过。
- 默认配置结构验证通过，并正确列出尚待填写的四类、九个选择字段。
- Bash 脚本语法检查、GitHub workflow YAML 解析通过。
- `git ls-remote origin` 成功；本次检查远端无 refs，为空仓库。

测试涵盖：PDF proxy normal equation 和 K covariance；FW 梯度数值差分、单调性、gap，以及与独立 SLSQP 优化器的目标值对照；经验 prompt 权重；BT 数值恢复及标签方向；输入不变性；重复 edge 的独立 draw 调用与断点续跑；EOS hidden reward 对齐和全序列 log-prob；真实微型 GPT-2 / TRL PPO 更新与 LoRA 重载；完整候选—设计—标签—RM—PPO—评估—汇总的产物链。

完整链路中的 Judge 是仅存在于测试中的显式模拟对象，**不是实际大模型 Judge 验收或效果实验**。微型 PPO 使用真实优化器和模型更新。极小测试中的 TRL 单 minibatch 方差诊断可能未定义，输出 JSON 中明确保存为 null；GPT-2 Conv1D 的 PEFT 方向自动修正 warning 也是预期行为。

尚未执行：GitHub 上的新镜像构建、Linux 容器测试、HPC4 CUDA smoke、用户最终选择的 SFT/Judge/benchmark 验收，以及正式标签预算实验。工程已提供这些步骤的入口，不将本地测试称为平台验收。
