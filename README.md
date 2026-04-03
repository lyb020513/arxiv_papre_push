🤖 Arxiv-Digest-Bot：智能学术论文深度追踪与推送系统
Arxiv-Digest-Bot 是一款专为机器人（Robotics）、自动驾驶（Autonomous Driving）及强化学习（RL）领域研究者打造的自动化论文筛选系统。它通过 GitHub Actions 每天定时抓取 arXiv 最新论文，利用 DeepSeek-V3 深度解析 PDF 的引言与结论，从海量文献中为用户全局严选 5 篇最具价值的顶会论文，并推送详尽的中文解析至飞书。

🌟 核心特性
深度 PDF 解析：不同于仅阅读摘要的普通脚本，本项目集成 PyMuPDF 自动提取论文第一页（Introduction）与最后一页（Conclusion），让 AI 触达核心干货。

DeepSeek-V3 驱动：利用大模型超长上下文能力，对候选论文进行全方位质量评估与打分。

“六芒星”结构化报告：提供包含【定性、痛点、创新、对比、场景、建议】六大维度的深度学术解析。

通俗大白话总结：每篇论文附带一段“人话版本”，快速扫一眼即可捕捉核心价值。

全局 Top 5 严选：打破领域壁垒，在所有关注方向中通过 AI 评分进行全局大排名，确保每天只推最精华的 5 篇。

持久化去重机制：自动记录已处理论文 ID，避免重复推送，极大节省 API Token 消耗。

零成本运维：完全基于 GitHub Actions 运行，无需自备服务器，零电费实现 24/7 自动追踪。

🛠️ 环境搭建
1. 克隆仓库与安装依赖
Bash
git clone https://github.com/your-username/arxiv-digest-bot.git
cd arxiv-digest-bot
pip install -r requirements.txt
2. 配置关键词 (config.yaml)
根据你的研究方向修改 config.yaml。本项目默认配置了：

Robotics（机器人学）

Planning and Control（规划与控制）

Decision Making（决策规划）

Reinforcement Learning（强化学习）

Autonomous Driving（自动驾驶）

🚀 云端部署指南 (GitHub Actions)
配置权限：

进入仓库 Settings -> Actions -> General。

在 Workflow permissions 中勾选 Read and write permissions 并保存。

配置 Secrets：

进入 Settings -> Secrets and variables -> Actions。

点击 New repository secret 添加：

DEEPSEEK_API_KEY: 你的 DeepSeek API 密钥。

FEISHU_WEBHOOK: 飞书自定义机器人的 Webhook 地址。

激活定时任务：

默认每天北京时间早上 8:00 运行。

也可以在 Actions 页面手动点击 Run workflow 立即触发。

📊 推送样例展示
🏆 [全场 Top 5 | Autonomous Driving]
Efficient Equivariant Transformer for Self-Driving...

👤 作者: Scott Xu, Dian Chen...
🏷️ 分类: [Equivariant Transformer] [Agent Modeling]
💻 开源代码: https://github.com/xxx
🔥 AI 评分: 92 / 100

🔬 硬核深度解析
🎯 定性: 提出了一种高效的 SE(2) 等变 Transformer 架构 DriveGATr。
💢 痛点: 传统等变网络在处理大规模点云或轨迹预测时，相对位置编码计算开销呈几何级数增长，难以实时。
✨ 创新: 利用几何代数实现等变性，通过线性时间复杂度的算子替代了传统的注意力机制修正。
📊 对比: 在 nuScenes 榜单上相比原生 Waymo 模型推理延迟降低 35%，精度持平。
🚗 场景: 适用于城市复杂路口的长时程多主体轨迹预测。
📌 建议: 强烈建议关注。其轻量化思路非常适合目前咱们的车端感知平台改造。

🗣️ 通俗大白话总结

以前为了让自动驾驶大脑“认路”不管车头朝哪都一样准，得算半天。这帮人想了个数学窍门，不用笨办法算了，不仅快了一倍，而且算得还特别准，代码还开源了！

🔗 原件链接: http://arxiv.org/abs/2604.xxxxx

📂 项目结构
Plaintext
├── .github/workflows/
│   └── daily_run.yml        # GitHub Actions 自动化脚本
├── docs/                    # 存储生成的本地 JSON 与 Markdown 库
├── daily_arxiv.py           # 核心引擎：爬虫、PDF解析、AI评估、推送
├── config.yaml              # 搜索关键词与领域配置文件
├── requirements.txt         # 依赖库清单
└── processed_history.json   # 论文处理历史记录（自动更新）
🤝 致谢
本系统基于 cv-arxiv-daily 进行深度定制开发，感谢原作者的开源贡献。同时感谢 DeepSeek 提供强大的 API 支持。

Happy Researching! 如果这个项目对你有帮助，欢迎点个 ⭐️。