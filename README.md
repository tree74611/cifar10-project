# CIFAR-10 Image Classification (Course Project)

说明:
- 数据集: CIFAR-10 (UCI 版本 / 原始)
  - UCI 页面: https://archive.ics.uci.edu/dataset/559/cifar-10
  - 官方页面: https://www.cs.toronto.edu/~kriz/cifar.html
- 框架: PyTorch（推荐）
- 快速开始:
  1. 克隆仓库: git clone <repo_url>
  2. 创建虚拟环境并安装依赖: pip install -r requirements.txt
  3. 训练 baseline: python src/train.py --model simple_cnn --epochs 20 --batch-size 128
  4. 评估: python src/eval.py --checkpoint checkpoints/best.pth
- 团队分工（示例）: 项目管理 / 数据处理 / 模型实现 / 实验记录 / 报告与部署
- 注意: 不要把数据集的大文件直接推到 GitHub（在 README 中给出下载链接或写下载脚本）。
