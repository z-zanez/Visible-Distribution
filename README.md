# Training：LLM分布分析工具

Training 仓库聚焦于大型语言模型训练过程中的关键统计：为每一层生成权重、激活值与参数梯度的分布图，并输出对应的统计量。

## ✨ 功能亮点
- **双模型支持**：内置 GPT-2 与 LLaMA 分析器。
- **分布统计一站式产出**：一次前向/反向即可同时获得权重、激活、梯度的直方图与统计指标。
- **灵活的层选择**：可通过参数筛选感兴趣的层，避免对全量模型进行昂贵分析。
- **快速可视化**：为每个张量生成直方图与热力图（`*_heatmap.png`），图像保存在 `weights/`、`activations/`、`gradients/` 子目录。

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/z-zanez/Visible-Distribution.git
cd Visible-Distribution

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

运行示例脚本：

```bash
python examples/analyze_model.py \
  --model_type gpt2 \
  --model_path ./models/gpt2 \
  --output_dir ./outputs/gpt2-layer-stats \
  --layers 0,1,2 \
  --text "gpt2 test"
```

对于 LLaMA（请确保模型文件已本地可用）：

```bash
python examples/analyze_model.py \
  --model_type llama \
  --model_path ./models/Llama-3.2-1b \
  --output_dir ./outputs/llama-layer-stats \
  --layers 0,1 \
  --local_files_only \
  --text "llama test" \
  --no_plots  # 若只想保存JSON统计，可禁用图片
```

生成的目录结构示例：

> 提示：默认会生成直方图与热力图；若需要快速调试可加上 `--no_heatmaps`、`--no_gradients`、`--no_plots`，或调小 `--heatmap-max-dim` 减少输出体积。

```
outputs/
├── layer_statistics.json    # 权重/激活/梯度的统计量
├── activations/             # 每层激活分布图
├── gradients/               # 每层梯度分布图
└── weights/                 # 每层权重分布图
```

## 🧠 代码结构

```
training/
├── core/                    # 基础抽象与统计工具
│   ├── models/base.py       # 通用分析逻辑
│   └── utils/               # Hook 与统计函数
├── adapters/                # 架构特定实现（GPT-2 / LLaMA）
└── viz/                     # 分布绘图工具
```

## 📦 依赖
- Python 3.9+
- PyTorch 2.0+
- transformers 4.35+
- matplotlib、seaborn、numpy

## 🔍 常见问题
- **Q：可以在线下载模型吗？** 默认开启 `--local_files_only`，如需联网请移除此参数。
- **Q：梯度统计为空？** 请确认未使用 `--no_gradients`，并且模型允许反向传播（`use_cache=False` 已自动处理）。
- **Q：想自定义绘图？** 可直接使用 `training.viz.distribution.plot_tensor_distribution` 函数对任意张量绘图。

