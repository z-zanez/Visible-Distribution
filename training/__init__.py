"""Training：面向LLM权重、激活与梯度分布分析的工具包。"""

__version__ = "0.1.0"

from . import adapters, core, viz

__all__ = ["adapters", "core", "viz", "__version__"]
