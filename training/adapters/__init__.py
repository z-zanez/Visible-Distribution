"""模型适配器，提供不同架构的分析器实现。"""

from .gpt2 import GPT2Analyzer
from .llama import LlamaAnalyzer

__all__ = ["GPT2Analyzer", "LlamaAnalyzer"]
