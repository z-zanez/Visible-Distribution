"""核心工具模块导出。"""

from .hooks import ActivationStore
from .statistics import (
    histogram_tensor,
    summarize_tensor,
    tensor_distribution_summary,
)

__all__ = [
    "ActivationStore",
    "histogram_tensor",
    "summarize_tensor",
    "tensor_distribution_summary",
]
