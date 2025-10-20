"""统计与分布辅助函数。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch


def _flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """将张量展平到一维，并确保位于CPU上。

    Args:
        tensor: 任意形状的Torch张量

    Returns:
        展平后的float32张量
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor for statistics computation")

    if tensor.numel() == 0:
        raise ValueError("Cannot compute statistics for empty tensor")

    flat = tensor.detach().float().view(-1)
    if flat.device.type != "cpu":
        flat = flat.cpu()
    return flat


def summarize_tensor(tensor: torch.Tensor) -> Dict[str, float]:
    """计算张量的基础统计量。

    Args:
        tensor: 待分析张量

    Returns:
        包含均值、标准差等统计量的字典
    """
    flat = _flatten_tensor(tensor)

    mean = flat.mean()
    std = flat.std(unbiased=False)
    var = flat.var(unbiased=False)
    centered = flat - mean

    # 避免除零
    std_eps = torch.where(std == 0, torch.tensor(1.0, device=std.device), std)
    var_eps = torch.where(var == 0, torch.tensor(1.0, device=var.device), var)

    stats = {
        "mean": mean.item(),
        "std": std.item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "p05": torch.quantile(flat, 0.05).item(),
        "p25": torch.quantile(flat, 0.25).item(),
        "median": torch.quantile(flat, 0.5).item(),
        "p75": torch.quantile(flat, 0.75).item(),
        "p95": torch.quantile(flat, 0.95).item(),
        "l1": flat.abs().mean().item(),
        "l2": torch.sqrt((flat ** 2).mean()).item(),
        "sparsity": (flat == 0).float().mean().item(),
        "skewness": (torch.mean((centered ** 3)) / (std_eps ** 3)).item(),
        "kurtosis": (torch.mean((centered ** 4)) / (var_eps ** 2) - 3).item(),
    }
    return stats


def histogram_tensor(
    tensor: torch.Tensor,
    bins: int = 60,
    value_range: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """生成张量的直方图数据。

    Args:
        tensor: 待分析张量
        bins: 直方图分桶数
        value_range: 可选的取值范围 (min, max)

    Returns:
        包含分桶边界与计数的字典
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")

    flat = _flatten_tensor(tensor)

    if value_range is None:
        min_val = flat.min().item()
        max_val = flat.max().item()
    else:
        value_iter = list(value_range)
        if len(value_iter) != 2:
            raise ValueError("value_range must contain exactly two values")
        min_val, max_val = float(value_iter[0]), float(value_iter[1])

    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5

    edges = torch.linspace(min_val, max_val, bins + 1)
    hist = torch.histc(flat, bins=bins, min=min_val, max=max_val)

    density = hist / (hist.sum() + 1e-12)

    return {
        "bin_edges": edges.tolist(),
        "counts": hist.tolist(),
        "density": density.tolist(),
    }


def tensor_distribution_summary(
    tensor: torch.Tensor,
    bins: int = 60,
    value_range: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """组合统计量与直方图信息。

    Args:
        tensor: 待分析张量
        bins: 直方图分桶数
        value_range: 取值范围

    Returns:
        包含summary与histogram键的字典
    """
    return {
        "summary": summarize_tensor(tensor),
        "histogram": histogram_tensor(tensor, bins=bins, value_range=value_range),
    }
