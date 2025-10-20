"""分布绘图工具。"""
from __future__ import annotations

import math
import os
from typing import Dict

import matplotlib.pyplot as plt
import torch

from training.core.utils.statistics import summarize_tensor, tensor_distribution_summary


def _to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach().float()
    if data.device.type != "cpu":
        data = data.cpu()
    return data


def plot_tensor_distribution(
    tensor: torch.Tensor,
    title: str,
    output_path: str,
    bins: int = 60,
) -> Dict[str, float]:
    """绘制并保存单个张量的直方图。"""
    summary = tensor_distribution_summary(tensor, bins=bins)
    stats = summary["summary"]
    histogram = summary["histogram"]

    edges = torch.tensor(histogram["bin_edges"])
    density = torch.tensor(histogram["density"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]).item() if len(edges) > 1 else 1.0
    if width == 0:
        width = 1.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.bar(centers.numpy(), density.numpy(), width=width, alpha=0.8)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")

    text = (
        f"mean={stats['mean']:.4f}\n"
        f"std={stats['std']:.4f}\n"
        f"min={stats['min']:.4f}, max={stats['max']:.4f}"
    )
    plt.text(
        0.98,
        0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return stats


def _prepare_heatmap_tensor(tensor: torch.Tensor, max_dim: int = 256) -> torch.Tensor:
    data = _to_cpu_tensor(tensor)

    if data.ndim == 0:
        data = data.view(1, 1)
    elif data.ndim == 1:
        length = data.numel()
        side = int(math.ceil(math.sqrt(length)))
        padded = torch.zeros(side * side, dtype=data.dtype)
        padded[:length] = data
        data = padded.view(side, side)
    else:
        last_dim = data.shape[-1]
        if last_dim == 0:
            return torch.zeros((1, 1))
        data = data.reshape(-1, last_dim)

    if data.shape[0] > max_dim:
        step0 = max(1, math.ceil(data.shape[0] / max_dim))
        data = data[::step0]
    if data.shape[1] > max_dim:
        step1 = max(1, math.ceil(data.shape[1] / max_dim))
        data = data[:, ::step1]

    return data


def plot_tensor_heatmap(
    tensor: torch.Tensor,
    title: str,
    output_path: str,
    max_dim: int = 256,
) -> None:
    """绘制矩阵形式的热力图，颜色深浅表示数值大小。"""
    matrix = _prepare_heatmap_tensor(tensor, max_dim=max_dim)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(matrix.numpy(), aspect="auto", cmap="viridis")
    plt.title(title)
    plt.xlabel("Feature Index")
    plt.ylabel("Row Index")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_distributions(
    tensors: Dict[str, torch.Tensor],
    output_dir: str,
    prefix: str,
    bins: int = 60,
    include_hist: bool = True,
    include_heatmap: bool = False,
    heatmap_max_dim: int = 256,
) -> Dict[str, Dict[str, str]]:
    """批量绘制张量分布图与热力图。

    Returns:
        字典：键为张量名称，值为包含生成图片路径的字典。
    """
    figure_paths: Dict[str, Dict[str, str]] = {}
    if not tensors:
        return figure_paths

    os.makedirs(output_dir, exist_ok=True)

    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        safe_name = name.replace("/", "_").replace(".", "_")
        figures: Dict[str, str] = {}

        if include_hist:
            figure_path = os.path.join(output_dir, f"{prefix}_{safe_name}.png")
            plot_tensor_distribution(
                tensor,
                title=f"{prefix} | {name}",
                output_path=figure_path,
                bins=bins,
            )
            figures["histogram"] = figure_path

        if include_heatmap:
            heatmap_path = os.path.join(output_dir, f"{prefix}_{safe_name}_heatmap.png")
            plot_tensor_heatmap(
                tensor,
                title=f"{prefix} Heatmap | {name}",
                output_path=heatmap_path,
                max_dim=heatmap_max_dim,
            )
            figures["heatmap"] = heatmap_path

        if figures:
            figure_paths[name] = figures

    return figure_paths
