"""基础模型分析器，为特定架构提供统一接口。"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel

from training.core.utils import ActivationStore, tensor_distribution_summary


_LAYER_PATTERN = re.compile(r"layer_(?P<index>\d+)")


class ModelAnalyzer(ABC):
    """所有模型分析器的抽象基类。"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.activation_store = ActivationStore()

    # ------------------------ Hook管理 ------------------------
    @abstractmethod
    def register_hooks(self, layers: Optional[List[int]] = None) -> None:
        """在指定层上挂载Hook以捕获激活。"""

    def remove_hooks(self) -> None:
        """移除所有Hook。"""
        self.activation_store.remove_hooks()

    def clear_activations(self) -> None:
        self.activation_store.clear()

    # ------------------------ 层匹配辅助 ------------------------
    @abstractmethod
    def _parameter_in_layers(self, parameter_name: str, layers: List[int]) -> bool:
        """判断参数是否属于指定层。"""

    def _should_track_parameter(self, parameter_name: str) -> bool:
        """是否跟踪指定参数的统计。"""
        return True

    def _iter_parameters(self, layers: Optional[List[int]] = None):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if layers is not None and not self._parameter_in_layers(name, layers):
                continue
            if not self._should_track_parameter(name):
                continue
            yield name, param

    def _iter_activations(self, layers: Optional[List[int]] = None):
        for name, activation in self.activation_store.get_all_activations().items():
            if not isinstance(activation, torch.Tensor):
                continue
            if layers is not None and not self._activation_in_layers(name, layers):
                continue
            yield name, activation

    # ------------------------ 统计信息 ------------------------
    def get_weight_distributions(
        self,
        layers: Optional[List[int]] = None,
        bins: int = 60,
    ) -> Dict[str, Dict[str, Any]]:
        """返回权重的分布摘要。"""
        stats: Dict[str, Dict[str, Any]] = {}
        for name, param in self._iter_parameters(layers):
            if param.numel() == 0:
                continue
            try:
                stats[name] = tensor_distribution_summary(param.data, bins=bins)
            except (ValueError, RuntimeError):
                continue
        return stats

    def get_gradient_distributions(
        self,
        layers: Optional[List[int]] = None,
        bins: int = 60,
    ) -> Dict[str, Dict[str, Any]]:
        """返回梯度分布摘要（需提前调用 backward）。"""
        stats: Dict[str, Dict[str, Any]] = {}
        for name, param in self._iter_parameters(layers):
            grad = getattr(param, "grad", None)
            if grad is None or grad.numel() == 0:
                continue
            try:
                stats[name] = tensor_distribution_summary(grad, bins=bins)
            except (ValueError, RuntimeError):
                continue
        return stats

    def get_activation_distributions(
        self,
        layers: Optional[List[int]] = None,
        bins: int = 60,
    ) -> Dict[str, Dict[str, Any]]:
        """返回捕获的激活分布摘要。"""
        stats: Dict[str, Dict[str, Any]] = {}
        for name, activation in self._iter_activations(layers):
            if activation.numel() == 0:
                continue
            try:
                stats[name] = tensor_distribution_summary(activation, bins=bins)
            except (ValueError, RuntimeError):
                continue
        return stats

    @staticmethod
    def _activation_in_layers(name: str, layers: List[int]) -> bool:
        match = _LAYER_PATTERN.match(name)
        if match is None:
            return False
        layer_idx = int(match.group("index"))
        return layer_idx in layers

    # ------------------------ 运行辅助 ------------------------
    def tokenize(self, text: Any) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized for this analyzer")
        if text is None:
            raise ValueError("Input text for tokenization cannot be None")
        if isinstance(text, (list, tuple)):
            text = " ".join(map(str, text))
        elif not isinstance(text, str):
            text = str(text)
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded.get("input_ids")
        if input_ids is not None and input_ids.shape[-1] < 2:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is None:
                eos_id = getattr(self.tokenizer, "pad_token_id", None)
            if eos_id is not None:
                pad = torch.full((input_ids.shape[0], 1), eos_id, dtype=input_ids.dtype)
                encoded["input_ids"] = torch.cat([input_ids, pad], dim=-1)
                if "attention_mask" in encoded:
                    encoded["attention_mask"] = torch.cat(
                        [encoded["attention_mask"], torch.ones_like(pad)], dim=-1
                    )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def zero_grad(self) -> None:
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None

    def forward_with_labels(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行自回归损失计算，返回loss。"""
        labels = inputs["input_ids"].clone()
        outputs = self.model(**inputs, labels=labels)
        if not hasattr(outputs, "loss") or outputs.loss is None:
            raise RuntimeError("Model forward pass did not return a loss value")
        return outputs.loss

    def analyze_distributions(
        self,
        input_text: str,
        output_dir: str,
        layers: Optional[List[int]] = None,
        bins: int = 60,
        capture_weights: bool = True,
        capture_activations: bool = True,
        capture_gradients: bool = True,
        plot: bool = True,
        plot_heatmaps: Optional[bool] = None,
        heatmap_max_dim: int = 256,
    ) -> Dict[str, Any]:
        """对指定文本执行一次前向/反向并统计分布信息。"""
        os.makedirs(output_dir, exist_ok=True)

        self.clear_activations()
        hooks_active = False
        if capture_activations:
            self.remove_hooks()
            self.register_hooks(layers)
            hooks_active = True

        inputs = self.tokenize(input_text)
        self.zero_grad()

        try:
            loss = self.forward_with_labels(inputs)
            if capture_gradients:
                loss.backward()
        finally:
            if hooks_active:
                self.remove_hooks()

        results: Dict[str, Any] = {"loss": loss.item()}
        figures: Dict[str, Dict[str, Dict[str, str]]] = {}

        plot_hist = plot
        plot_heat = plot_heatmaps if plot_heatmaps is not None else plot_hist
        plot_module_loaded = False

        if capture_weights:
            weights = {name: param.data.detach() for name, param in self._iter_parameters(layers)}
            results["weights"] = {
                name: tensor_distribution_summary(param, bins=bins)
                for name, param in weights.items()
            }
            if (plot_hist or plot_heat) and weights:
                if not plot_module_loaded:
                    from training.viz.distribution import plot_distributions
                    plot_module_loaded = True
                figure_paths = plot_distributions(
                    weights,
                    os.path.join(output_dir, "weights"),
                    "weight",
                    bins=bins,
                    include_hist=plot_hist,
                    include_heatmap=plot_heat,
                    heatmap_max_dim=heatmap_max_dim,
                )
                if figure_paths:
                    figures["weights"] = figure_paths

        if capture_activations:
            activations = {name: act.detach() for name, act in self._iter_activations(layers)}
            results["activations"] = {
                name: tensor_distribution_summary(act, bins=bins)
                for name, act in activations.items()
            }
            if (plot_hist or plot_heat) and activations:
                if not plot_module_loaded:
                    from training.viz.distribution import plot_distributions
                    plot_module_loaded = True
                figure_paths = plot_distributions(
                    activations,
                    os.path.join(output_dir, "activations"),
                    "activation",
                    bins=bins,
                    include_hist=plot_hist,
                    include_heatmap=plot_heat,
                    heatmap_max_dim=heatmap_max_dim,
                )
                if figure_paths:
                    figures["activations"] = figure_paths

        if capture_gradients:
            gradients = {
                name: param.grad.detach()
                for name, param in self._iter_parameters(layers)
                if getattr(param, "grad", None) is not None
            }
            results["gradients"] = {
                name: tensor_distribution_summary(grad, bins=bins)
                for name, grad in gradients.items()
            }
            if (plot_hist or plot_heat) and gradients:
                if not plot_module_loaded:
                    from training.viz.distribution import plot_distributions
                    plot_module_loaded = True
                figure_paths = plot_distributions(
                    gradients,
                    os.path.join(output_dir, "gradients"),
                    "gradient",
                    bins=bins,
                    include_hist=plot_hist,
                    include_heatmap=plot_heat,
                    heatmap_max_dim=heatmap_max_dim,
                )
                if figure_paths:
                    figures["gradients"] = figure_paths

        if figures:
            results["figures"] = figures

        return results
