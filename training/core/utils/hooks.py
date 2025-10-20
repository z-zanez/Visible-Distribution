"""管理前向与反向Hook的辅助类。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set
import weakref

import torch
from torch import nn


class ActivationStore:
    """用于缓存模型的前向/反向激活张量。"""

    def __init__(self) -> None:
        self._activations: Dict[str, Any] = {}
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._modules: Set[weakref.ReferenceType[nn.Module]] = set()

    def register_hook(
        self,
        module: nn.Module,
        name: str,
        hook_type: str = "forward",
        retain_input: bool = False,
        retain_output: bool = True,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """在指定模块上注册Hook。

        Args:
            module: 需要插入Hook的模块
            name: 保存激活时使用的名称
            hook_type: "forward" 或 "backward"
            retain_input: 是否保存输入
            retain_output: 是否保存输出
            transform: 可选的预处理函数（如detach+cpu）
        """
        if name in self._hooks:
            raise ValueError(f"Hook name '{name}' is already registered")

        self._modules.add(weakref.ref(module))

        def _process(value: Any) -> Any:
            if transform is not None:
                return transform(value)
            if isinstance(value, torch.Tensor):
                tensor = value.detach()
                if tensor.device.type != "cpu":
                    tensor = tensor.cpu()
                return tensor
            if isinstance(value, tuple):
                return tuple(_process(v) for v in value)
            return value

        if hook_type == "forward":
            def hook(mod: nn.Module, inputs: tuple, outputs: Any) -> None:
                if retain_input:
                    self._activations[f"{name}_input"] = _process(inputs[0] if len(inputs) == 1 else inputs)
                if retain_output:
                    self._activations[name] = _process(outputs)

            handle = module.register_forward_hook(hook)
        elif hook_type == "backward":
            def hook(mod: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
                if retain_input and grad_input is not None:
                    processed = tuple(_process(g) for g in grad_input)
                    self._activations[f"{name}_grad_input"] = processed[0] if len(processed) == 1 else processed
                if retain_output and grad_output is not None:
                    processed = tuple(_process(g) for g in grad_output)
                    self._activations[name] = processed[0] if len(processed) == 1 else processed

            handle = module.register_full_backward_hook(hook)
        else:
            raise ValueError(f"Unsupported hook type: {hook_type}")

        self._hooks[name] = handle

    def get_activation(self, name: str) -> Any:
        if name not in self._activations:
            raise KeyError(f"Activation '{name}' not found")
        return self._activations[name]

    def get_all_activations(self) -> Dict[str, Any]:
        return dict(self._activations)

    def clear(self) -> None:
        self._activations.clear()

    def remove_hooks(self, names: Optional[List[str]] = None) -> None:
        targets = names or list(self._hooks.keys())
        for name in targets:
            handle = self._hooks.pop(name, None)
            if handle is not None:
                handle.remove()

    def __del__(self) -> None:
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()
        self._activations.clear()
