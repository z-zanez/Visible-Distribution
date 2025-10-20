"""GPT-2 模型的分布分析器。"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from training.core.models import ModelAnalyzer


class GPT2Analyzer(ModelAnalyzer):
    """针对GPT-2架构的分析器，实现权重、激活与梯度分布统计。"""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        local_files_only: bool = False,
        use_fast_tokenizer: bool = True,
    ) -> None:
        dtype = torch.float16 if _prefer_half_precision(device) else torch.float32

        config_kwargs = {"local_files_only": local_files_only}
        config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
        if getattr(config, "loss_type", None) is None:
            setattr(config, "loss_type", "ForCausalLM")

        model_kwargs = {
            "config": config,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                **model_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                **model_kwargs,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            use_fast=use_fast_tokenizer,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "pad_token_id", None) is not None and getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if getattr(model, "loss_type", None) is None:
            model.loss_type = "ForCausalLM"
        if getattr(model.config, "loss_type", None) != "ForCausalLM":
            model.config.loss_type = "ForCausalLM"

        super().__init__(model=model, tokenizer=tokenizer, device=device)

        self.num_layers = self._infer_num_layers()
        self.hidden_size = getattr(self.model.config, "n_embd", None) or getattr(self.model.config, "hidden_size", None)
        if self.hidden_size is None:
            raise ValueError("Unable to determine GPT-2 hidden size for QKV slicing")

    def _infer_num_layers(self) -> int:
        if hasattr(self.model.config, "n_layer"):
            return int(self.model.config.n_layer)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)
        raise ValueError("Unable to determine GPT-2 layer count")

    def register_hooks(self, layers: Optional[List[int]] = None) -> None:
        if not hasattr(self.model, "transformer") or not hasattr(self.model.transformer, "h"):
            raise RuntimeError("GPT-2 model structure not recognized")

        blocks = self.model.transformer.h
        if layers is None:
            layers = list(range(len(blocks)))

        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= len(blocks):
                continue
            block = blocks[layer_idx]
            hidden = self.hidden_size

            def _slice_transform(start: int, end: int):
                def transformer(output):
                    if isinstance(output, torch.Tensor):
                        tensor = output.detach()
                        sliced = tensor[..., start:end]
                        if sliced.device.type != "cpu":
                            sliced = sliced.cpu()
                        return sliced
                    return output
                return transformer

            self.activation_store.register_hook(
                block.attn.c_attn,
                f"layer_{layer_idx}_attention_qkv_linear",
                retain_input=True,
                retain_output=False,
            )
            self.activation_store.register_hook(
                block.attn.c_attn,
                f"layer_{layer_idx}_attention_q_output",
                transform=_slice_transform(0, hidden),
            )
            self.activation_store.register_hook(
                block.attn.c_attn,
                f"layer_{layer_idx}_attention_k_output",
                transform=_slice_transform(hidden, hidden * 2),
            )
            self.activation_store.register_hook(
                block.attn.c_attn,
                f"layer_{layer_idx}_attention_v_output",
                transform=_slice_transform(hidden * 2, hidden * 3),
            )
            self.activation_store.register_hook(
                block.attn.c_proj,
                f"layer_{layer_idx}_attention_output_linear",
                retain_input=True,
                retain_output=True,
            )

            self.activation_store.register_hook(
                block.mlp.c_fc,
                f"layer_{layer_idx}_mlp_fc_linear",
                retain_input=True,
                retain_output=True,
            )
            self.activation_store.register_hook(
                block.mlp.c_proj,
                f"layer_{layer_idx}_mlp_proj_linear",
                retain_input=True,
                retain_output=True,
            )

            activation_callable = getattr(block.mlp, "act", None)
            if callable(activation_callable):
                def _activation_transform(output, fn=activation_callable):
                    if isinstance(output, torch.Tensor):
                        tensor = output.detach()
                        activated = fn(tensor)
                        if isinstance(activated, torch.Tensor):
                            if activated.device.type != "cpu":
                                activated = activated.cpu()
                            return activated
                        return activated
                    return output
                self.activation_store.register_hook(
                    block.mlp.c_fc,
                    f"layer_{layer_idx}_mlp_activation_output",
                    transform=_activation_transform,
                )

    def _parameter_in_layers(self, parameter_name: str, layers: List[int]) -> bool:
        for layer_idx in layers:
            if f"transformer.h.{layer_idx}." in parameter_name:
                return True
        return False

    def _should_track_parameter(self, parameter_name: str) -> bool:
        targets = (
            ".attn.c_attn",
            ".attn.c_proj",
            ".mlp.c_fc",
            ".mlp.c_proj",
        )
        return any(target in parameter_name for target in targets)


def _prefer_half_precision(device: Optional[Union[str, torch.device]]) -> bool:
    if device is None:
        return torch.cuda.is_available()
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")
