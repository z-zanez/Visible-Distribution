"""LLaMA 架构的分布分析器。"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from training.core.models import ModelAnalyzer


class LlamaAnalyzer(ModelAnalyzer):
    """针对LLaMA系列模型的分析器。"""

    def __init__(
        self,
        model_path: str,
        device: Optional[Union[str, torch.device]] = None,
        local_files_only: bool = True,
        use_flash_attention: bool = False,
    ) -> None:
        dtype = torch.float16 if _prefer_half_precision(device) else torch.float32

        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                attn_impl = "eager"

        config_kwargs = {
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        config = AutoConfig.from_pretrained(model_path, **config_kwargs)
        if getattr(config, "loss_type", None) is None:
            setattr(config, "loss_type", "ForCausalLM")

        model_kwargs = {
            "config": config,
            "attn_implementation": attn_impl,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                **model_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                **model_kwargs,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            padding_side="left",
            trust_remote_code=True,
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
        self.hidden_size = getattr(self.model.config, "hidden_size", None)

    def _infer_num_layers(self) -> int:
        if hasattr(self.model.config, "num_hidden_layers"):
            return int(self.model.config.num_hidden_layers)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        raise ValueError("Unable to determine LLaMA layer count")

    def register_hooks(self, layers: Optional[List[int]] = None) -> None:
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise RuntimeError("LLaMA model structure not recognized")

        blocks = self.model.model.layers
        if layers is None:
            layers = list(range(len(blocks)))

        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= len(blocks):
                continue
            layer = blocks[layer_idx]

            self.activation_store.register_hook(
                layer.self_attn.q_proj,
                f"layer_{layer_idx}_attention_q_linear",
                retain_input=True,
                retain_output=True,
            )
            self.activation_store.register_hook(
                layer.self_attn.k_proj,
                f"layer_{layer_idx}_attention_k_output",
                retain_input=False,
                retain_output=True,
            )
            self.activation_store.register_hook(
                layer.self_attn.v_proj,
                f"layer_{layer_idx}_attention_v_output",
                retain_input=False,
                retain_output=True,
            )
            self.activation_store.register_hook(
                layer.self_attn.o_proj,
                f"layer_{layer_idx}_attention_output_linear",
                retain_input=True,
                retain_output=True,
            )

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                if hasattr(mlp, "gate_proj"):
                    self.activation_store.register_hook(
                        mlp.gate_proj,
                        f"layer_{layer_idx}_mlp_gate_linear",
                        retain_input=True,
                        retain_output=True,
                    )
                if hasattr(mlp, "up_proj"):
                    self.activation_store.register_hook(
                        mlp.up_proj,
                        f"layer_{layer_idx}_mlp_up_linear",
                        retain_input=False,
                        retain_output=True,
                    )
                if hasattr(mlp, "down_proj"):
                    self.activation_store.register_hook(
                        mlp.down_proj,
                        f"layer_{layer_idx}_mlp_down_linear",
                        retain_input=True,
                        retain_output=True,
                    )
                activation_module = getattr(mlp, "act_fn", None)
                if isinstance(activation_module, nn.Module):
                    self.activation_store.register_hook(
                        activation_module,
                        f"layer_{layer_idx}_mlp_activation_output",
                        retain_input=True,
                        retain_output=True,
                    )

    def _parameter_in_layers(self, parameter_name: str, layers: List[int]) -> bool:
        for layer_idx in layers:
            if f"model.layers.{layer_idx}." in parameter_name:
                return True
        return False

    def _should_track_parameter(self, parameter_name: str) -> bool:
        attn_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
        if any(f".self_attn.{suffix}" in parameter_name for suffix in attn_suffixes):
            return True
        mlp_suffixes = ("gate_proj", "up_proj", "down_proj")
        return any(f".mlp.{suffix}" in parameter_name for suffix in mlp_suffixes)


def _prefer_half_precision(device: Optional[Union[str, torch.device]]) -> bool:
    if device is None:
        return torch.cuda.is_available()
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")
