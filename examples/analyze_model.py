"""示例脚本：分析GPT-2或LLaMA模型的层级分布。"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from training.adapters import GPT2Analyzer, LlamaAnalyzer


def _parse_layers(layers: Optional[str]) -> Optional[List[int]]:
    if layers is None or layers.strip() == "":
        return None
    return [int(item) for item in layers.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-layer distributions for LLMs")
    parser.add_argument("--model_type", choices=["gpt2", "llama"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text", default="Once upon a time")
    parser.add_argument("--layers", default=None, help="Comma separated layer indices, e.g. 0,1,2")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_flash_attention", action="store_true", help="Only valid for LLaMA models")
    parser.add_argument("--no_weights", action="store_true")
    parser.add_argument("--no_activations", action="store_true")
    parser.add_argument("--no_gradients", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_heatmaps", action="store_true")
    parser.add_argument("--heatmap-max-dim", type=int, default=256, help="最大热力图尺寸（默认256）")

    args = parser.parse_args()

    layers = _parse_layers(args.layers)

    if args.model_type == "gpt2":
        analyzer = GPT2Analyzer(
            model_name_or_path=args.model_path,
            device=args.device,
            local_files_only=args.local_files_only,
        )
    else:
        analyzer = LlamaAnalyzer(
            model_path=args.model_path,
            device=args.device,
            local_files_only=args.local_files_only,
            use_flash_attention=args.use_flash_attention,
        )

    results = analyzer.analyze_distributions(
        input_text=args.text,
        output_dir=args.output_dir,
        layers=layers,
        bins=args.bins,
        capture_weights=not args.no_weights,
        capture_activations=not args.no_activations,
        capture_gradients=not args.no_gradients,
        plot=not args.no_plots,
        plot_heatmaps=not args.no_heatmaps,
        heatmap_max_dim=args.heatmap_max_dim,
    )

    stats_path = os.path.join(args.output_dir, "layer_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Analysis finished. Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
