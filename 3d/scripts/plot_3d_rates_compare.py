import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


METRICS = [
    "parse_success_rate",
    "valid_rate",
    "novelty_rate",
    "recovery_rate",
]


def load_metrics(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats = data.get("statistics", {})
    result: Dict[str, float] = {}
    for m in METRICS:
        v = stats.get(m)
        if isinstance(v, dict):
            result[m] = float(v.get("mean", 0.0))
        elif isinstance(v, (int, float)):
            result[m] = float(v)
        else:
            result[m] = 0.0
    return result


def plot_grouped_bar(metric_values: Dict[str, Dict[str, float]], out_path: str, title: str = "3D Benchmark Rates Comparison"):
    labels = METRICS
    series_names = list(metric_values.keys())  # e.g., ["instruct_pre", "instruct_post", ...]

    x = range(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, name in enumerate(series_names):
        vals = [metric_values[name].get(m, 0.0) for m in labels]
        # Offset for each series
        offs = [(xi + (i - (len(series_names)-1)/2) * width) for xi in x]
        bars = ax.bar(offs, vals, width, label=name)
        # Annotate values as percentage
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h*100:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("_", "\n") for m in labels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot comparison of *_rate metrics for instruct and thinking (pre vs finetuned)")
    parser.add_argument("--instruct_pre", required=True, help="Path to instruct baseline JSON result")
    parser.add_argument("--instruct_post", required=True, help="Path to instruct finetuned JSON result")
    parser.add_argument("--thinking_pre", required=True, help="Path to thinking baseline JSON result")
    parser.add_argument("--thinking_post", required=True, help="Path to thinking finetuned JSON result")
    parser.add_argument("--out_instruct", required=False, default="/opt/data/private/FYP33_OYXX/_HypoSpace/3d/results/compare_instruct_rates.png", help="Output PNG path for instruct figure")
    parser.add_argument("--out_thinking", required=False, default="/opt/data/private/FYP33_OYXX/_HypoSpace/3d/results/compare_thinking_rates.png", help="Output PNG path for thinking figure")
    args = parser.parse_args()

    # Build two figures with renamed series
    instruct_values: Dict[str, Dict[str, float]] = {
        "instruct": load_metrics(args.instruct_pre),
        "instruct_finetuned": load_metrics(args.instruct_post),
    }
    thinking_values: Dict[str, Dict[str, float]] = {
        "thinking": load_metrics(args.thinking_pre),
        "thinking_finetuned": load_metrics(args.thinking_post),
    }

    plot_grouped_bar(instruct_values, args.out_instruct, title="Instruct: Fine-tune Comparison (Rates)")
    plot_grouped_bar(thinking_values, args.out_thinking, title="Thinking: Fine-tune Comparison (Rates)")

    # Print a small summary table to stdout
    print("Summary (means):")
    for group_name, series in ("instruct", instruct_values), ("thinking", thinking_values):
        print(f"[{group_name}]")
        for name, vals in series.items():
            row = ", ".join(f"{m}={vals[m]:.3f}" for m in METRICS)
            print(f"- {name}: {row}")
    print(f"Saved figures to:\n  {args.out_instruct}\n  {args.out_thinking}")


if __name__ == "__main__":
    main()
