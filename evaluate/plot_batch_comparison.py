"""Plot the two-model clinical batch comparison from results JSONL telemetry.

Produces a four-panel figure: per-domain accuracy, per-domain mean latency,
per-task latency scatter, and the per-task score matrix.

Examples::

    python evaluate/plot_batch_comparison.py
    python evaluate/plot_batch_comparison.py --output results/batch_model_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "results" / "evaluation_run.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "batch_model_comparison.png"

DOMAINS = (
    "symptom_triage",
    "drug_interaction_checker",
    "medical_literature_search",
    "general_code_math",
)
DOMAIN_LABELS = {
    "symptom_triage": "Symptom\nTriage",
    "drug_interaction_checker": "Drug\nInteraction",
    "medical_literature_search": "Medical\nLiterature",
    "general_code_math": "Clinical\nMath / Code",
}
MODELS = ("llama3:8b", "mistral:7b")
MODEL_COLORS = {"llama3:8b": "#0072B2", "mistral:7b": "#E69F00"}  # Okabe-Ito
PASS_COLOR = "#2e7d32"
FAIL_COLOR = "#c62828"


def load_real_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load scored, non-dry-run, non-test records from JSONL log files."""
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping malformed JSON: {path}:{line_number}")
                    continue
                if row.get("dry_run") or row.get("test_case"):
                    continue
                if row.get("status", "ok") != "ok":
                    continue
                records.append(row)
    return records


def _model_series(records: Sequence[Mapping[str, Any]], model: str) -> list[Mapping[str, Any]]:
    return [row for row in records if str(row.get("model")) == model]


def plot_comparison(records: Sequence[Mapping[str, Any]], output_path: Path) -> Path:
    present_models = [model for model in MODELS if _model_series(records, model)]
    if not present_models:
        raise ValueError("No scored records found for the expected models")

    # Preserve the task-file order from the first model's run for the matrix rows.
    task_order: list[str] = []
    for row in records:
        task_id = str(row.get("task_id", ""))
        if task_id and task_id not in task_order:
            task_order.append(task_id)

    scores = {
        (str(row.get("model")), str(row.get("task_id"))): int(bool(row.get("score", 0)))
        for row in records
    }
    latencies = {
        (str(row.get("model")), str(row.get("task_id"))): float(row.get("latency_seconds", 0) or 0)
        for row in records
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    x = np.arange(len(DOMAINS))
    width = 0.36

    # (a) Accuracy by domain.
    ax = axes[0][0]
    for offset, model in enumerate(present_models):
        rows = _model_series(records, model)
        accuracy = [
            100 * float(np.mean([int(bool(r.get("score", 0))) for r in rows if r.get("task_type") == domain]))
            for domain in DOMAINS
        ]
        bars = ax.bar(
            x + (offset - 0.5) * width,
            accuracy,
            width,
            color=MODEL_COLORS[model],
            label=model,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8.5, padding=2)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("(a) Accuracy by domain", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], fontsize=9)

    # (b) Mean latency by domain.
    ax = axes[0][1]
    for offset, model in enumerate(present_models):
        rows = _model_series(records, model)
        mean_latency = [
            float(np.mean([float(r.get("latency_seconds", 0) or 0) for r in rows if r.get("task_type") == domain]))
            for domain in DOMAINS
        ]
        bars = ax.bar(
            x + (offset - 0.5) * width,
            mean_latency,
            width,
            color=MODEL_COLORS[model],
            label=model,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar_label(bars, fmt="%.1fs", fontsize=8.5, padding=2)
    ax.set_ylabel("Mean latency (s)", fontweight="bold")
    ax.set_title("(b) Mean latency by domain", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], fontsize=9)

    # (c) Per-task latency scatter.
    ax = axes[1][0]
    rng = np.random.default_rng(7)
    for model in present_models:
        for index, domain in enumerate(DOMAINS):
            points = [
                float(r.get("latency_seconds", 0) or 0)
                for r in _model_series(records, model)
                if r.get("task_type") == domain
            ]
            jitter = rng.uniform(-0.16, 0.16, size=len(points))
            ax.scatter(index + jitter, points, s=34, color=MODEL_COLORS[model], alpha=0.85, label=model, edgecolors="white", linewidths=0.5)
    ax.set_ylabel("Latency per task (s)", fontweight="bold")
    ax.set_title("(c) Per-task latency distribution", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # (d) Per-task score matrix.
    ax = axes[1][1]
    matrix = np.array([[scores.get((model, task_id), 0) for model in present_models] for task_id in task_order])
    ax.imshow(matrix, cmap=ListedColormap([FAIL_COLOR, PASS_COLOR]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(present_models)))
    ax.set_xticklabels(present_models, fontsize=9.5)
    ax.set_yticks(range(len(task_order)))
    ax.set_yticklabels(task_order, fontsize=7.5)
    ax.set_title("(d) Per-task score matrix (green = pass)", fontweight="bold", fontsize=11)
    # Domain block separators + right-side domain labels.
    boundary = 0
    for domain in DOMAINS:
        count = sum(1 for task_id in task_order if task_id.startswith(domain))
        boundary += count
        if boundary < len(task_order):
            ax.axhline(boundary - 0.5, color="white", linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in present_models]
    fig.legend(handles, present_models, loc="upper right", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.99, 0.975))
    fig.suptitle("Two-Model Clinical Routing Benchmark — 16 tasks × 2 local models (Ollama, temperature 0.0)", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.01, 0.007, "Source: results/evaluation_run.jsonl · runs 20261003T031440 (llama3:8b) & 20261003T031555 (mistral:7b) · scoring via ClinicalEvaluator", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.015, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the two-model clinical batch comparison figure")
    parser.add_argument("--logs", nargs="+", type=Path, default=[DEFAULT_LOG], help="JSONL telemetry file(s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path")
    args = parser.parse_args(argv)

    missing = [path for path in args.logs if not path.is_file()]
    if missing:
        parser.error(f"log file(s) not found: {', '.join(map(str, missing))}")

    records = load_real_records(args.logs)
    if not records:
        parser.error("no scored (non-dry-run) records found in the selected logs")

    output = plot_comparison(records, args.output.expanduser().resolve())
    print(f"✅ Comparison figure written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
