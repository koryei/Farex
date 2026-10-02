"""Plot latency and accuracy metrics from saved evaluation JSONL logs.

Examples::

    python evaluate/plot_metrics.py
    python evaluate/plot_metrics.py --model llama3:8b
    python evaluate/plot_metrics.py --logs logs/clinical_results.jsonl --output dashboard.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .evaluate import ClinicalEvaluator
except ImportError:  # Support direct invocation: python evaluate/plot_metrics.py
    from evaluate import ClinicalEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluate" / "llama3_performance_dashboard.png"
DOMAIN_LABELS = {
    "symptom_triage": "Symptom Triage",
    "drug_interaction_checker": "Drug Interaction",
    "medical_literature_search": "Medical Literature",
    "general_code_math": "Clinical Math",
}


def _log_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            paths.update(item.glob("*.jsonl"))
        elif item.is_file():
            paths.add(item)
        else:
            print(f"⚠️ Log file not found, skipping: {item}")
    return sorted(paths)


def load_records(inputs: Iterable[Path], *, model_filter: str = "", include_tests: bool = False) -> list[dict[str, Any]]:
    """Load valid JSONL rows and deduplicate cumulative/per-run log copies."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    scorer = ClinicalEvaluator()

    for path in _log_paths(inputs):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping malformed JSON: {path}:{line_number}")
                    continue
                if not isinstance(row, dict) or row.get("test_case") and not include_tests:
                    continue
                if model_filter and model_filter.casefold() not in str(row.get("model", "")).casefold():
                    continue

                task_id = str(row.get("task_id", ""))
                run_id = str(row.get("run_id", ""))
                if run_id and task_id:
                    # History and per-run files contain identical records. Timestamp
                    # preserves intentional repeated task IDs inside a single run.
                    key = (run_id, task_id, str(row.get("timestamp", "")))
                    if key in seen:
                        continue
                    seen.add(key)

                score = row.get("score")
                if score is None and row.get("status", "ok") == "ok":
                    task_type = str(row.get("task_type", ""))
                    if task_type in DOMAIN_LABELS:
                        score = scorer.process_telemetry_run(
                            task_id,
                            task_type,
                            str(row.get("response", "")),
                            str(row.get("expected_answer", row.get("correct_answer", ""))),
                        )
                    else:
                        score = 0
                row["_score"] = int(bool(score))
                try:
                    latency_ms = float(row.get("latency_ms", 0))
                except (TypeError, ValueError):
                    latency_ms = 0.0
                row["_latency_sec"] = max(0.0, latency_ms / 1000)
                records.append(row)
    return records


def generate_telemetry_dashboard(
    log_paths: Iterable[Path],
    output_path: Path = DEFAULT_OUTPUT,
    *,
    model_filter: str = "",
    include_tests: bool = False,
) -> Path:
    """Generate a dual-axis domain dashboard from persisted evaluation results."""
    records = load_records(log_paths, model_filter=model_filter, include_tests=include_tests)
    if not records:
        raise ValueError("No evaluation records found in the selected logs")

    latency_by_domain: dict[str, list[float]] = defaultdict(list)
    score_by_domain: dict[str, list[int]] = defaultdict(list)
    models: set[str] = set()
    for row in records:
        task_type = str(row.get("task_type", "unknown"))
        latency_by_domain[task_type].append(row["_latency_sec"])
        score_by_domain[task_type].append(row["_score"])
        if row.get("model"):
            models.add(str(row["model"]))

    domains = [domain for domain in DOMAIN_LABELS if domain in latency_by_domain]
    domains.extend(sorted(set(latency_by_domain) - set(domains)))
    labels = [DOMAIN_LABELS.get(domain, domain.replace("_", " ").title()) for domain in domains]
    average_latency_sec = [float(np.mean(latency_by_domain[domain])) for domain in domains]
    accuracy_percent = [100 * float(np.mean(score_by_domain[domain])) for domain in domains]

    print(f"📈 Plotting {len(records)} saved evaluation records across {len(domains)} domains...")
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    color = "#1b2333"
    ax1.set_xlabel("Evaluation Domain", fontweight="bold", labelpad=12)
    ax1.set_ylabel("Average Latency (Seconds)", color=color, fontweight="bold")
    ax1.bar(
        labels,
        average_latency_sec,
        color=color,
        alpha=0.18,
        edgecolor="#1b2333",
        width=0.55,
        label="Average latency (s)",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    ax2 = ax1.twinx()
    color_metrics = "#c23a3a"
    ax2.set_ylabel("Accuracy (%)", color=color_metrics, fontweight="bold")
    ax2.plot(
        labels,
        accuracy_percent,
        color=color_metrics,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Accuracy",
    )
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor=color_metrics)
    ax2.set_yticks(np.arange(0, 101, 20))

    model_label = model_filter or (", ".join(sorted(models)) if len(models) <= 3 else "Multiple models")
    title = "Evaluation Telemetry: Latency vs. Domain Accuracy"
    if model_label:
        title = f"{model_label} — {title}"
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Dashboard exported to: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot evaluation latency and accuracy from JSONL logs")
    parser.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "logs"],
        help="JSONL file(s) or directory/directories containing logs (default: logs/)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path")
    parser.add_argument("--model", default="", help="Only include records whose model contains this text")
    parser.add_argument("--include-tests", action="store_true", help="Include saved self-test cases")
    args = parser.parse_args()
    try:
        generate_telemetry_dashboard(
            args.logs,
            args.output,
            model_filter=args.model,
            include_tests=args.include_tests,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
