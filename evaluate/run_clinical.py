"""Run clinical benchmark tasks using provider settings from ``.env``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__:
    from .clinical_agents import CLINICAL_DESTINATIONS
    from .clinical_benchmark import (
        ClinicalRunConfig,
        iter_benchmark,
        redact,
        summarize_scores,
        validate_setup,
    )
else:  # Support direct invocation: python evaluate/run_clinical.py
    from clinical_agents import CLINICAL_DESTINATIONS
    from clinical_benchmark import (
        ClinicalRunConfig,
        iter_benchmark,
        redact,
        summarize_scores,
        validate_setup,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run clinical tasks with the provider, model, endpoints, and credentials "
            "configured in the project .env (command-line options override settings)."
        )
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        help="Clinical JSON task file (top-level 'tasks' or 'clinical_tasks' list)",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Run only this task; repeat the option to select multiple tasks",
    )
    parser.add_argument(
        "--limit", type=int, help="Run at most this many tasks after task id selection"
    )
    parser.add_argument(
        "--agent",
        choices=sorted(CLINICAL_DESTINATIONS),
        help="Send every selected task to this clinical agent (routing override)",
    )
    parser.add_argument("--provider", help="ollama, openrouter, or requesty")
    parser.add_argument("--model", help="Model override; Ollama defaults to llama3:8b")
    parser.add_argument("--log", type=Path, help="Append per-task JSONL results here")
    parser.add_argument("--ollama-url")
    parser.add_argument("--openrouter-url")
    parser.add_argument("--requesty-url")
    parser.add_argument("--api-key-env", help="Credential environment variable name")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check settings, task data, and prompt files without calling a model",
    )
    return parser


def _render_result(index: int, total: int, result: dict) -> None:
    print(f"[{index}/{total}] Processing {result['task_id']} ({result['agent_type']})...")
    print(f"   Latency: {result['latency_ms']:.2f} ms")
    if result["status"] == "ok":
        print(f"   Response: {result['response']}\n")
    else:
        print(f"   Error: {result['error']}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = ClinicalRunConfig.from_environment(
            tasks_path=args.tasks,
            log_path=args.log,
            provider=args.provider,
            model=args.model,
            ollama_url=args.ollama_url,
            openrouter_url=args.openrouter_url,
            requesty_url=args.requesty_url,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            timeout=args.timeout,
            task_ids=args.task_ids,
            agent_override=args.agent,
            limit=args.limit,
        )
        if args.validate_only:
            _, counts = validate_setup(config)
            print(
                f"Configuration OK: provider={config.provider}, model={redact(config.model)}, "
                f"temperature={config.temperature}, timeout={config.timeout}s"
            )
            print(f"Tasks OK: {sum(counts.values())} loaded; destinations={counts}")
            print("Clinical prompt files: OK; no model request sent")
            return 0

        print(
            f"Clinical benchmark | provider={config.provider} "
            f"| model={redact(config.model)}"
        )
        results = []
        for index, total, result in iter_benchmark(config):
            results.append(result)
            _render_result(index, total, result)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Clinical benchmark setup failed: {redact(str(exc))}", file=sys.stderr)
        return 2

    failed = sum(result["status"] == "error" for result in results)
    scores = summarize_scores(results)
    run_log = results[0].get("run_log") if results else "(no tasks selected)"
    run_id = results[0].get("run_id") if results else "(no tasks selected)"
    print(
        f"Benchmark complete: {len(results) - failed}/{len(results)} succeeded, "
        f"{failed} failed.\n"
        f"   Run ID: {run_id}\n"
        f"   Run log: {run_log}\n"
        f"   Cumulative history: {config.log_path}"
    )
    print(
        f"📊 ClinicalEvaluator: {scores['passed']}/{scores['evaluated']} tasks passed "
        f"(accuracy: {scores['accuracy']:.1%})"
    )
    if results:
        latencies = [result["latency_ms"] for result in results]
        print(
            f"Latency: avg={sum(latencies) / len(latencies):.2f} ms, "
            f"min={min(latencies):.2f} ms, max={max(latencies):.2f} ms"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
