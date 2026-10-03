"""Run the 16-query clinical matrix against one or more local Ollama models.

Models are processed strictly sequentially: every task for one model completes
before the next model is touched, so only one request is ever in flight and
local memory pressure stays bounded even with large models loaded.

Normal run::

    python evaluate/batch_eval.py --models llama3:8b mistral:7b

Safe setup/logging run without any model calls::

    python evaluate/batch_eval.py --dry-run

Rows are flushed and fsynced to ``results/evaluation_run.jsonl`` after every
query so completed work remains available if a later model or task fails.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

if __package__:
    from . import evaluate as evaluator
    from .clinical_agents import CLINICAL_DESTINATIONS, load_clinical_tasks
else:  # Support direct invocation: python evaluate/batch_eval.py
    import evaluate as evaluator
    from clinical_agents import CLINICAL_DESTINATIONS, load_clinical_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_PATH = PROJECT_ROOT / "tasks" / "clinical_tasks.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "results" / "evaluation_run.jsonl"
DEFAULT_MODELS = ("llama3:8b", "mistral:7b")
OLLAMA_CHARACTERS_PER_TOKEN = 4  # Approximation only; Ollama transport exposes no usage data.
BASELINE_INPUT_USD_PER_MILLION_TOKENS = 0.15
BASELINE_OUTPUT_USD_PER_MILLION_TOKENS = 0.60


@contextmanager
def append_jsonl(
    output_path: Path,
) -> Iterator[Callable[[Mapping[str, Any]], None]]:
    """Open an append-only JSONL writer that durably flushes each complete row."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        def write_record(record: Mapping[str, Any]) -> None:
            line = json.dumps(dict(record), ensure_ascii=False, default=str) + "\n"
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

        yield write_record


def _estimate_tokens(text: str) -> int:
    """Estimate tokens using a documented 4-characters-per-token heuristic."""
    return math.ceil(len(text) / OLLAMA_CHARACTERS_PER_TOKEN) if text else 0


def _estimated_baseline_cost(input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
    """Return simulated input, output, and total USD cost at the baseline rates."""
    input_cost = input_tokens * BASELINE_INPUT_USD_PER_MILLION_TOKENS / 1_000_000
    output_cost = output_tokens * BASELINE_OUTPUT_USD_PER_MILLION_TOKENS / 1_000_000
    return input_cost, output_cost, input_cost + output_cost


CODE_TASK_IDS = {"general_code_math_3", "general_code_math_4"}
_PYTHON_FENCE = re.compile(
    r"\s*```(?:python)?[ \t]*\r?\n(?P<code>.*?)```\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


def _python_ast(source: str) -> str | None:
    """Return an AST fingerprint for one complete Python answer, or None if malformed."""
    fenced = _PYTHON_FENCE.fullmatch(source)
    if "```" in source and fenced is None:
        return None  # Reject extra fences, trailing prose/code, or unsupported fence languages.
    code = fenced.group("code") if fenced else source
    if not code.strip():
        return None

    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError, ValueError):
        return None

    # Alpha-normalize bound names only. Function names, attributes, builtins,
    # literals, and unresolved names remain significant in the structural match.
    bound_names = dict.fromkeys(
        node.arg if isinstance(node, ast.arg) else node.id
        for node in ast.walk(tree)
        if (isinstance(node, ast.arg) or isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)))
    )
    names = {name: f"_local_{index}" for index, name in enumerate(bound_names)}

    class NormalizeBoundNames(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id in names:
                node.id = names[node.id]
            return node

        def visit_arg(self, node: ast.arg) -> ast.arg:
            if node.arg in names:
                node.arg = names[node.arg]
            return node

    normalized_tree = NormalizeBoundNames().visit(tree)
    return ast.dump(normalized_tree, include_attributes=False)


def _score_response(
    scorer: evaluator.ClinicalEvaluator,
    task_id: str,
    task_type: str,
    response: str,
    expected_answer: str,
) -> int:
    """Use exact supported evaluator methods, with narrow AST matching for code tasks."""
    if task_type == "general_code_math":
        try:
            float(expected_answer.strip())
        except (ValueError, TypeError):
            if task_id not in CODE_TASK_IDS:
                return 0
            expected_ast = _python_ast(expected_answer)
            response_ast = _python_ast(response)
            return int(expected_ast is not None and response_ast == expected_ast)
        return scorer.score_clinical_math(response, expected_answer)
    if task_type in {
        "symptom_triage",
        "drug_interaction_checker",
        "medical_literature_search",
    }:
        return scorer.score_clinical_prose(response, expected_answer)
    return 0


def load_clinical_setup(
    tasks_path: Path = DEFAULT_TASKS_PATH,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Validate the clinical matrix and load each destination's system prompt."""
    tasks = load_clinical_tasks(tasks_path, require_balance=True)
    if len(tasks) != 16:
        raise ValueError(f"Expected 16 clinical matrix tasks, found {len(tasks)} in {tasks_path}")

    prompts: dict[str, str] = {}
    for task in tasks:
        task_type = str(task["type"])
        if task_type not in prompts:
            destination = CLINICAL_DESTINATIONS[task_type]
            prompts[task_type] = destination.prompt_path.read_text(encoding="utf-8").strip()
    return tasks, prompts


def _execute_model_request(
    prompt: str,
    task_type: str,
    system_prompt: str,
    model: str,
    *,
    ollama_url: str,
    timeout: float,
    temperature: float,
    dry_run: bool,
) -> tuple[str, float, str, str | None]:
    """Execute one Ollama cell, converting provider exceptions into row data."""
    if dry_run:
        return "", 0.0, "dry_run", None

    request_started = time.time()
    try:
        response = evaluator.send_to_agent(
            prompt,
            task_type,
            provider="ollama",
            model=model,
            system_prompt=system_prompt,
            ollama_url=ollama_url,
            temperature=temperature,
            timeout=timeout,
        )
        status, error = "ok", None
    except Exception as exc:  # The batch continues after a single model failure.
        response = ""
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    latency_seconds = max(0.0, time.time() - request_started)
    return response, latency_seconds, status, error


def _build_telemetry_record(
    *,
    run_id: str,
    timestamp: str,
    model: str,
    task: Mapping[str, Any],
    system_prompt: str,
    response: str,
    latency_seconds: float,
    status: str,
    score: int | None,
    error: str | None,
) -> dict[str, Any]:
    """Build the persisted row and its simulated token/cost telemetry."""
    prompt = str(task["prompt"])
    combined_input = system_prompt + "\n" + prompt
    input_tokens = _estimate_tokens(combined_input)
    output_tokens = _estimate_tokens(response)
    total_tokens = input_tokens + output_tokens
    input_cost, output_cost, total_cost = _estimated_baseline_cost(input_tokens, output_tokens)
    estimated_wasted = status == "error" or score == 0

    record: dict[str, Any] = {
        "timestamp": timestamp,
        "run_id": run_id,
        "task_id": str(task["id"]),
        "task_type": str(task["type"]),
        "model": model,
        "provider": "ollama",
        "status": status,
        "dry_run": status == "dry_run",
        "prompt_length_chars": len(prompt),
        "system_prompt_length_chars": len(system_prompt),
        "input_prompt_length_chars": len(combined_input),
        "input_prompt_length_definition": "system prompt + user task prompt",
        "estimated_input_tokens": input_tokens,
        "response_length_chars": len(response),
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": total_tokens,
        "latency_seconds": round(latency_seconds, 6),
        "score": score,
        "estimated_baseline_input_cost_usd": round(input_cost, 12),
        "estimated_baseline_output_cost_usd": round(output_cost, 12),
        "estimated_baseline_total_cost_usd": round(total_cost, 12),
        "estimated_wasted_tokens": total_tokens if estimated_wasted else 0,
        "estimated_wasted_cost_usd": round(total_cost if estimated_wasted else 0.0, 12),
        "provider_billed_cost_usd": None,
        "telemetry_assumptions": {
            "token_estimation": f"approximately 1 token per {OLLAMA_CHARACTERS_PER_TOKEN} characters",
            "input_baseline_usd_per_million_tokens": BASELINE_INPUT_USD_PER_MILLION_TOKENS,
            "output_baseline_usd_per_million_tokens": BASELINE_OUTPUT_USD_PER_MILLION_TOKENS,
            "cost_is_simulated_not_billed": True,
            "waste_definition": "estimated total baseline cost for errored requests or score-0 responses",
        },
        "response": response,
    }
    if error is not None:
        record["error"] = error
    return record


def run_batch(
    tasks: Sequence[Mapping[str, Any]],
    system_prompts: Mapping[str, str],
    models: Sequence[str],
    output_path: Path,
    *,
    ollama_url: str = evaluator.DEFAULT_OLLAMA_URL,
    timeout: float = 120.0,
    temperature: float = 0.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run every task against every model, persisting telemetry immediately.

    A provider error is recorded as a failed task and does not prevent later
    matrix cells from running. A logging error propagates so execution stops
    rather than continuing without the requested durable telemetry.
    """
    if not models:
        raise ValueError("At least one Ollama model must be configured")

    run_id = evaluator.new_run_id()
    scorer = evaluator.ClinicalEvaluator()
    attempted = 0
    evaluated = 0
    passed = 0
    errors = 0
    started_batch = time.time()

    with append_jsonl(output_path) as write_record:
        for model in models:
            for task in tasks:
                task_id = str(task["id"])
                task_type = str(task["type"])
                expected_answer = str(task["correct_answer"])
                system_prompt = str(system_prompts[task_type])
                prompt = str(task["prompt"])
                timestamp = datetime.now(timezone.utc).isoformat()
                response, latency_seconds, status, error = _execute_model_request(
                    prompt,
                    task_type,
                    system_prompt,
                    model,
                    ollama_url=ollama_url,
                    timeout=timeout,
                    temperature=temperature,
                    dry_run=dry_run,
                )
                errors += int(status == "error")

                score = None
                if status == "ok":
                    score = _score_response(
                        scorer,
                        task_id,
                        task_type,
                        response,
                        expected_answer,
                    )
                    evaluated += 1
                    passed += score
                record = _build_telemetry_record(
                    run_id=run_id,
                    timestamp=timestamp,
                    model=model,
                    task=task,
                    system_prompt=system_prompt,
                    response=response,
                    latency_seconds=latency_seconds,
                    status=status,
                    score=score,
                    error=error,
                )

                # Each row is flushed and fsynced before the next model request starts.
                write_record(record)
                attempted += 1
                print(
                    f"[{attempted}/{len(models) * len(tasks)}] {model} / {task_id}: "
                    f"score={score}, latency={latency_seconds:.3f}s"
                    + (f" ({status})" if status != "ok" else "")
                )

    elapsed = max(0.0, time.time() - started_batch)
    return {
        "run_id": run_id,
        "attempted": attempted,
        "evaluated": evaluated,
        "passed": passed,
        "errors": errors,
        "accuracy": passed / evaluated if evaluated else None,
        "elapsed_seconds": round(elapsed, 3),
        "output_path": str(output_path),
    }


def _configured_models(cli_models: Sequence[str] | None) -> list[str]:
    """Resolve CLI models, then environment models, then documented local defaults."""
    if cli_models:
        candidates = cli_models
    else:
        configured = os.getenv("BATCH_EVAL_MODELS") or os.getenv("OLLAMA_MODELS")
        candidates = configured.split(",") if configured else list(DEFAULT_MODELS)
    models = list(dict.fromkeys(
        model.strip() for item in candidates for model in item.split(",") if model.strip()
    ))
    if not models:
        raise ValueError("No models configured; pass --models or set BATCH_EVAL_MODELS")
    return models


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local Ollama models on the 16-query clinical routing matrix"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Ollama model names (default from BATCH_EVAL_MODELS, or llama3:8b mistral:7b)",
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", evaluator.DEFAULT_OLLAMA_URL),
    )
    parser.add_argument("--timeout", type=float, default=float(os.getenv("AGENT_TIMEOUT", "120")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("AGENT_TEMPERATURE", "0")))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a full telemetry matrix without contacting Ollama or assigning scores",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        models = _configured_models(args.models)
        tasks, prompts = load_clinical_setup(args.tasks)
        summary = run_batch(
            tasks,
            prompts,
            models,
            args.output,
            ollama_url=args.ollama_url,
            timeout=args.timeout,
            temperature=args.temperature,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Batch evaluation setup or logging failed: {exc}", file=sys.stderr)
        return 2

    accuracy = (
        f"accuracy={summary['accuracy']:.1%}"
        if summary["accuracy"] is not None
        else "accuracy=n/a (dry run)"
    )
    print(
        f"Batch complete: {summary['attempted']} records written, "
        f"{summary['evaluated']} responses scored, {summary['passed']} passed; "
        f"{summary['errors']} model errors; {accuracy}; "
        f"elapsed={summary['elapsed_seconds']:.2f}s"
    )
    print(f"Run ID: {summary['run_id']} | append-only telemetry: {summary['output_path']}")
    if args.dry_run:
        print("Dry run only: no model request was sent; simulated costs are not billed.")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
