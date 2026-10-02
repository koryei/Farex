"""Configuration and execution for the clinical benchmark."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

if __package__:
    from . import evaluate as evaluator
    from .clinical_agents import CLINICAL_DESTINATIONS, load_clinical_tasks
    from .evaluate import ClinicalEvaluator
else:
    import evaluate as evaluator
    from clinical_agents import CLINICAL_DESTINATIONS, load_clinical_tasks
    from evaluate import ClinicalEvaluator


PROJECT_ROOT = evaluator.PROJECT_ROOT
DEFAULT_TASKS_PATH = PROJECT_ROOT / "tasks" / "clinical_tasks.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "clinical_results.jsonl"
_SECRET_NAME = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.I)


def redact(value: str) -> str:
    """Remove environment credentials and common credential strings from text."""
    for name, secret in os.environ.items():
        if _SECRET_NAME.search(name) and secret and len(secret) >= 4:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(
        r"(?i)\b(api[\s_-]?key|token|secret|password|credential)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        value,
    )
    value = re.sub(r"(https?://)[^/@\s]+@", r"\1[REDACTED]@", value)
    return re.sub(
        r"(?i)([?&](?:api[_-]?key|token|secret|password)=)[^&\s]+",
        r"\1[REDACTED]",
        value,
    )


def _resolve_model(provider: str, requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    if os.getenv("CLINICAL_MODEL"):
        return os.environ["CLINICAL_MODEL"]
    if os.getenv("AGENT_MODEL"):
        return os.environ["AGENT_MODEL"]
    provider_model = os.getenv(f"{provider.upper()}_MODEL")
    return provider_model or ("llama3:8b" if provider == "ollama" else None)


@dataclass(frozen=True)
class ClinicalRunConfig:
    """Resolved settings for one benchmark invocation."""

    tasks_path: Path
    log_path: Path
    provider: str
    model: str
    ollama_url: str
    openrouter_url: str
    requesty_url: str
    api_key_env: str
    temperature: float
    timeout: float
    task_ids: Optional[Sequence[str]] = None
    agent_override: Optional[str] = None
    limit: Optional[int] = None

    @classmethod
    def from_environment(
        cls,
        *,
        tasks_path: Optional[Path | str] = None,
        log_path: Optional[Path | str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        openrouter_url: Optional[str] = None,
        requesty_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        task_ids: Optional[Sequence[str]] = None,
        agent_override: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> "ClinicalRunConfig":
        """Resolve explicit options over clinical/shared environment defaults."""
        provider_value = (
            provider
            or os.getenv("CLINICAL_PROVIDER")
            or os.getenv("AGENT_PROVIDER", "ollama")
        )
        normalized_provider = evaluator.PROVIDER_ALIASES.get(
            evaluator._normalize_type(provider_value)
        )
        if normalized_provider is None:
            raise ValueError("provider must be 'ollama', 'openrouter', or 'requesty'")

        resolved_model = _resolve_model(normalized_provider, model)
        if not resolved_model:
            raise ValueError(
                f"A model is required; set CLINICAL_MODEL or "
                f"{normalized_provider.upper()}_MODEL"
            )
        if agent_override and agent_override not in CLINICAL_DESTINATIONS:
            raise ValueError(f"Unknown clinical agent override: {agent_override}")
        if limit is not None and limit < 0:
            raise ValueError("limit must be zero or greater")
        key_env_default = (
            "REQUESTY_API_KEY_ENV"
            if normalized_provider == "requesty"
            else "OPENROUTER_API_KEY_ENV"
        )
        key_env_fallback = (
            "REQUESTY_API_KEY"
            if normalized_provider == "requesty"
            else "OPENROUTER_API_KEY"
        )
        return cls(
            tasks_path=Path(
                tasks_path
                or os.getenv(
                    "CLINICAL_TASKS_PATH",
                    os.getenv("AGENT_TASKS_PATH", str(DEFAULT_TASKS_PATH)),
                )
            ),
            log_path=Path(
                log_path
                or os.getenv(
                    "CLINICAL_LOG_PATH",
                    os.getenv("AGENT_LOG_PATH", str(DEFAULT_LOG_PATH)),
                )
            ),
            provider=normalized_provider,
            model=resolved_model,
            ollama_url=ollama_url
            or os.getenv("OLLAMA_BASE_URL", evaluator.DEFAULT_OLLAMA_URL),
            openrouter_url=openrouter_url
            or os.getenv("OPENROUTER_BASE_URL", evaluator.DEFAULT_OPENROUTER_URL),
            requesty_url=requesty_url
            or os.getenv("REQUESTY_BASE_URL", evaluator.DEFAULT_REQUESTY_URL),
            api_key_env=api_key_env or os.getenv(key_env_default, key_env_fallback),
            temperature=float(
                os.getenv("AGENT_TEMPERATURE", "0.0") if temperature is None else temperature
            ),
            timeout=float(os.getenv("AGENT_TIMEOUT", "60.0") if timeout is None else timeout),
            task_ids=task_ids,
            agent_override=agent_override,
            limit=limit,
        )


def _system_prompt(agent_type: str) -> str:
    destination = CLINICAL_DESTINATIONS[agent_type]
    if not destination.prompt_path.is_file():
        raise FileNotFoundError(f"Missing clinical agent prompt: {destination.prompt_file}")
    return destination.prompt_path.read_text(encoding="utf-8").strip()


def _load_run_data(
    config: ClinicalRunConfig,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    tasks = load_clinical_tasks(config.tasks_path, require_balance=False)
    counts: dict[str, int] = {}
    prompts: dict[str, str] = {}
    for task in tasks:
        task_type = str(task["type"])
        if task_type not in prompts:
            prompts[task_type] = _system_prompt(task_type)
        counts[task_type] = counts.get(task_type, 0) + 1
    if config.agent_override and config.agent_override not in prompts:
        prompts[config.agent_override] = _system_prompt(config.agent_override)
    return tasks, counts, prompts


def validate_setup(config: ClinicalRunConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate task and prompt wiring; return loaded tasks and destination counts."""
    tasks, counts, _ = _load_run_data(config)
    return tasks, counts


def iter_benchmark(
    config: ClinicalRunConfig,
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Yield each logged result with its position for incremental CLI reporting."""
    tasks, _, prompts = _load_run_data(config)
    if config.task_ids is not None:
        by_id = {str(task["id"]): task for task in tasks}
        missing = [task_id for task_id in config.task_ids if task_id not in by_id]
        if missing:
            raise KeyError("Unknown task id(s): " + ", ".join(missing))
        tasks = [by_id[task_id] for task_id in config.task_ids]
    if config.limit is not None:
        tasks = tasks[: config.limit]

    total = len(tasks)
    run_id = evaluator.new_run_id()
    clinical_evaluator = ClinicalEvaluator()
    for index, task in enumerate(tasks, start=1):
        task_id = str(task["id"])
        task_type = str(task["type"])
        agent_type = config.agent_override or task_type
        started = time.perf_counter()
        result: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "run_log": evaluator.run_log_path(config.log_path, run_id).name,
            "task_id": task_id,
            "task_type": task_type,
            "agent_type": agent_type,
            "provider": config.provider,
            "model": redact(config.model),
            "temperature": config.temperature,
            "timeout_s": config.timeout,
            "prompt": redact(str(task["prompt"])),
            "expected_answer": redact(str(task.get("correct_answer", ""))),
            "status": "ok",
            "response": "",
            "score": 0,
        }
        try:
            result["response"] = redact(
                evaluator.send_to_agent(
                    str(task["prompt"]),
                    agent_type,
                    provider=config.provider,
                    model=config.model,
                    system_prompt=prompts[agent_type],
                    ollama_url=config.ollama_url,
                    openrouter_url=config.openrouter_url,
                    requesty_url=config.requesty_url,
                    api_key_env=config.api_key_env,
                    temperature=config.temperature,
                    timeout=config.timeout,
                )
            )
            result["score"] = clinical_evaluator.process_telemetry_run(
                task_id=task_id,
                task_type=task_type,
                response_text=result["response"],
                target=str(task.get("correct_answer", "")),
            )
        except Exception as exc:  # Preserve individual failures and continue the run.
            result["status"] = "error"
            result["error"] = redact(f"{type(exc).__name__}: {exc}")

        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        evaluator.append_run_result(result, config.log_path)
        yield index, total, result


def run_benchmark(config: ClinicalRunConfig) -> list[dict[str, Any]]:
    """Run the selected tasks and return their logged results."""
    return [result for _, _, result in iter_benchmark(config)]


def summarize_scores(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replay logged responses through ClinicalEvaluator for aggregate accuracy telemetry.

    Only successful responses (status == "ok") are evaluated, mirroring the live
    pipeline where request failures never reach the scorer. Works both on results
    returned by run_benchmark/iter_benchmark and on raw JSONL log entries.
    """
    replay_evaluator = ClinicalEvaluator()
    for result in results:
        if result.get("status") != "ok":
            continue
        replay_evaluator.process_telemetry_run(
            task_id=str(result.get("task_id", "")),
            task_type=str(result.get("task_type", "")),
            response_text=str(result.get("response", "")),
            target=str(result.get("expected_answer", "")),
        )
    return {
        "evaluated": replay_evaluator.total_evaluated,
        "passed": replay_evaluator.total_passed,
        "accuracy": replay_evaluator.accuracy,
    }
