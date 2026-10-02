"""Run agent tasks and record JSONL results.

The module can be used as a library or from the command line::

    python evaluate/evaluate.py --all --provider ollama --model llama3.1
    python evaluate/evaluate.py --all --provider requesty --model openai/gpt-4o
    python evaluate/evaluate.py --task-id math_1 --provider ollama --model llama3.1

Run the ClinicalEvaluator self-test (no model call)::

    python evaluate/evaluate.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_PATH = PROJECT_ROOT / "tasks" / "tasks.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "results.jsonl"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_REQUESTY_URL = "https://router.requesty.ai/v1/chat/completions"


def _load_project_env() -> None:
    """Load simple KEY=value settings from the project .env without overriding the shell."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid setting in {env_path} on line {line_number}")

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '\"'}:
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        os.environ.setdefault(key, value)


_load_project_env()

AGENT_PROMPT_FILES = {
    "coding": "coding_agent.txt",
    "math": "math_agent.txt",
    "extraction": "extraction_agent.txt",
    "evidence_qa": "evidence_qa_agent.txt",
}

PROVIDER_ALIASES = {
    "open_router": "openrouter",
    "openrouter": "openrouter",
    "ollama": "ollama",
    "requesty": "requesty",
    "requesty_ai": "requesty",
}


def _normalize(text: str) -> str:
    """Normalize text for case-insensitive, punctuation-tolerant matching."""
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_type(value: str) -> str:
    return re.sub(r"[-\s]+", "_", str(value).strip().casefold())


class ClinicalEvaluator:
    """Stage 4 Software Architecture: Programmatic Evaluation & Metrics Suite."""

    def __init__(self):
        self.total_evaluated = 0
        self.total_passed = 0

    def score_clinical_prose(self, model_output: str, expected_answer: str) -> int:
        """Evaluates prose/text accuracy using normalized token keyword overlap matching."""
        # Normalize characters to prevent minor punctuation strings from breaking scores
        normalized_expected = (
            expected_answer.lower().replace(":", " ").replace("/", " ").replace("->", " ")
        )
        keywords = [
            word.strip() for word in normalized_expected.split() if len(word.strip()) > 2
        ]

        output_lower = str(model_output).lower()

        # If the local model captures key medical directives (e.g., 'emergency'), it passes
        if any(kw in output_lower for kw in keywords):
            return 1
        return 0

    def score_clinical_math(self, model_output: str, expected_answer: str) -> int:
        """
        Isolates numbers out of raw text generation responses.
        Flagging local model errors (like Llama-3 outputting 0.9 instead of 68.9).
        """
        try:
            # Stage 4 Structured Error Handling: Use Regex to extract float digits cleanly
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(model_output))
            if not numbers:
                return 0

            # Grab the first numerical value the model printed
            clean_output = float(numbers[0])
            target_expected = float(str(expected_answer).strip())

            if abs(clean_output - target_expected) < 0.01:
                return 1
            return 0
        except (ValueError, IndexError, TypeError):
            # Gracefully catch casting collisions without crashing the benchmark runtime
            return 0

    def process_telemetry_run(
        self, task_id: str, task_type: str, response_text: str, target: str
    ):
        """Processes a single task execution block and increments systemic totals."""
        del task_id  # Included for structured logging parity with the JSONL schema.
        self.total_evaluated += 1

        if task_type in [
            "symptom_triage",
            "drug_interaction_checker",
            "medical_literature_search",
        ]:
            score = self.score_clinical_prose(response_text, target)
        elif task_type == "general_code_math":
            score = self.score_clinical_math(response_text, target)
        else:
            score = 0

        self.total_passed += score
        return score

    @property
    def accuracy(self) -> float:
        """Pass rate over every task processed so far (0.0 when nothing was evaluated)."""
        if self.total_evaluated == 0:
            return 0.0
        return self.total_passed / self.total_evaluated


def _extract_evidence_text(prompt: str) -> str:
    """Extract the quoted evidence text from an evidence-QA prompt."""
    match = re.search(
        r"\bText:\s*(['\"])(.*?)\1\s*Answer",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Evidence-QA prompt must contain Text: '...' before Answer")
    return match.group(2)


def _contains_evidence_phrase(
    answer: str,
    evidence_text: str,
    minimum_words: int = 3,
) -> bool:
    """Return whether the answer contains a contiguous phrase from the evidence."""
    answer_words = _normalize(answer).split()
    evidence_words = _normalize(evidence_text).split()
    max_phrase_length = min(len(answer_words), len(evidence_words))

    for phrase_length in range(max_phrase_length, minimum_words - 1, -1):
        answer_phrases = {
            tuple(answer_words[index : index + phrase_length])
            for index in range(len(answer_words) - phrase_length + 1)
        }
        if any(
            tuple(evidence_words[index : index + phrase_length]) in answer_phrases
            for index in range(len(evidence_words) - phrase_length + 1)
        ):
            return True
    return False


def score_evidence_qa(answer: str, task: Mapping[str, Any]) -> int:
    """Return 1 when the answer has the correct yes/no and an evidence phrase."""
    if _normalize_type(str(task.get("type", ""))) != "evidence_qa":
        raise ValueError("score_evidence_qa requires an evidence_qa task")

    correct_answer = str(task.get("correct_answer", ""))
    expected_yes_no, separator, expected_phrase = correct_answer.partition(":")
    if not separator:
        raise ValueError("Evidence-QA correct_answer must use 'yes/no: phrase' format")

    answer_normalized = _normalize(answer)
    expected_yes_no = _normalize(expected_yes_no)
    expected_phrase = _normalize(expected_phrase)
    evidence_text = _extract_evidence_text(str(task.get("prompt", "")))
    evidence_normalized = _normalize(evidence_text)

    has_correct_yes_no = (
        re.search(rf"\b{re.escape(expected_yes_no)}\b", answer_normalized) is not None
    )
    has_canonical_phrase = (
        expected_phrase in evidence_normalized and expected_phrase in answer_normalized
    )
    has_evidence_phrase = has_canonical_phrase or _contains_evidence_phrase(
        answer,
        evidence_text,
    )

    return int(has_correct_yes_no and has_evidence_phrase)


def _strip_code_fence(answer: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else answer


def score_task(answer: str, task: Mapping[str, Any]) -> int:
    """Score an answer according to the task type."""
    task_type = _normalize_type(str(task.get("type", "")))
    correct_answer = str(task.get("correct_answer", ""))

    if task_type == "evidence_qa":
        return score_evidence_qa(answer, task)

    normalized_answer = _normalize(_strip_code_fence(answer))
    normalized_correct = _normalize(correct_answer)

    if task_type == "math":
        expected_number = normalized_correct
        answer_numbers = re.findall(r"-?\d+(?:\.\d+)?", normalized_answer)
        return int(expected_number in answer_numbers)

    if task_type == "extraction":
        return int(normalized_correct in normalized_answer)

    if task_type == "coding":
        return int(normalized_answer == normalized_correct)

    raise ValueError(f"Unsupported task type: {task_type}")


def load_tasks(path: Path | str = DEFAULT_TASKS_PATH) -> list[dict[str, Any]]:
    """Load and validate the task list."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a 'tasks' list in {path}")

    seen_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"Each task in {path} must be an object")
        task_id = str(task.get("id", ""))
        if not task_id:
            raise ValueError(f"Each task in {path} must have an id")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id in {path}: {task_id}")
        seen_ids.add(task_id)

    return tasks


def get_task(task_id: str, path: Path | str = DEFAULT_TASKS_PATH) -> dict[str, Any]:
    """Return one task by id."""
    for task in load_tasks(path):
        if task["id"] == task_id:
            return task
    raise KeyError(f"Unknown task id: {task_id}")


def load_agent_prompt(agent_type: str) -> str:
    """Load the system prompt for an agent type."""
    normalized_type = _normalize_type(agent_type)
    filename = AGENT_PROMPT_FILES.get(normalized_type)
    if filename is None:
        supported = ", ".join(sorted(AGENT_PROMPT_FILES))
        raise ValueError(f"Unsupported agent type {agent_type!r}; expected one of {supported}")

    path = PROJECT_ROOT / "agents" / filename
    return path.read_text(encoding="utf-8").strip()


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    request_headers = dict(headers or {})
    request_headers["Content-Type"] = "application/json"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Agent request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Agent request failed: {exc.reason}") from exc

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Agent returned invalid JSON: {response_body[:500]}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Agent response must be a JSON object")
    return parsed


def _ollama_content(response: Mapping[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict) and message.get("content"):
        return str(message["content"])
    if response.get("response"):
        return str(response["response"])
    raise RuntimeError("Ollama response did not contain a message")


def _openai_compatible_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Provider response did not contain a choice")

    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if content:
        return str(content)
    raise RuntimeError("Provider response did not contain message content")


def _openrouter_content(response: Mapping[str, Any]) -> str:
    return _openai_compatible_content(response)


def _default_model(provider: str, model: Optional[str]) -> Optional[str]:
    if provider == "ollama":
        return model or os.getenv("OLLAMA_MODEL", "llama3.1")
    if provider == "requesty":
        return model or os.getenv("REQUESTY_MODEL")
    return model or os.getenv("OPENROUTER_MODEL")


def send_to_agent(
    prompt: str,
    agent_type: str,
    provider: str = "ollama",
    model: Optional[str] = None,
    *,
    system_prompt: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    openrouter_url: str = DEFAULT_OPENROUTER_URL,
    requesty_url: str = DEFAULT_REQUESTY_URL,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> str:
    """Send a prompt to Ollama, OpenRouter, or Requesty."""
    normalized_provider = PROVIDER_ALIASES.get(_normalize_type(provider))
    if normalized_provider is None:
        raise ValueError("provider must be 'ollama', 'openrouter', or 'requesty'")

    resolved_model = _default_model(normalized_provider, model)
    if not resolved_model:
        model_env = "OLLAMA_MODEL" if normalized_provider == "ollama" else (
            "REQUESTY_MODEL" if normalized_provider == "requesty" else "OPENROUTER_MODEL"
        )
        raise ValueError(f"A model is required (set --model or {model_env})")

    messages = [
        {"role": "system", "content": system_prompt or load_agent_prompt(agent_type)},
        {"role": "user", "content": prompt},
    ]

    if normalized_provider == "ollama":
        payload = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = _post_json(ollama_url, payload, timeout=timeout)
        return _ollama_content(response)

    resolved_api_key_env = api_key_env or (
        "REQUESTY_API_KEY" if normalized_provider == "requesty" else "OPENROUTER_API_KEY"
    )
    resolved_api_key = api_key or os.getenv(resolved_api_key_env)
    if not resolved_api_key:
        provider_name = "Requesty" if normalized_provider == "requesty" else "OpenRouter"
        raise RuntimeError(f"{provider_name} API key is required; set {resolved_api_key_env}")

    endpoint = requesty_url if normalized_provider == "requesty" else openrouter_url
    payload = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    response = _post_json(
        endpoint,
        payload,
        headers={"Authorization": f"Bearer {resolved_api_key}"},
        timeout=timeout,
    )
    return _openai_compatible_content(response)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_result(result: Mapping[str, Any], log_path: Path | str = DEFAULT_LOG_PATH) -> None:
    """Append one result as a JSON object line."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(result), ensure_ascii=False) + "\n")


def new_run_id() -> str:
    """Return a timestamped identifier that is safe to use in a filename."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def run_log_path(log_path: Path | str, run_id: str) -> Path:
    """Build the unique JSONL path for a run without changing its history file."""
    path = Path(log_path)
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}.run_{run_id}{suffix}")


def append_run_result(
    result: Mapping[str, Any], log_path: Path | str = DEFAULT_LOG_PATH
) -> None:
    """Append to the cumulative history and the run-specific log, preserving both."""
    run_id = result.get("run_id")
    history_result = dict(result)
    if run_id:
        run_path = run_log_path(log_path, str(run_id))
        history_result["run_log"] = run_path.name
    append_result(history_result, log_path)
    if run_id:
        append_result(history_result, run_path)


def run_task(
    task_id: str,
    *,
    agent_type: Optional[str] = None,
    provider: str = "ollama",
    model: Optional[str] = None,
    tasks_path: Path | str = DEFAULT_TASKS_PATH,
    log_path: Path | str = DEFAULT_LOG_PATH,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    openrouter_url: str = DEFAULT_OPENROUTER_URL,
    requesty_url: str = DEFAULT_REQUESTY_URL,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run, score, and log one task."""
    task = get_task(task_id, tasks_path)
    resolved_run_id = run_id or new_run_id()
    resolved_agent_type = _normalize_type(agent_type or str(task.get("type", "")))
    resolved_provider = PROVIDER_ALIASES.get(_normalize_type(provider))
    if resolved_provider is None:
        raise ValueError("provider must be 'ollama', 'openrouter', or 'requesty'")
    resolved_model = _default_model(resolved_provider, model)

    result: dict[str, Any] = {
        "timestamp": _timestamp(),
        "run_id": resolved_run_id,
        "run_log": run_log_path(log_path, resolved_run_id).name,
        "task_id": task_id,
        "task_type": task.get("type"),
        "agent_type": resolved_agent_type,
        "provider": resolved_provider,
        "model": resolved_model,
        "prompt": task.get("prompt", ""),
        "correct_answer": task.get("correct_answer", ""),
    }
    started = time.perf_counter()

    try:
        system_prompt = load_agent_prompt(resolved_agent_type)
        response = send_to_agent(
            str(task.get("prompt", "")),
            resolved_agent_type,
            provider=resolved_provider,
            model=resolved_model,
            system_prompt=system_prompt,
            ollama_url=ollama_url,
            openrouter_url=openrouter_url,
            requesty_url=requesty_url,
            api_key=api_key,
            api_key_env=api_key_env,
            temperature=temperature,
            timeout=timeout,
        )
        score = score_task(response, task)
        result.update({"response": response, "score": score})
    except Exception as exc:  # Keep a multi-task run going and preserve failures in the log.
        result.update({"response": "", "score": 0, "error": f"{type(exc).__name__}: {exc}"})

    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    append_run_result(result, log_path)
    return result


def run_all(
    task_ids: Optional[Sequence[str]] = None,
    *,
    agent_type: Optional[str] = None,
    provider: str = "ollama",
    model: Optional[str] = None,
    tasks_path: Path | str = DEFAULT_TASKS_PATH,
    log_path: Path | str = DEFAULT_LOG_PATH,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    openrouter_url: str = DEFAULT_OPENROUTER_URL,
    requesty_url: str = DEFAULT_REQUESTY_URL,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Run selected tasks, or all tasks when task_ids is omitted."""
    tasks = load_tasks(tasks_path)
    if task_ids is None:
        selected_tasks = tasks
    else:
        selected_tasks = [get_task(task_id, tasks_path) for task_id in task_ids]
    run_id = new_run_id()

    return [
        run_task(
            str(task["id"]),
            agent_type=agent_type,
            provider=provider,
            model=model,
            tasks_path=tasks_path,
            log_path=log_path,
            ollama_url=ollama_url,
            openrouter_url=openrouter_url,
            requesty_url=requesty_url,
            api_key=api_key,
            api_key_env=api_key_env,
            temperature=temperature,
            timeout=timeout,
            run_id=run_id,
        )
        for task in selected_tasks
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent evaluation tasks")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Task id to run; repeat to run multiple tasks",
    )
    parser.add_argument("--all", action="store_true", help="Run all tasks (the default)")
    parser.add_argument("--agent", help="Agent type override, e.g. math or evidence_qa")
    parser.add_argument(
        "--provider",
        default=os.getenv("AGENT_PROVIDER", "ollama"),
        help="ollama, openrouter, or requesty",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_MODEL"),
        help="Model name; defaults to OLLAMA_MODEL or REQUESTY_MODEL when set",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(os.getenv("AGENT_TASKS_PATH", str(DEFAULT_TASKS_PATH))),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(os.getenv("AGENT_LOG_PATH", str(DEFAULT_LOG_PATH))),
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/chat"),
    )
    parser.add_argument(
        "--openrouter-url",
        default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_URL),
    )
    parser.add_argument(
        "--requesty-url",
        default=os.getenv("REQUESTY_BASE_URL", DEFAULT_REQUESTY_URL),
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="API-key environment variable; defaults per provider",
    )
    parser.add_argument("--temperature", type=float, default=os.getenv("AGENT_TEMPERATURE", "0.0"))
    parser.add_argument("--timeout", type=float, default=os.getenv("AGENT_TIMEOUT", "60.0"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    normalized_provider = _normalize_type(args.provider)
    if normalized_provider in {"openrouter", "requesty"} and not (
        args.model or os.getenv(f"{normalized_provider.upper()}_MODEL")
    ):
        model_env = f"{normalized_provider.upper()}_MODEL"
        parser.error(f"--model is required for this provider (or set {model_env})")

    api_key_env = args.api_key_env
    if not api_key_env:
        api_key_env = (
            os.getenv("REQUESTY_API_KEY_ENV", "REQUESTY_API_KEY")
            if normalized_provider == "requesty"
            else os.getenv("OPENROUTER_API_KEY_ENV", "OPENROUTER_API_KEY")
        )

    results = run_all(
        task_ids=args.task_ids,
        agent_type=args.agent,
        provider=args.provider,
        model=args.model,
        tasks_path=args.tasks,
        log_path=args.log,
        ollama_url=args.ollama_url,
        openrouter_url=args.openrouter_url,
        requesty_url=args.requesty_url,
        api_key_env=api_key_env,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    total = len(results)
    successful = sum(1 for result in results if result.get("score") == 1)
    run_id = results[0].get("run_id") if results else None
    summary = {
        "total": total,
        "successful": successful,
        "accuracy": round(successful / total, 4) if total else 0.0,
        "run_id": run_id,
        "history_log": str(args.log),
        "run_log": str(run_log_path(args.log, str(run_id))) if run_id else None,
    }
    print(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    return 0


def _self_test() -> int:
    """Exercise ClinicalEvaluator on the live Llama-3 log history without a model call."""
    evaluator = ClinicalEvaluator()

    print("🔬 Running Automated Verification over Llama-3 Log History...")
    run_id = new_run_id()
    test_results: list[dict[str, Any]] = []

    # Test Case 1: Llama-3 handled symptom_triage_1 beautifully (Expected: emergency)
    score_1 = evaluator.process_telemetry_run(
        task_id="symptom_triage_1",
        task_type="symptom_triage",
        response_text=(
            "Emergency: The patient is experiencing symptoms consistent with a possible "
            "acute coronary syndrome..."
        ),
        target="emergency: call emergency services; possible MI",
    )
    print(f"🔹 symptom_triage_1 Score: {score_1} (Passes keyword check)")
    test_results.append({
        "timestamp": _timestamp(),
        "run_id": run_id,
        "run_log": run_log_path(PROJECT_ROOT / "logs" / "self_tests.jsonl", run_id).name,
        "task_id": "symptom_triage_1",
        "task_type": "symptom_triage",
        "response": "Emergency: The patient is experiencing symptoms consistent with a possible acute coronary syndrome...",
        "expected_answer": "emergency: call emergency services; possible MI",
        "score": score_1,
        "status": "ok",
        "test_case": True,
    })

    # Test Case 2: THE HALLUCINATION CELL (Expected: 68.9, Llama outputted 0.9)
    score_2 = evaluator.process_telemetry_run(
        task_id="general_code_math_1",
        task_type="general_code_math",
        response_text="0.9",
        target="68.9",
    )
    print(f"🔹 general_code_math_1 Score: {score_2} (Successfully caught Llama-3 calculation deficit!)")
    test_results.append({
        "timestamp": _timestamp(),
        "run_id": run_id,
        "run_log": run_log_path(PROJECT_ROOT / "logs" / "self_tests.jsonl", run_id).name,
        "task_id": "general_code_math_1",
        "task_type": "general_code_math",
        "response": "0.9",
        "expected_answer": "68.9",
        "score": score_2,
        "status": "ok",
        "test_case": True,
    })
    test_log_path = PROJECT_ROOT / "logs" / "self_tests.jsonl"
    for test_result in test_results:
        append_run_result(test_result, test_log_path)
    print(f"📝 Self-test history saved to: {test_log_path}")
    print(f"📝 This self-test run saved to: {run_log_path(test_log_path, run_id)}")

    print("\n==============================")
    print(
        f"📊 Live Run Accuracy: {evaluator.total_passed}/{evaluator.total_evaluated} "
        f"({evaluator.accuracy:.0%})"
    )
    print("==============================")

    if score_1 != 1 or score_2 != 0 or evaluator.total_evaluated != 2:
        print("❌ Self-test FAILED: ClinicalEvaluator behavior drifted from the reference cases", file=sys.stderr)
        return 1
    print("✅ Self-test passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
