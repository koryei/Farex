"""Declare the four clinical sub-agent destinations used by the routing study.

This module is the single source of truth for the clinical routing axis: it
binds each destination name to its handoff target (the agent prompt file in
/agents/) and to the task file that feeds it. The full routing pipeline and
the condition runners are a later step; for now this module declares the
destinations and provides a CLI that validates the wiring against
tasks/clinical_tasks.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLINICAL_TASKS_PATH = PROJECT_ROOT / "tasks" / "clinical_tasks.json"
AGENTS_DIR = PROJECT_ROOT / "agents"


@dataclass(frozen=True)
class ClinicalDestination:
    """One clinical sub-agent destination and where tasks for it are handed off."""

    name: str
    description: str
    prompt_file: str
    tasks_file: Path

    @property
    def prompt_path(self) -> Path:
        return AGENTS_DIR / self.prompt_file


# The four sub-agent destinations. Every clinical task in
# tasks/clinical_tasks.json must name exactly one of these as its `type`.
CLINICAL_DESTINATIONS: dict[str, ClinicalDestination] = {
    destination.name: destination
    for destination in (
        ClinicalDestination(
            name="symptom_triage",
            description=(
                "Handles patient symptom descriptions and classifies urgency "
                "(emergency / urgent / routine) with a recommended action."
            ),
            prompt_file="symptom_triage_agent.txt",
            tasks_file=CLINICAL_TASKS_PATH,
        ),
        ClinicalDestination(
            name="drug_interaction_checker",
            description=(
                "Evaluates medication lists and drug pairs for adverse "
                "interactions, contraindications, and required monitoring."
            ),
            prompt_file="drug_interaction_checker_agent.txt",
            tasks_file=CLINICAL_TASKS_PATH,
        ),
        ClinicalDestination(
            name="medical_literature_search",
            description=(
                "Searches published studies and guidelines for specific "
                "clinical evidence questions; never personal medical advice."
            ),
            prompt_file="medical_literature_search_agent.txt",
            tasks_file=CLINICAL_TASKS_PATH,
        ),
        ClinicalDestination(
            name="general_code_math",
            description=(
                "Handles general data formatting and mathematical calculations "
                "(clinical scoring formulas, unit conversions, small functions)."
            ),
            prompt_file="general_code_math_agent.txt",
            tasks_file=CLINICAL_TASKS_PATH,
        ),
    )
}

REQUIRED_FIELDS = ("id", "type", "prompt", "correct_answer")


def load_clinical_tasks(
    path: Path = CLINICAL_TASKS_PATH,
    *,
    require_balance: bool = True,
) -> list[dict]:
    """Load tasks against the clinical schema; optionally enforce the canonical 4x4 set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = None
    if isinstance(data, dict):
        # Keep the repository's canonical key while accepting the name used by
        # simple benchmark-loop examples and external clinical datasets.
        tasks = data.get("tasks") or data.get("clinical_tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path}: expected a non-empty 'tasks' or 'clinical_tasks' list")

    declared = set(CLINICAL_DESTINATIONS)
    errors: list[str] = []
    seen_ids: set[str] = set()
    per_destination: dict[str, int] = {name: 0 for name in declared}

    if require_balance:
        listed_agents = data.get("agents")
        if set(listed_agents or []) != declared:
            errors.append(
                f"top-level 'agents' {listed_agents!r} does not match declared destinations "
                f"{sorted(declared)}"
            )

    for task in tasks:
        if not isinstance(task, dict):
            errors.append("task is not an object")
            continue
        task_id = str(task.get("id", ""))
        if not task_id:
            errors.append("task missing id")
            continue
        if task_id in seen_ids:
            errors.append(f"duplicate task id: {task_id}")
        seen_ids.add(task_id)

        for field in REQUIRED_FIELDS:
            if not str(task.get(field, "")).strip():
                errors.append(f"{task_id}: missing or empty field {field!r}")

        task_type = str(task.get("type", ""))
        if task_type not in declared:
            errors.append(f"{task_id}: unknown type {task_type!r}")
            continue
        per_destination[task_type] += 1

    if require_balance:
        for name, count in sorted(per_destination.items()):
            if count != 4:
                errors.append(f"destination {name!r} has {count} tasks, expected 4")

    if errors:
        raise ValueError("clinical task file failed validation:\n  - " + "\n  - ".join(errors))
    return tasks


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List or validate the four clinical sub-agent destinations "
            "(symptom_triage, drug_interaction_checker, medical_literature_search, "
            "general_code_math)."
        )
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the destination wiring against tasks/clinical_tasks.json",
    )
    args = parser.parse_args(argv)

    for name, destination in CLINICAL_DESTINATIONS.items():
        prompt_ok = destination.prompt_path.is_file()
        print(
            f"{name}\n  description : {destination.description}\n"
            f"  handoff     : agents/{destination.prompt_file}"
            f" ({'ok' if prompt_ok else 'MISSING'})\n"
            f"  tasks       : {destination.tasks_file.relative_to(PROJECT_ROOT)}"
        )

    if args.validate:
        tasks = load_clinical_tasks(require_balance=True)
        print(f"\nvalidation: OK ({len(tasks)} tasks, 4 per destination)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
