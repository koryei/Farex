# My research project
What is this? ->
- I study what happens when a multi-agent system sends tasks to the wrong agent.
- I use 4 agent types: coding, math, extraction, evidence-qa.
- I test 5 conditions: perfect routing, random routing, wrong routing, local router, local + escalation.

How to run (simple steps) ->
1. Install Ollama on your Mac / server and pull a model (for example, `ollama pull llama3:8b` for clinical tasks or `ollama pull llama3.1` for generic tasks).
2. Copy `.env.example` to `.env` and customize the settings for your run. `.env` is git-ignored.
3. Put your tasks in `/tasks/` or configure a different task file.
4. Run the evaluator. It appends results to `/logs/results.jsonl` by default.

Run all tasks with your `.env` defaults ->
`python evaluate/evaluate.py --all`

Run one task ->
`python evaluate/evaluate.py --task-id evidence_qa_1`

Override a setting for one run with a command-line argument ->
`python evaluate/evaluate.py --all --provider ollama --model mistral-nemo`

Configuration ->
- `AGENT_PROVIDER`: default provider (`ollama`, `openrouter`, or `requesty`; default `ollama`).
- `AGENT_MODEL`: optional model override; otherwise use the selected provider's model variable.
- `OLLAMA_MODEL`, `OPENROUTER_MODEL`, `REQUESTY_MODEL`: provider-specific models. Hosted providers require a model.
- `OLLAMA_BASE_URL`, `OPENROUTER_BASE_URL`, `REQUESTY_BASE_URL`: API endpoints.
- `OPENROUTER_API_KEY`, `REQUESTY_API_KEY`: hosted provider credentials. Keep secrets in your ignored `.env` or shell, never commit them.
- `AGENT_TEMPERATURE`, `AGENT_TIMEOUT`: generation temperature and request timeout in seconds.
- `AGENT_TASKS_PATH`, `AGENT_LOG_PATH`: optional task and result file paths.

The evaluator loads `.env` automatically from the project root, without needing `python-dotenv`. Existing shell variables override `.env`; command-line options override both. `.env` supports `KEY=value`, optional `export KEY=value`, blank lines, and comments. Copy the variable names and example values from `.env.example`; replace/remove the placeholder credentials before using hosted providers.

OpenRouter example: set `AGENT_PROVIDER=openrouter`, `OPENROUTER_MODEL=your-model`, and `OPENROUTER_API_KEY=your-key` in `.env`. Requesty example: set `AGENT_PROVIDER=requesty`, `REQUESTY_MODEL=openai/gpt-4o`, and `REQUESTY_API_KEY=your-key`. Requesty supports models from multiple providers using `provider/model` IDs; you can also override its endpoint with `REQUESTY_BASE_URL` or `--requesty-url`.

Each task is sent to the agent matching its task type, scored, and appended to `/logs/results.jsonl` as one JSON object per line. Every invocation also receives a unique run ID and a separate `results.run_<run-id>.jsonl` file in the same log directory. Both files are append-only: historical rows and prior run files are preserved; rerunning evaluations never replaces them.

What I measure ->
- Routing accuracy (was the right agent chosen?)
- Task success (did the answer match?)
- Tokens used
- Time (latency)
- Estimated token/cost waste (batch mode; simulated baseline rates, not billed)

Files ->
- /tasks/ : the 10-50 test questions
- /agents/ : the agent prompts
- /logs/ : every run saves a JSON file
- /evaluate/ : my counting script
- /results/ : multi-model batch telemetry (JSONL), run summary, and comparison figure

Clinical scenario set (added) ->
Alongside the generic set there is a clinical routing benchmark with four sub-agent destinations declared in `evaluate/clinical_agents.py`:
- `symptom_triage`: patient symptom descriptions and urgency classification.
- `drug_interaction_checker`: medication lists, interactions, and safety risks.
- `medical_literature_search`: published studies and guidelines for clinical questions.
- `general_code_math`: data formatting and mathematical calculations.

The 16 text queries (4 per destination) live in `tasks/clinical_tasks.json`; each destination uses its prompt in `/agents/` (for example `symptom_triage_agent.txt`). Run the clinical benchmark with the provider configured in `.env`:
`python evaluate/run_clinical.py`

The clinical runner uses the existing `.env` loader and provider API, including configured endpoints, credentials, temperature, and timeout. It uses `CLINICAL_PROVIDER` and `CLINICAL_MODEL` when set; otherwise it follows shared `AGENT_PROVIDER`, `AGENT_MODEL`, and provider-specific model settings (falling back to Ollama and `llama3:8b` if no model is configured). It also honors `AGENT_TASKS_PATH`/`AGENT_LOG_PATH` unless clinical-specific paths are set. Shell variables override `.env`, and command-line options override both. The runner appends one structured JSON object per task to `logs/clinical_results.jsonl`, including run/task IDs, prompt, expected answer, response, score, status/error, model, latency, and the per-run log path; secrets are redacted and per-task request failures do not stop later tasks. Each invocation also appends its rows to a new `clinical_results.run_<run-id>.jsonl` file. The cumulative file and all per-run files are append-only and are never overwritten by a later evaluation. Use `--task-id symptom_triage_1` for a focused run, `--agent drug_interaction_checker` to override routing, or `--validate-only` to check configuration and task/prompt wiring without calling the model. Task files may use either a top-level `tasks` or `clinical_tasks` list.

Run the clinical benchmark from the repository root:
`python evaluate/run_clinical.py`

Scoring ->
`evaluate/evaluate.py` defines `ClinicalEvaluator`, the Stage 4 programmatic evaluation suite. It scores each clinical task by type: prose tasks (`symptom_triage`, `drug_interaction_checker`, `medical_literature_search`) use keyword-overlap matching against the expected answer, and `general_code_math` tasks extract the model's first number with regex and pass only when it is within 0.01 of the expected value. Every clinical run now logs a `score` per task (0/1) alongside the response, and the runner prints a final `ClinicalEvaluator` accuracy line. A `--self-test` flag replays the reference Llama-3 cases (a passing triage response and the caught 68.9-vs-0.9 calculation error) without calling a model. The cases append to `logs/self_tests.jsonl` and a unique per-self-test JSONL file:
`python evaluate/evaluate.py --self-test`

Check your `.env` and dataset without sending any model request:
`python evaluate/run_clinical.py --validate-only`

Run a quick subset:
`python evaluate/run_clinical.py --limit 2`

Generate a dual-axis latency and accuracy dashboard from saved JSONL logs (the default scans all JSONL files under `logs/` and writes `evaluate/llama3_performance_dashboard.png`):
`python evaluate/plot_metrics.py`

Filter the dashboard to a model, or choose specific input log files and an output path:
`python evaluate/plot_metrics.py --model llama3:8b`
`python evaluate/plot_metrics.py --logs logs/clinical_results.jsonl --output evaluate/clinical_dashboard.png`

The dashboard reads actual run records, deduplicates the cumulative/per-run copies, and scores older clinical logs that predate the `score` field. Install the plotting libraries if needed with `python -m pip install matplotlib numpy`.

Two-model batch comparison (batch_eval.py) ->
To compare local models head-to-head, `evaluate/batch_eval.py` runs the 16-query clinical matrix against each model. Models are processed strictly sequentially (all 16 tasks for one model finish before the next model starts), so only one request is in flight and local memory stays bounded with large models loaded. Each cell is scored by `ClinicalEvaluator` inside the run loop and appended immediately (flush + fsync per row) to `results/evaluation_run.jsonl`, so completed rows survive a mid-run crash. Rows record the timestamp, model, task id/type, latency, raw response, and estimated token/cost/waste telemetry (simulated baseline rates, clearly marked not billed). A `--dry-run` writes the full telemetry matrix without contacting Ollama.

Run it ->
`python evaluate/batch_eval.py --models llama3:8b mistral:7b`
`python evaluate/batch_eval.py --dry-run`
`python evaluate/plot_batch_comparison.py` (four-panel figure: accuracy by domain, mean latency, per-task scatter, per-task score matrix)

First real run (2026-10-03) ->
Both models completed all 16 cells with zero request errors and tied at 75% overall (12/16), but the tie hides a clean domain split:
- Symptom triage, drug interaction check, medical literature search: 12/12 prose tasks correct for BOTH models (100%)
- Clinical math/code: 0/4 for BOTH models (0%)
- llama3:8b reproduced the reference hallucination live: printed `0.9` where `68.9` was expected (Cockcroft-Gault); mistral:7b printed `108.8` plus unwanted prose
- 2 of llama3:8b's 4 math/code failures are functionally plausible code that only fails the strict AST structural match - a scoring-strictness caveat when reading the 0%

Full write-up with tables and the comparison figure: `results/batch_summary.md` (+ `results/batch_model_comparison.png`). Takeaway for the routing conditions: local 7-8B models are reliable clinical prose answerers but unreliable calculators - exactly what the local-router + escalation condition should exploit. (Results narrative drafted with AI assistance; see the disclaimer in `results/batch_summary.md`.)

The selected provider is configured with `CLINICAL_PROVIDER` or `AGENT_PROVIDER` (`ollama`, `openrouter`, or `requesty`). Set its model (`CLINICAL_MODEL`, `AGENT_MODEL`, or provider-specific model variable) and, for hosted providers, its API key in `.env`. Configure provider URLs with `OLLAMA_BASE_URL`, `OPENROUTER_BASE_URL`, and `REQUESTY_BASE_URL`; shared `AGENT_TEMPERATURE` and `AGENT_TIMEOUT` also apply. The existing `.env.example` contains the complete setting list. Results are append-only JSONL, one result per line, and each invocation's rows share a generated `run_id` for easy filtering.

Structure: `evaluate/clinical_agents.py` owns the clinical destination registry and task schema validation; runtime loads the same schema without enforcing the canonical benchmark's 4-per-destination balance. `evaluate/clinical_benchmark.py` owns immutable resolved `ClinicalRunConfig`, task selection, provider dispatch, latency/error capture, and JSONL writes. `evaluate/run_clinical.py` owns CLI parsing and terminal presentation, consuming one logged result at a time from the engine. Data flows CLI/environment → resolved config → schema-valid tasks/prompts → existing provider API → appended JSONL result → CLI output. `evaluate/evaluate.py` remains the owner of `.env` loading and provider transport. `evaluate/batch_eval.py` owns multi-model sequential batching with fsynced per-row telemetry, and `evaluate/plot_batch_comparison.py` renders the two-model comparison figure.

Check destination wiring with:
`python evaluate/clinical_agents.py --validate`

The generic evaluator remains available as before: `python evaluate/evaluate.py --all`.

Clinical-specific task and log paths are included in `.env.example`; with the shared Ollama defaults in that template, the clinical runner uses `OLLAMA_MODEL` unless you uncomment the clinical-only provider/model overrides.

License ->
MIT License - Code assist and draft structure used AI tools; experiment design, task selection, evaluation logic, and interpretation by Aarav Banshiwala.

Contact ->
If you want to learn or give feedback: aaravbanshiwala@gmail.com
