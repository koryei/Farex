# Two-Model Clinical Batch Summary

**Date:** 2026-10-03 (UTC) · **Provider:** Ollama (localhost) · **Temperature:** 0.0 · **Timeout:** 120 s
**Design:** 16-query clinical routing matrix × 2 local models, processed strictly sequentially (one request in flight; all tasks for a model complete before the next starts) to bound local memory pressure.
**Data:** `results/evaluation_run.jsonl` (append-only, one JSON object per cell, fsynced per row)

| Run | Model | Run ID |
|---|---|---|
| 1 | llama3:8b | `20261003T031440.255333Z_55000661` |
| 2 | mistral:7b | `20261003T031555.873287Z_d89b98de` |

## Headline results

| Model | Accuracy | Avg latency | Latency range | Model errors |
|---|---|---|---|---|
| llama3:8b | **12/16 (75.0%)** | 4.14 s | 0.31 – 11.07 s | 0 |
| mistral:7b | **12/16 (75.0%)** | 4.40 s | 0.85 – 8.77 s | 0 |

## Accuracy and latency by domain

| Domain | llama3:8b | mistral:7b | llama avg s | mistral avg s |
|---|---|---|---|---|
| Symptom triage | 4/4 (100%) | 4/4 (100%) | 2.85 | 2.41 |
| Drug interaction check | 4/4 (100%) | 4/4 (100%) | 5.05 | 3.99 |
| Medical literature search | 4/4 (100%) | 4/4 (100%) | 7.48 | 6.88 |
| Clinical math / code | **0/4 (0%)** | **0/4 (0%)** | 1.20 | 4.31 |

## Findings

1. **The entire accuracy gap lives in one domain.** Both models are perfect on the 12 clinical prose tasks (triage, interactions, literature) and fail all 4 `general_code_math` tasks. Overall score is an identical 75% tie; the per-domain split, not the aggregate, is the informative result for the routing study.
2. **The reference hallucination reproduced live.** On `general_code_math_1` (Cockcroft–Gault, expected `68.9`) llama3:8b printed `0.9` — the exact calculation error the Stage 4 `ClinicalEvaluator` self-test was built to catch. mistral:7b printed `108.8` plus unwanted prose.
3. **Unit-conversion near-miss.** On `general_code_math_2` (154 lb → kg, expected `69.9`) mistral answered `70.3` and llama answered `68.6` — both outside the ±0.01 numeric tolerance.
4. **Scoring strictness matters.** Two of llama's four failures are functionally plausible Python (`filter_lab_values` for the abnormal-labs task; a `datetime`-based age function) that fail the alpha-normalized AST structural match because function names and structure differ from the reference. mistral additionally wraps code in prose + fences, which the structural matcher rejects. The 0% math/code score conflates true calculation errors with strict-match failures — worth separating in later analysis.
5. **Terse-but-wrong is cheap.** llama's math cells averaged 1.20 s (bare, wrong answers); mistral's averaged 4.31 s (verbose wrong answers), and mistral's estimated wasted tokens were correspondingly higher (610 vs 459) at simulated baseline rates ($0.00020 vs $0.00011; simulated, not billed).

## Estimated telemetry (simulated baseline: $0.15 / $0.60 per 1M in/out tokens, 4 chars ≈ 1 token)

| Model | Est. input tok | Est. output tok | Est. total cost | Est. wasted tok | Est. wasted cost |
|---|---|---|---|---|---|
| llama3:8b | 1,577 | 1,633 | $0.00122 | 459 | $0.00011 |
| mistral:7b | 1,577 | 1,598 | $0.00120 | 610 | $0.00020 |

## Implication for the routing conditions

Local 7–8B models are reliable clinical prose routers/answerers but unreliable calculators. This supports the escalation hypothesis in `routing_conditions.txt`: a local router can keep prose tasks local while escalating (or tool-checking) math/code tasks, capturing most of the quality of oracle routing at a fraction of frontier-model cost.

## Reproduce

```bash
python evaluate/batch_eval.py --models llama3:8b mistral:7b   # real run
python evaluate/batch_eval.py --dry-run                        # telemetry wiring check, no model calls
```
