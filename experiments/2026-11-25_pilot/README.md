# Pilot, 25 Nov 2026: when do compression methods differ?

**Question:** do eviction and compression methods only separate when a task depends on a fact that appears once?

- Model: Qwen2.5-1.5B-Instruct, Colab T4 GPU, 20 questions per setting
- Methods: full context, random eviction, SnapKV (via `kvpress`), LLMLingua-2
- Budgets: 10%, 25%, 50% of the context kept
- Tasks: a synthetic single-fact task, and GSM8K with 8 worked examples

Files: [`proposal.md`](proposal.md) is the plan. [`results/summary.md`](results/summary.md) has the numbers, plots and caveats.

**Reproduce** (from the repo root):

```bash
pip install -r requirements.txt
python scripts/run_pilot.py --n 20 --out results_20.jsonl
python scripts/analyze.py --inp results_20.jsonl
```

**Headline:** SnapKV matches the full model on GSM8K even at 10% kept, while random eviction and LLMLingua-2 collapse. On the single-fact task every method collapses. See the caveats in the summary before drawing conclusions.
