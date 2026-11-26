# eviction-vs-compression

**When do KV-cache eviction and prompt-compression methods actually differ?**

Long agent sessions fill the model's context and its KV cache. Two families of methods cut that cost:
- **Prompt compression** shortens the text before the model reads it (e.g. LLMLingua-2).
- **KV-cache eviction** drops stored keys and values while the model runs (e.g. SnapKV, random eviction).

On standard benchmarks the best methods already match the full model at small budgets, so the benchmarks can't tell them apart. This repo builds tasks and evaluations that can.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Reusable code: task generators and scoring |
| `scripts/` | Entry points: `run_pilot.py` (run methods x budgets), `analyze.py` (CIs, paired tests, plots) |
| `experiments/` | One folder per experiment: plan, how to run it, results summary and plots |
| `docs/` | Meeting notes and AI-assistant logs |
| `CLAUDE.md` | Shared context for AI assistants |

## Quick start

A GPU is recommended. A free Colab T4 works.

```bash
pip install -r requirements.txt
python scripts/run_pilot.py --n 20          # ~35 min on a T4
python scripts/analyze.py --inp results.jsonl
```

## Experiments

| Date | Experiment | Status |
| --- | --- | --- |
| 2026-11-25 | [Pilot: when do methods differ?](experiments/2026-11-25_pilot/) | Done (n=20). Caveats to fix before scaling up |

## Contributing

Work on a branch (`yourname/short-description`), open a pull request, and get one teammate's review before merging to `main`.

## Licence

To be decided (MIT or Apache-2.0), pending confirmation from our mentor.
