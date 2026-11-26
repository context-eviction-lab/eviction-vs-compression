# Context for AI assistants working in this repo

Project: G-Research student project (brief 2, "Prompt Compression and Meta-Tuning for Agentic Workflows").
Team: Angela, Ervin, Patrick. Mentor: Mike.

Core question: when do KV-cache eviction methods and prompt-compression methods actually differ, and on which tasks?
Standard benchmarks (MATH500, GSM8K, GovReport) are too easy: strong methods already match the full model at small budgets.

Conventions:
- Reusable code goes in `src/`, entry points in `scripts/`, and one folder per experiment in `experiments/YYYY-MM-DD_name/`.
- Commit summaries and plots only. Raw per-answer `.jsonl` files stay out of git.
- Changes to `main` go through pull requests.
- Known environment issue: on Python 3.13, upgrade `fire` (kvpress pins an old version that imports the removed `pipes` module).
- `import kvpress` must run before `transformers.pipeline("kv-press-text-generation")`.
