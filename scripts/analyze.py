"""Accuracy vs budget with 95% bootstrap CIs, paired McNemar tests, and one plot per task.

    python analyze.py --inp results.jsonl
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np


def bootstrap_ci(xs, n_boot=1000, seed=0):
    xs = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, size=(n_boot, len(xs)), replace=True).mean(axis=1)
    return xs.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired booleans.
    Returns (A right & B wrong, A wrong & B right, p-value)."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n10, n01, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results.jsonl")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # res[task][(method, keep)][id] = correct
    res = defaultdict(lambda: defaultdict(dict))
    with open(args.inp) as f:
        for line in f:
            r = json.loads(line)
            res[r["task"]][(r["method"], r["keep"])][r["id"]] = r["correct"]

    for task, settings in res.items():
        print(f"\n===== {task} =====")
        print(f"{'method':>11} {'keep':>5} {'n':>4} {'acc':>6}   95% CI")
        rows = []
        for (method, keep), d in sorted(settings.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            acc, lo, hi = bootstrap_ci(list(d.values()))
            rows.append((method, keep, acc, lo, hi))
            print(f"{method:>11} {keep:>5.2f} {len(d):>4} {acc:>6.2f}   [{lo:.2f}, {hi:.2f}]")

        # Paired comparisons at each budget, on the questions both settings answered
        keeps = sorted({k for (_, k) in settings if k < 1.0})
        methods = sorted({m for (m, k) in settings if k < 1.0})
        print("\nPaired McNemar tests (A right & B wrong / A wrong & B right, p):")
        for keep in keeps:
            for i, a in enumerate(methods):
                for b in methods[i + 1:]:
                    da, db = settings.get((a, keep)), settings.get((b, keep))
                    if not da or not db:
                        continue
                    ids = sorted(set(da) & set(db))
                    n10, n01, p = mcnemar([da[x] for x in ids], [db[x] for x in ids])
                    flag = "  *" if p < 0.05 else ""
                    print(f"  keep={keep:.2f}  {a} vs {b}: {n10} / {n01}, p={p:.3f}{flag}")

        if not args.no_plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            full = [r for r in rows if r[0] == "full"]
            if full:
                _, _, acc, lo, hi = full[0]
                ax.axhspan(lo, hi, color="grey", alpha=0.2)
                ax.axhline(acc, color="grey", ls="--", label="full context")
            for m in sorted({r[0] for r in rows} - {"full"}):
                pts = sorted((r for r in rows if r[0] == m), key=lambda r: r[1])
                x = [p[1] * 100 for p in pts]
                y = [p[2] for p in pts]
                err = [[p[2] - p[3] for p in pts], [p[4] - p[2] for p in pts]]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=m)
            ax.set_xlabel("Context kept (%)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{task}: accuracy vs budget (95% bootstrap CI)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"plot_{task}.png", dpi=150)
            print(f"Saved plot_{task}.png")


if __name__ == "__main__":
    main()
