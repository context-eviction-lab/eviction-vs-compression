"""Run every method x budget x task and save one line per answer to results.jsonl.

Examples
    python run_pilot.py --n 20                      # bug-catching run
    python run_pilot.py --n 100                     # full pilot
    python run_pilot.py --n 100 --tasks single_fact --entropy-probe

The script is resumable: rerun the same command and finished settings are skipped.
"""
import argparse
import json
import math
import os
import re
import time

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tasks import SCORERS, make_gsm8k, make_single_fact

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LINGUA_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["method"], r["keep"]))
    return done


def make_press(method, keep):
    """kvpress uses compression_ratio = fraction REMOVED, so it is 1 - keep."""
    from kvpress import RandomPress, SnapKVPress

    ratio = 1.0 - keep
    if method == "random":
        return RandomPress(compression_ratio=ratio, seed=0)
    if method == "snapkv":
        return SnapKVPress(compression_ratio=ratio)
    raise ValueError(method)


@torch.no_grad()
def answer_surprise(model, tok, device, context, question, answer):
    """Mean -log p (nats) of the gold answer tokens, teacher-forced. Used for the entropy probe."""
    prompt = tok.apply_chat_template([{"role": "user", "content": context + "\n\n" + question}],
                                     add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    a_ids = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([p_ids, a_ids], dim=1).to(device)
    logits = model(ids).logits[0, p_ids.shape[1] - 1:-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    tgt = a_ids[0].to(device)
    return float(-logp[torch.arange(len(tgt)), tgt].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="questions per task")
    ap.add_argument("--tasks", nargs="+", default=["single_fact", "gsm8k"])
    ap.add_argument("--methods", nargs="+", default=["full", "random", "snapkv", "llmlingua2"])
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--entropy-probe", action="store_true",
                    help="also record answer surprise with and without the fact (single_fact only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import kvpress  # noqa: F401  (importing it registers the "kv-press-text-generation" pipeline)
    from transformers import pipeline

    device, dtype = pick_device()
    print(f"Device: {device}")
    pipe = pipeline("kv-press-text-generation", model=MODEL, device=device, torch_dtype=dtype)

    compressor = None
    if "llmlingua2" in args.methods:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(model_name=LINGUA_MODEL, use_llmlingua2=True,
                                      device_map=device)

    data = {}
    if "single_fact" in args.tasks:
        data["single_fact"] = make_single_fact(args.n, seed=args.seed)
    if "gsm8k" in args.tasks:
        data["gsm8k"] = make_gsm8k(args.n, seed=args.seed)

    done = load_done(args.out)
    out = open(args.out, "a")

    for task, examples in data.items():
        ctx_tokens = len(pipe.tokenizer(examples[0]["context"]).input_ids)
        print(f"\n== {task}: {len(examples)} questions, ~{ctx_tokens} context tokens ==")
        max_new = 16 if task == "single_fact" else 320

        for method in args.methods:
            keeps = [1.0] if method == "full" else args.keeps
            for keep in keeps:
                t0, correct, count = time.time(), 0, 0
                for ex in examples:
                    if (ex["id"], method, keep) in done:
                        continue
                    torch.manual_seed(args.seed)
                    context, press = ex["context"], None
                    if method in ("random", "snapkv"):
                        press = make_press(method, keep)
                    elif method == "llmlingua2":
                        context = compressor.compress_prompt(
                            ex["context"], rate=keep, force_tokens=["\n", "?", "."]
                        )["compressed_prompt"]

                    output = pipe(context, question=ex["question"], press=press,
                                  max_new_tokens=max_new)["answer"]
                    ok = bool(SCORERS[task](output, ex["answer"]))
                    rec = {"id": ex["id"], "task": task, "method": method, "keep": keep,
                           "correct": ok, "output": output, "answer": ex["answer"]}
                    if "needle_position" in ex:
                        rec["needle_position"] = ex["needle_position"]
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    correct += ok
                    count += 1
                if count:
                    print(f"{method:>10} keep={keep:.2f}: {correct}/{count} correct "
                          f"({time.time() - t0:.0f}s)")

    # Optional entropy probe (Mike's idea): does deleting the fact raise the model's surprise?
    if args.entropy_probe and "single_fact" in data:
        print("\n== entropy probe ==")
        probe_out = open("entropy_probe.jsonl", "a")
        for ex in data["single_fact"]:
            no_fact = re.sub(r" The learning rate used in run \d+ was [\d.]+\.", "", ex["context"])
            s_with = answer_surprise(pipe.model, pipe.tokenizer, device,
                                     ex["context"], ex["question"], ex["answer"])
            s_without = answer_surprise(pipe.model, pipe.tokenizer, device,
                                        no_fact, ex["question"], ex["answer"])
            probe_out.write(json.dumps({"id": ex["id"], "surprise_with_fact": s_with,
                                        "surprise_without_fact": s_without}) + "\n")
            print(f"{ex['id']}: with={s_with:.2f}  without={s_without:.2f} nats "
                  f"(info gained ~ {(s_without - s_with) / math.log(2):.2f} bits)")
        probe_out.close()

    out.close()
    print(f"\nSaved to {args.out}. Next: python analyze.py --inp {args.out}")


if __name__ == "__main__":
    main()
