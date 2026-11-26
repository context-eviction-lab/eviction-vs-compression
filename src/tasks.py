"""Task generators for the pilot.

Each example is a dict:
    id        unique string
    task      "single_fact" or "gsm8k"
    context   long text that compression / eviction is allowed to cut
    question  short instruction that is never cut
    answer    gold answer (string)
"""
import random
import re

# ---------------------------------------------------------------------------
# Task 1: single-fact ("needle") task, fully synthetic
# ---------------------------------------------------------------------------
MODELS = ["ResNet-50", "ViT-B/16", "a 2-layer LSTM", "GPT-2 small", "a gradient-boosted tree",
          "a linear factor model", "a temporal CNN", "a small transformer"]
DATASETS = ["the equity returns panel", "CIFAR-10", "the order-book snapshots", "WikiText-103",
            "the FX tick data", "the futures curve dataset", "ImageNet-1k", "the news-sentiment corpus"]
OUTCOMES = ["validation loss plateaued early", "the Sharpe ratio on the hold-out set improved slightly",
            "training diverged after a few thousand steps", "results were within noise of the baseline",
            "overfitting appeared after epoch 12", "the out-of-sample correlation dropped",
            "the run was stopped because of a data bug", "accuracy improved by about one point"]
NOTES = ["Batch size was {bs}.", "Weight decay was set to {wd}.", "Training ran for {ep} epochs.",
         "Dropout was {do}.", "The warm-up lasted {wu} steps.", "Gradient clipping was set to {gc}."]


def _filler_sentence(rng, run_id):
    note = rng.choice(NOTES).format(
        bs=rng.choice([32, 64, 128, 256]), wd=rng.choice([0.0, 0.01, 0.05, 0.1]),
        ep=rng.randint(3, 40), do=rng.choice([0.0, 0.1, 0.2, 0.3]),
        wu=rng.choice([100, 500, 1000, 2000]), gc=rng.choice([0.5, 1.0, 5.0]))
    return (f"Run {run_id} trained {rng.choice(MODELS)} on {rng.choice(DATASETS)}; "
            f"{rng.choice(OUTCOMES)}. {note}")


def make_single_fact(n, n_runs=80, seed=0):
    """About 2,000 tokens of fake experiment notes, with ONE learning rate hidden in it.

    Filler sentences never mention a learning rate, so the fact appears exactly once.
    """
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        run_ids = rng.sample(range(1, 500), n_runs)
        target = run_ids[rng.randrange(n_runs)]
        lr = f"{rng.uniform(0.0001, 0.009):.4f}"
        lines = []
        for r in run_ids:
            line = _filler_sentence(rng, r)
            if r == target:
                line += f" The learning rate used in run {r} was {lr}."
            lines.append(line)
        examples.append({
            "id": f"single_fact_{i}",
            "task": "single_fact",
            "context": "Experiment log:\n" + "\n".join(lines),
            "question": f"What was the learning rate used in run {target}? Reply with the number only.",
            "answer": lr,
            "needle_position": run_ids.index(target) / n_runs,
        })
    return examples


def score_single_fact(output, answer):
    return answer in output


# ---------------------------------------------------------------------------
# Task 2: repetitive task, GSM8K with long worked examples in the prompt
# ---------------------------------------------------------------------------
def _gsm8k_final(answer_text):
    return answer_text.split("####")[-1].strip().replace(",", "")


def make_gsm8k(n, n_shots=8, seed=0):
    from datasets import load_dataset  # imported here so the needle task works offline

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    shots = []
    for ex in train.select(range(n_shots)):
        reasoning, final = ex["answer"].split("####")
        shots.append(f"Question: {ex['question']}\nAnswer: {reasoning.strip()} "
                     f"The answer is {final.strip()}.")
    demo = "\n\n".join(shots)

    rng = random.Random(seed)
    idx = rng.sample(range(len(test)), n)
    examples = []
    for i in idx:
        ex = test[i]
        examples.append({
            "id": f"gsm8k_{i}",
            "task": "gsm8k",
            # Both the worked examples and the problem can be cut.
            "context": f"Worked examples:\n\n{demo}\n\nNow the new problem.\nQuestion: {ex['question']}",
            "question": "Solve the new problem step by step, then end with 'The answer is N.'",
            "answer": _gsm8k_final(ex["answer"]),
        })
    return examples


def score_gsm8k(output, answer):
    m = re.findall(r"The answer is\s*\$?(-?[\d,]*\.?\d+)", output)
    nums = m if m else re.findall(r"-?\d[\d,]*\.?\d*", output)
    if not nums:
        return False
    pred = nums[-1].replace(",", "").rstrip(".")
    try:
        return abs(float(pred) - float(answer)) < 1e-6
    except ValueError:
        return False


SCORERS = {"single_fact": score_single_fact, "gsm8k": score_gsm8k}
