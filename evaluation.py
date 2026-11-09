"""
Evaluation script -- computes accuracy metrics on the held-out test set.

Produces, entirely locally (no API calls):
  - Exact Match (EM):  % of answers that match the reference exactly (normalized)
  - Token-level F1:    overlap between predicted and reference answer tokens
                        (standard SQuAD-style metric, tolerant of partial/reworded matches)
  - Abstention accuracy: for reference answers that are "I don't have that information"
                        style, did the model also abstain rather than hallucinate?
  - Per-example CSV + aggregate JSON summary, ready to paste into a report.

Usage:
    python evaluate.py --adapter telkomsel_qa_lora/final_adapter --model_size 1.5b --test_data telkomsel_sft_50_test.jsonl
"""

import argparse
import json
import re
import csv
import string


def normalize(text):
    """Lowercase, strip punctuation/articles/extra whitespace -- standard QA normalization."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(pred, ref):
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), ref_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))


ABSTENTION_MARKERS = ["tidak ada informasi", "tidak memiliki informasi", "maaf, saya tidak"]

def is_abstention(text):
    t = normalize(text)
    return any(marker in t for marker in ABSTENTION_MARKERS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to the saved LoRA adapter")
    parser.add_argument("--model_size", default="1.5b", choices=["0.5b", "1.5b"])
    parser.add_argument("--test_data", required=True, help="Held-out test JSONL (instruction/output pairs)")
    parser.add_argument("--output_csv", default="eval_results.csv")
    parser.add_argument("--output_summary", default="eval_summary.json",
                         help="Where to save the aggregate summary JSON -- use a distinct "
                              "name per run (e.g. eval_summary_unseen.json vs "
                              "eval_summary_seen_paraphrase.json) so separate evaluate.py "
                              "runs don't overwrite each other.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    model_name = {
        "0.5b": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "1.5b": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    }[args.model_size]

    print(f"Loading base model + adapter: {model_name} + {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=384, load_in_4bit=True,
    )
    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)

    examples = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Evaluating on {len(examples)} held-out examples...\n")

    results = []
    for i, ex in enumerate(examples):
        messages = [{"role": "user", "content": ex["instruction"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        output = model.generate(
            inputs, max_new_tokens=args.max_new_tokens, temperature=0.1, do_sample=False
        )
        pred = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        ref = ex["output"]

        em = exact_match(pred, ref)
        f1 = token_f1(pred, ref)
        ref_is_abstention = is_abstention(ref)
        pred_is_abstention = is_abstention(pred)

        results.append({
            "instruction": ex["instruction"],
            "reference": ref,
            "prediction": pred,
            "exact_match": em,
            "f1": round(f1, 3),
            "ref_is_abstention": ref_is_abstention,
            "pred_is_abstention": pred_is_abstention,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(examples)} done")

    # Aggregate metrics
    n = len(results)
    avg_em = sum(r["exact_match"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n

    abstention_cases = [r for r in results if r["ref_is_abstention"]]
    factual_cases = [r for r in results if not r["ref_is_abstention"]]

    abstention_accuracy = (
        sum(r["pred_is_abstention"] for r in abstention_cases) / len(abstention_cases)
        if abstention_cases else None
    )
    # Hallucination proxy: on factual (non-abstention) questions, how often did
    # the model INCORRECTLY abstain instead of giving the known answer?
    false_abstention_rate = (
        sum(r["pred_is_abstention"] for r in factual_cases) / len(factual_cases)
        if factual_cases else None
    )

    summary = {
        "num_examples": n,
        "exact_match": round(avg_em, 4),
        "token_f1": round(avg_f1, 4),
        "num_abstention_cases": len(abstention_cases),
        "abstention_accuracy": round(abstention_accuracy, 4) if abstention_accuracy is not None else None,
        "num_factual_cases": len(factual_cases),
        "false_abstention_rate_on_factual": round(false_abstention_rate, 4) if false_abstention_rate is not None else None,
    }

    print("\n=== EVALUATION SUMMARY (held-out, unseen products) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-example results written to: {args.output_csv}")

    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {args.output_summary}")


if __name__ == "__main__":
    main()