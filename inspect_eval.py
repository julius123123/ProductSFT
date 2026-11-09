"""
Print N random rows from an evaluate.py output CSV, for manual inspection.

Usage:
    python inspect_eval.py eval_unseen.csv --n 5
    python inspect_eval.py eval_unseen.csv --n 10 --only_wrong   # only show EM=0 rows
"""

import argparse
import csv
import random

parser = argparse.ArgumentParser()
parser.add_argument("csv_path")
parser.add_argument("--n", type=int, default=5)
parser.add_argument("--only_wrong", action="store_true",
                     help="Only sample from rows where exact_match == 0 (the interesting failures)")
parser.add_argument("--only_true", action="store_true",
                     help="Only sample from rows where exact_match == 1")
parser.add_argument("--min_f1", type=float, default=None,
                     help="Only sample from rows where f1 > this value")
parser.add_argument("--max_f1", type=float, default=None,
                     help="Only sample from rows where f1 <= this value")
parser.add_argument("--seed", type=int, default=None, help="Set for reproducible sampling")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

with open(args.csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if args.only_wrong:
    rows = [r for r in rows if float(r["exact_match"]) == 0.0]
    print(f"(Filtered to {len(rows)} rows with exact_match == 0)\n")

if args.only_true:
    rows = [r for r in rows if float(r["exact_match"]) == 1.0]
    print(f"(Filtered to {len(rows)} rows with exact_match == 0)\n")

if args.min_f1 is not None:
    rows = [r for r in rows if float(r["f1"]) > args.min_f1]
    print(f"(Filtered to {len(rows)} rows with f1 > {args.min_f1})\n")

if args.max_f1 is not None:
    rows = [r for r in rows if float(r["f1"]) <= args.max_f1]
    print(f"(Filtered to {len(rows)} rows with f1 <= {args.max_f1})\n")

sample = random.sample(rows, min(args.n, len(rows)))

for i, r in enumerate(sample, 1):
    print(f"--- Example {i} ---")
    print(f"Q:         {r['instruction']}")
    print(f"Reference: {r['reference']}")
    print(f"Predicted: {r['prediction']}")
    print(f"EM: {r['exact_match']}  F1: {r['f1']}  "
          f"ref_abstention: {r['ref_is_abstention']}  pred_abstention: {r['pred_is_abstention']}")
    print()