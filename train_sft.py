"""
Local SFT Fine-Tuning Script -- Telkomsel Product Catalog QA
==============================================================
Fine-tunes a small model (default: Qwen2.5-1.5B-Instruct, 4-bit QLoRA) on
your locally-generated JSONL dataset (instruction/output pairs).

Tuned for very tight VRAM (e.g. GTX 1650, 4GB). Everything runs on your
machine -- the only network activity is the ONE-TIME download of the base
model weights from Hugging Face the first time you run this. Your product
data and the JSONL file never leave your machine or get sent anywhere.

Usage:
    python train_sft.py --data telkomsel_sft_id_only.jsonl --model_size 1.5b

Requirements (see setup steps in the chat message):
    pip install unsloth bitsandbytes datasets trl
"""

import argparse
import json
from datasets import Dataset

def load_jsonl_as_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def to_chat_format(example, tokenizer):
    """Convert {instruction, output} into the model's chat template string."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSONL training dataset (instruction/output pairs)")
    parser.add_argument("--test_data", default=None,
                         help="Optional path to a held-out JSONL test set (e.g. *_test.jsonl from "
                              "build_sft_dataset.py --test_split). If given, eval loss is computed "
                              "and logged at the end of every epoch.")
    parser.add_argument("--model_size", default="1.5b", choices=["0.5b", "1.5b"],
                         help="Fallback to 0.5b if 1.5b runs out of memory on your GPU")
    parser.add_argument("--output_dir", default="telkomsel_qa_lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Use only the first N rows of the dataset -- for a quick smoke test "
                              "to measure real it/s before committing to a full run.")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer, SFTConfig

    model_name = {
        "0.5b": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "1.5b": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    }[args.model_size]

    print(f"Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # Attach small LoRA adapters -- only these get trained, base weights stay frozen.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # critical for fitting in 4GB
        random_state=42,
    )

    print(f"Loading dataset: {args.data}")
    raw_dataset = load_jsonl_as_dataset(args.data)
    if args.max_samples is not None:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))
        print(f"  Using a slice of {len(raw_dataset)} examples (--max_samples={args.max_samples})")
    else:
        print(f"  {len(raw_dataset)} examples")

    dataset = raw_dataset.map(
        lambda ex: to_chat_format(ex, tokenizer),
        remove_columns=raw_dataset.column_names,
    )

    eval_dataset = None
    if args.test_data:
        print(f"Loading held-out test set: {args.test_data}")
        raw_test = load_jsonl_as_dataset(args.test_data)
        print(f"  {len(raw_test)} held-out examples (products never seen in --data)")
        eval_dataset = raw_test.map(
            lambda ex: to_chat_format(ex, tokenizer),
            remove_columns=raw_test.column_names,
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=10,
            logging_steps=5,
            optim="adamw_8bit",       # low-memory optimizer
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            output_dir=args.output_dir,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            report_to="none",
        ),
    )

    steps_per_epoch = -(-len(dataset) // (args.batch_size * args.grad_accum))  # ceil division
    total_steps = steps_per_epoch * args.epochs
    print(f"Dataset size: {len(dataset)} | Steps/epoch: {steps_per_epoch} | "
          f"Epochs: {args.epochs} | Total steps: {total_steps}")
    print("Watch the it/s in the first few steps below, then multiply "
          "(seconds/it) x total_steps to get a real time estimate.\n")

    print("Starting training...")
    trainer.train()

    # Save the full per-step/per-epoch loss history (train loss logs every
    # `logging_steps`, eval loss logs every epoch) -- this is what you want
    # for a loss-per-epoch chart/table in a report.
    log_history = trainer.state.log_history
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    train_losses = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e]
    eval_losses = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    print(f"\nSaved full loss history to {args.output_dir}/training_history.json")
    if eval_losses:
        print("\nEval loss per epoch (held-out test products):")
        for epoch, loss in eval_losses:
            print(f"  epoch {epoch:.1f}: eval_loss = {loss:.4f}")
    if train_losses:
        print(f"\nTrain loss logged at {len(train_losses)} points "
              f"(every {args.__dict__.get('logging_steps', 5)} steps) -- see training_history.json for the full curve.")

    if eval_dataset is not None:
        final_metrics = trainer.evaluate()
        print("\nFinal eval metrics (loss on held-out test products):")
        for k, v in final_metrics.items():
            print(f"  {k}: {v}")
        with open(f"{args.output_dir}/eval_loss_report.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Saved to {args.output_dir}/eval_loss_report.json")

    print(f"Saving LoRA adapter to {args.output_dir}/final_adapter")
    model.save_pretrained(f"{args.output_dir}/final_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/final_adapter")

    print("\nDone. To chat with your fine-tuned model, run:")
    print(f"  python infer.py --adapter {args.output_dir}/final_adapter --model_size {args.model_size}")
    if eval_dataset is not None:
        print("\nFor accuracy/F1/exact-match metrics (not just loss), run:")
        print(f"  python evaluate.py --adapter {args.output_dir}/final_adapter "
              f"--model_size {args.model_size} --test_data {args.test_data}")


if __name__ == "__main__":
    main()