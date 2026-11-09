"""
Inference script -- chat with your fine-tuned Telkomsel product QA model.

Usage:
    python infer.py --adapter telkomsel_qa_lora/final_adapter --model_size 1.5b
"""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to the saved LoRA adapter")
    parser.add_argument("--model_size", default="1.5b", choices=["0.5b", "1.5b"])
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    model_name = {
        "0.5b": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "1.5b": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    }[args.model_size]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=384,
        load_in_4bit=True,
    )
    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)

    print("Fine-tuned model loaded. Type a question (or 'quit' to exit).\n")
    while True:
        q = input("Tanya: ")
        if q.strip().lower() in ("quit", "exit"):
            break
        messages = [{"role": "user", "content": q}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        output = model.generate(inputs, max_new_tokens=128, temperature=0.3, do_sample=True)
        response = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"Jawab: {response}\n")


if __name__ == "__main__":
    main()
