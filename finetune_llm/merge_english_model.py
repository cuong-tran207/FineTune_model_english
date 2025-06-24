import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import setup_chat_format

def main(args):
    # Load tokenizer (from PEFT model dir)
    tokenizer = AutoTokenizer.from_pretrained(args.peft_model_path)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Format model for chat (if trained as assistant)
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)

    # Load LoRA adapter and merge into base
    peft_model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    merged_model = peft_model.merge_and_unload()

    # Save final model and tokenizer
    merged_model.save_pretrained(args.output_path, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output_path)

    print(f"✅ Merged model saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")

    parser.add_argument("--base_model_path", type=str, default="1TuanPham/T-VisStar-7B-v0.1",
                        help="HuggingFace model hub ID or local path to base model")
    parser.add_argument("--peft_model_path", type=str, default="output/checkpoint-9",
                        help="Path to fine-tuned LoRA checkpoint (e.g., output/checkpoint-9)")
    parser.add_argument("--output_path", type=str, default="merged_model",
                        help="Output directory to save merged model")

    args = parser.parse_args()
    main(args)
