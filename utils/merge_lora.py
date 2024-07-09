"""
Usage:
python merge_lora.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH]
"""
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_lora(base_model_name, lora_path):

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = peft_model.merge_and_unload()
    target_model_path = lora_path.replace("checkpoint", "full")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/mimer/NOBACKUP/groups/bloom/shenghui/llms/Llama-2-7b-hf")
    parser.add_argument("--lora_path", type=str, default="/mimer/NOBACKUP/groups/bloom/shenghui/OpenFedLLM/output/alpaca-gpt4_20000_fedavg_c20s2_i10_b16a1_l512_r32a64_20240710000916/checkpoint-10")

    args = parser.parse_args()

    merge_lora(args.base_model_path, args.lora_path)