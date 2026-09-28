"""Merge a LoRA adapter into the base model and save the full merged weights."""
import sys
# Setting to None makes importlib.util.find_spec return None → deepspeed treated as absent
sys.modules["deepspeed"] = None

import argparse

from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", required=True)
parser.add_argument("--base", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

print(f"Loading base model from {args.base}")
model = AutoModelForImageTextToText.from_pretrained(args.base, torch_dtype="auto", trust_remote_code=True)

print(f"Loading adapter from {args.adapter}")
model = PeftModel.from_pretrained(model, args.adapter)

print("Merging and unloading adapter")
model = model.merge_and_unload()

print(f"Saving merged model to {args.output}")
model.save_pretrained(args.output)

print("Copying processor/tokenizer files")
processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
processor.save_pretrained(args.output)

print("Done.")
