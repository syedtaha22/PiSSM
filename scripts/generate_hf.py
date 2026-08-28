"""
Loads mamba-1.4b and prints its generated output for a fixed prompt.

Usage:
    python scripts/generate_hf.py
"""

import torch
from transformers import AutoTokenizer, MambaForCausalLM

CHECKPOINT = "state-spaces/mamba-130m-hf"
TOKENIZER = "EleutherAI/gpt-neox-20b"
PROMPT = "hello?"
MAX_NEW_TOKENS = 30

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

model = MambaForCausalLM.from_pretrained(CHECKPOINT)
model.to("cpu")
model.eval()

input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

with torch.no_grad():
    output_ids = model.generate(
        input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Prompt: {PROMPT!r}")
print(f"Output: {output_text!r}")
