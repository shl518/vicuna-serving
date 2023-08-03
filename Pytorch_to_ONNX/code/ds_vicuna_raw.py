from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
import deepspeed
from typing import Iterable, Optional
from utils import get_gpu_memory
import torch
from loguru import logger
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import deepspeed
from fastchat.model import get_conversation_template
ORIG_MODEL_LLAMA2 = '/root1/llama2/llama-2-7b-hf'
ORIG_MODEL_VICUNA_v3 = "/root2/models/vicuna/vicuna-7b/vicuna-7b-v1.3"
WORLD_SIZE = 2
model_path = ORIG_MODEL_LLAMA2
device = 'cuda'


def load_origin_model(
    model_path: str,
    device: str = 'cuda',
    num_gpus: int = WORLD_SIZE,
    max_gpu_memory: Optional[str] = None,
):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            # kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {
                    i: max_gpu_memory for i in range(num_gpus)}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    return model, tokenizer


# load model and tokenizer
model, tokenizer = load_origin_model(model_path, device, WORLD_SIZE)
# set config
CONFIG = {"temperature": 0.7, "repetition_penalty": 1.0,
          "top_p": 1, "top_k": -1, "max_new_tokens": 1024}
# tokenize msg and generate prompt
msg = "Hello my name is Philipp."
conv = get_conversation_template(model_path)
conv.append_message(conv.roles[0], msg)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
inputs = tokenizer([prompt])
inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}

print(f"input prompt: \n \n{prompt}")
print(f'Payload sequence length is: {len(tokenizer(prompt)["input_ids"])}')
# use deepspeed wrap

# ds_model = deepspeed.init_inference(
#     model=model,      # Transformers models
#     mp_size=WORLD_SIZE,        # Number of GPU
#     dtype=torch.float16,  # dtype of the weights (fp16)
#     replace_with_kernel_inject=True,  # replace the model with the kernel injector
# )
ds_model = deepspeed.init_inference(
    model=model,
    replace_with_kernel_inject=True,
    mp_size=WORLD_SIZE,
    dtype=torch.float16,
)
print(f"model is loaded on device {ds_model.module.device}")

ds_logits = ds_model.generate(**inputs,
                              do_sample=True if CONFIG['temperature'] > 1e-5 else False,
                              temperature=CONFIG['temperature'],
                              repetition_penalty=CONFIG['repetition_penalty'],
                              max_new_tokens=CONFIG['max_new_tokens'])
ds_logits = ds_logits[0][len(inputs["input_ids"][0]):]
ds_outputs = tokenizer.decode(
    ds_logits, skip_special_tokens=True, spaces_between_special_tokens=False
)
print(
    f"ds prediction: \n \n {ds_outputs}")
