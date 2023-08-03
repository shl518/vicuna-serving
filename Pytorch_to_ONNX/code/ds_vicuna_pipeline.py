from transformers import pipeline
import torch
import os
import deepspeed

ORIG_MODEL_LLAMA2 = '/root2/llama2/llama-2-7b-hf'
ORIG_MODEL_VICUNA_v3 = "/root1/models/vicuna/vicuna-7b/vicuna-7b-v1.3"
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))
generator = pipeline("text-generation",
                     model=ORIG_MODEL_LLAMA2, max_length=1024)
# Initialize the DeepSpeed-Inference engine
generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_with_kernel_inject=True)
string = generator("DeepSpeed is")
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)
