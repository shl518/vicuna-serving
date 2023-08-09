from transformers import pipeline
import torch
import os
import deepspeed

ORIG_MODEL_LLAMA2_7 = "/root1/llama2/llama-2-7b-chat-hf"
ORIG_MODEL_VICUNA7_v3 = "/root2/models/vicuna/vicuna-7b/vicuna-7b-v1.3"
ORIG_MODEL_VICUNA13_v3 = "/root2/models/vicuna/vicuna-13b/vicuna-13b-v1.3"
ORIG_MODEL_LLAMA2_13 = "/root1/llama2/llama-2-13b-chat-hf"
ORIG_MODEL_YuLan_13b = "/root1/YuLan-Chat-2-13b-fp16"
ORIG_MODEL_VICUNA33_v3 = "/root2/models/vicuna/vicuna-33b/vicuna-33b-v1.3"
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))
pipe = pipeline("text-generation",
                model=ORIG_MODEL_LLAMA2_7, device=local_rank)


pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float16,
)

pipe.device = torch.device(f'cuda:{local_rank}')
# Initialize the DeepSpeed-Inference engine
# generator.model = deepspeed.init_inference(generator.model,
#                                            mp_size=world_size,
#                                            dtype=torch.float,
#                                            replace_with_kernel_inject=True)
string = pipe("DeepSpeed is")
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)
