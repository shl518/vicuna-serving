import argparse
import gc
from functools import partial
from typing import Any, List, Tuple, Union, Callable
import torch.distributed as dist
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import deepspeed
import time
ORIG_MODEL_LLAMA2_7 = "/root1/llama2/llama-2-7b-chat-hf"
ORIG_MODEL_VICUNA7_v3 = "/root2/models/vicuna/vicuna-7b/vicuna-7b-v1.3"
ORIG_MODEL_VICUNA13_v3 = "/root2/models/vicuna/vicuna-13b/vicuna-13b-v1.3"
ORIG_MODEL_LLAMA2_13 = "/root1/llama2/llama-2-13b-chat-hf"

# settings
WORLD_SIZE = 2
benchmark_cycles = 10
model_path = ORIG_MODEL_LLAMA2_13
device = 'cuda'
ds = True
prompt = "Hello my name is Philipp."
CONFIG = {"temperature": 0.7, "repetition_penalty": 1.0,
          "top_p": 1, "top_k": -1, "max_new_tokens": 1024}


def run_and_log_time(execs: Union[List[partial], partial]) -> Tuple[Union[List[Any], Any], float]:
    # runs a function / list of functions and times them
    start_time = time.time()

    if type(execs) == list:
        results = []
        for f in execs:
            results.append(f())
    else:
        results = execs()

    time_elapsed = time.time() - start_time
    return results, time_elapsed


def run_rank_n(func: Callable, rank: int = 0, barrier: bool = False) -> None:
    # wrapper function for the rank to execute on
    def func_rank_n(*args, **kwargs):
        output = func(*args, **kwargs)
        if barrier:
            dist.barrier()
        return output

    # a dummy method that doesn't do anything
    def func_rank_other(*args, **kwargs):
        if barrier:
            dist.barrier()

    if dist.is_initialized():
        if dist.get_rank() == rank:
            return func_rank_n
        return func_rank_other
    else:
        return func


@run_rank_n
def print_rank_0(*args, **kwargs) -> None:
    print(*args, **kwargs)


def load_model(
    model_path: str,
    ds: bool = True
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    if ds:
        model = deepspeed.init_inference(
            model=model,
            replace_with_kernel_inject=True,
            replace_method="auto",
            mp_size=WORLD_SIZE,
            dtype=torch.float16,
        )
    else:
        model.to(device)
    return model


def benchmark_generation(model, tokenizer, echo_len,  cycles: int = 5):
    # run benchmarks for number of cycles
    total_new_tokens_generated = 0
    for _ in range(cycles):
        inputs = tokenizer([prompt])
        if ds:
            inputs = {k: torch.tensor(v).to(model.module.device)
                      for k, v in inputs.items()}
        else:
            inputs = {k: torch.tensor(v).to(device)
                      for k, v in inputs.items()}
        logits = model.generate(**inputs,
                                do_sample=True if CONFIG['temperature'] > 1e-5 else False,
                                temperature=CONFIG['temperature'],
                                repetition_penalty=CONFIG['repetition_penalty'],
                                max_new_tokens=CONFIG['max_new_tokens'])
        logits = logits[0][echo_len:]
        total_new_tokens_generated += len(logits)
    return total_new_tokens_generated


def get_benchmark_results(
    benchmark_time: float, initialization_time: float, total_new_tokens_generated: int, cycles: int
) -> str:
    throughput = total_new_tokens_generated / benchmark_time
    latency = benchmark_time / cycles
    return f"""
*** Performance stats:
Throughput (including tokenization) = {throughput:.2f} tokens/sec
Throughput (including tokenization) = {1000 / throughput:.2f} msecs/token
Model loading time = {initialization_time:.2f} secs
Total tokens generated = {total_new_tokens_generated}
Latency = {latency:.2f} secs
Model loading time + generation time per batch = {initialization_time + latency:.2f} secs
"""


def benchmark_end_to_end() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False
    )
    model, initialization_time = run_and_log_time(
        partial(load_model, model_path=model_path, ds=ds))

    inputs = tokenizer([prompt])
    print_rank_0(f"input prompt: \n \n{prompt}")
    print_rank_0(
        f'Payload sequence length is: {len(tokenizer(prompt)["input_ids"])}')
    if ds:
        inputs = {k: torch.tensor(v).to(model.module.device)
                  for k, v in inputs.items()}
    else:
        inputs = {k: torch.tensor(v).to(device)
                  for k, v in inputs.items()}

    # warmup is a must if measuring speed as it's when all the optimizations are performed
    # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
    logits = model.generate(**inputs,
                            do_sample=True if CONFIG['temperature'] > 1e-5 else False,
                            temperature=CONFIG['temperature'],
                            repetition_penalty=CONFIG['repetition_penalty'],
                            max_new_tokens=CONFIG['max_new_tokens'])

    logits = logits[0][len(inputs["input_ids"][0]):]
    outputs = tokenizer.decode(
        logits, skip_special_tokens=True, spaces_between_special_tokens=False)
    print_rank_0(f"output prediction: \n \n{outputs}")
    print_rank_0(
        f'output sequence length(new tokens) is: {len(logits)}')

    if benchmark_cycles > 0:
        print_rank_0("*** Running benchmark")

        torch.cuda.empty_cache()
        gc.collect()

        # warm up
        model.generate(**inputs)
        torch.cuda.synchronize()

        # benchmark
        total_new_tokens_generated, benchmark_time = run_and_log_time(
            partial(benchmark_generation, model=model, tokenizer=tokenizer,
                    echo_len=len(inputs["input_ids"][0]), cycles=benchmark_cycles)
        )

        print_rank_0(
            get_benchmark_results(
                benchmark_time, initialization_time, total_new_tokens_generated, benchmark_cycles
            )
        )


def main() -> None:
    benchmark_end_to_end()


if __name__ == "__main__":
    main()
