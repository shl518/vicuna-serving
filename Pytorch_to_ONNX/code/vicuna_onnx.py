from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
    pipeline
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import numpy as np
import os
import pdb
from loguru import logger
import onnx
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTModelForCausalLM
import onnxruntime
from utils import get_gpu_memory
from typing import Iterable, Optional
PROMPT_DICT = {
    "prompt_input":
    ("Below is an instruction that describes a task, paired with an input that provides further context. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
     ),
    "prompt_no_input":
    ("Below is an instruction that describes a task. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Response:"),
}
PROMPT = PROMPT_DICT['prompt_no_input']

# PATH
ORIG_MODEL = '/root/models/moss-ft-vicuna/moss-ft-vicuna-7b'
ONNXDIR = '/root/models/moss-ft-vicuna/moss_ft_vicuna_7b_onnx'
ONNXDIR_02opt = '/root/models/moss-ft-vicuna/moss_ft_vicuna_7b_02opt_onnx'


def load_origin_model(
    model_path: str,
    device: str = 'cpu',
    num_gpus: int = 0,
    max_gpu_memory: Optional[str] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
    load_onnx: bool = True,
):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
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

    # load model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path, use_fast=False
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        ONNXDIR, use_fast=False
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, low_cpu_mem_usage=True, **kwargs
    # )

    # session_options = onnxruntime.SessionOptions()
    # session_options.device.id = 0
    # session_options.log_severity_level = 0
    # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # session_options.enable_mem_pattern = True
    # session_options.enable_cpu_mem_arena = True

    # session_options.providers.append('CUDAExecutionProvider')
    # session_options.provider_options['CUDAExecutionProvider'] = cuda_options

    model = ORTModelForCausalLM.from_pretrained(ONNXDIR)
    if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device == "mps":
        model.to(device)
    if debug:
        print(model)
        print(type(model))
    return model, tokenizer


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(
            RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


class ONNX_Vicuna:
    def __init__(self, onnxDir=ONNXDIR, config: dict = {}):
        # some settings
        self.use_fast_tokenizer = False

        if not os.path.exists(onnxDir):
            logger.error('{} not exist'.format(onnxDir))

        assert os.path.isdir(onnxDir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            onnxDir, use_fast=self.use_fast_tokenizer
        )
        print('Tokenizer load success')
        self.onnx_model = ORTModelForCausalLM.from_pretrained(onnxDir)
        print('Onnx Model load success')
        prompt = "hello,what's your name?"
        format_prompt = PROMPT.format_map({'instruction': prompt})
        config = {"temperature": 0.8}
        onnx_gen = pipeline("text-generation", model=self.onnx_model,
                            tokenizer=self.tokenizer, max_length=2048, config=config)
        gen = onnx_gen(format_prompt)
        print(gen)
        print('Gen success')


class Vicuna:
    def __init__(self, modelDir=ORIG_MODEL):
        self.model, self.tokenizer = load_origin_model(
            ORIG_MODEL, 'cpu', 0, debug=True)

    @torch.inference_mode()
    def inference(self, params: dict = {}, context_len=2048, device='cuda'):
        prompt = params["prompt"]
        len_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)
        echo = bool(params.get("echo", True))
        stop_token_ids = params.get("stop_token_ids", None) or []
        stop_token_ids.append(self.tokenizer.eos_token_id)
        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)
        output_ids = list(input_ids)
        if self.model.config.is_encoder_decoder:
            max_src_len = context_len
        else:
            max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        if self.model.config.is_encoder_decoder:
            encoder_output = self.model.encoder(
                input_ids=torch.as_tensor([input_ids], device=device)
            )[0]
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=device,
            )
        past_key_values = out = None

        # start inference
        for i in range(max_new_tokens):
            if i == 0:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(torch.as_tensor(
                        [input_ids], device=device), use_cache=True)
                    logits = out.logits
                past_key_values = out.past_key_values
            else:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=torch.as_tensor([[token]], device=device),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )

                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = out.logits
                past_key_values = out.past_key_values

            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor(
                        [output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(
                    tmp_output_ids, logits[:, -1, :])[0]
            else:
                last_token_logits = logits[0, -1, :]

            if device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                output = self.tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                break

        # finish stream event, which contains finish reason
        if i == max_new_tokens - 1:
            finish_reason = "length"
        elif stopped:
            finish_reason = "stop"
        else:
            finish_reason = None

        return {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }


def main():
    # vicuna_onnx = ONNX_Vicuna()
    prompt = "hello,what's your name?"
    format_prompt = PROMPT.format_map({'instruction': prompt})
    default_params = {
        "prompt": format_prompt,
        "temperature": 0.8,
    }
    vicuna = Vicuna()
    res = vicuna.inference(default_params)
    print(res)


if __name__ == '__main__':
    main()
