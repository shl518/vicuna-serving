# !/bin/bash
python /home/shizy/vicuna-serving/Pytorch_to_ONNX/code/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /home/nfs_data_02/llama2/llama-2-7b-chat \
    --model_size 7B \
    --output_dir /home/nfs_data_02/llama2/llama-2-7b-chat
