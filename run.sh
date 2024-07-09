# !/bin/bash
# SBATCH -A NAISS2023-22-1181 -p alvis
# SBATCH -N 1 --gpus-per-node=A100fat:1
# SBATCH --job-name=ofl
# SBATCH --tasks-per-node=1
# SBATCH --time=01:50:00

fingpt_dir=evaluation/FinGPT
export PYTHONPATH=$PYTHONPATH:$fingpt_dir

CUDA_VISIBLE_DEVICES=0 python main_sft.py \
 --model_name_or_path "/mimer/NOBACKUP/groups/bloom/shenghui/llms/Llama-2-7b-hf" \
 --dataset_name "vicgalle/alpaca-gpt4" \
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients 20 \
 --sample_clients 2 \
 --max_steps 10 \
 --num_rounds 50 \
 --batch_size 16 \
 --gradient_accumulation_steps 1 \
 --save_model_freq 10 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --load_in_8bit \
 --output_dir "./output" \
 --template "alpaca" \