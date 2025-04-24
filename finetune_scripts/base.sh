#!/bin/bash
source env/bin/activate

# Run the training script
python training_scripts/train_base.py \
    --slice $2 \
    --startup-size $1 \
    --raw_data_path data/final_dataset.json \
    --base_output_dir outputs/meta-llama/base_"$1"_"$2"_"$3" \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --use_pseudo_labels False \
    --keep_model True \
    --training_seed $3

