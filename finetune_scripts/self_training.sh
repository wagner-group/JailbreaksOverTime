#!/bin/bash
### ARGS
# 1: uncertainty percentage (0-1) -> Those that are certain will be dropped
# 2: startup size (number of samples to use for the first training)
# 3: slice size
# 4: seed

# Activate virtual environment
source env/bin/activate

python training_scripts/train_uncertainty.py \
    --slice $3 \
    --startup-size $2 \
    --data_path data/final_dataset.json \
    --output_dir outputs/meta-llama/continuous_self_"$1"_"$2"_"$3"_"$4" \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points False \
    --pseudo_labels False \
    --self-training True \
    --uncertainty_percentage `echo $1` \
    --keep_model True \
    --seed $4
