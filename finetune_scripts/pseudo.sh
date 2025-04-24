#!/bin/bash
### ARGS
# 1: startup size
# 2: slice
# 3: toxic threshold high
# 4: benign threshold low
# 5: drop uncertain (True/False)
# 6: uncertainty percentage (0-1)
# 7: seed

# Activate virtual environment
source env/bin/activate

python training_scripts/train_uncertainty.py \
    --slice $1 \
    --startup-size $2 \
    --data_path data/final_dataset_output.json \
    --output_dir outputs/meta-llama/continuous_pseudo_"$1"_"$2"_"$3"_"$4"_"$5"_"$6"_"$7" \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points True \
    --pseudo_labels True \
    --uncertainty_percentage `echo $6` \
    --keep_model True \
    --jailbreak_threshold $3 \
    --non_jailbreak_threshold $4 \
    --drop_ambiguous $5 \
    --seed $7
