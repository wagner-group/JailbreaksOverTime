#!/bin/bash
### ARGS
# 1: label source (real, pseudo)
# 2: uncertainty percentage (0-1)
# 3: startup size
# 4: slice
# 5: seed

# Activate virtual environment
source env/bin/activate

python training_scripts/train_uncertainty.py \
    --slice $4 \
    --startup-size $3 \
    --data_path data/final_dataset_output.json \
    --output_dir outputs/meta-llama/continuous_`echo $1`_`echo $2`_`echo $3`_`echo $4`_"$5" \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points True \
    --pseudo_labels $([ "$1" = "pseudo" ] && echo "True" || echo "False") \
    --uncertainty_percentage `echo $2` \
    --keep_model True \
    --seed $5
