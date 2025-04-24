#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --job-name=continuous_"$1"_"$2"_"$3"_"$4"_drop
#SBATCH --output=/data/julien_piet/llm-attack-detect/logs/%x.log
#SBATCH --error=/data/julien_piet/llm-attack-detect/logs/%x.log
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16

### ARGS
# 1: label source (real, pseudo)
# 2: uncertainty percentage (0-1)
# 3: data variant (orig, new, alt)
# 4: seed

# Activate virtual environment
source env/bin/activate

# Run the training script
echo python training_scripts/train_uncertainty.py \
    --slice 7 \
    --startup-size 4 \
    --data_path data_`echo $3`/dataset_variants.json \
    --output_dir outputs/meta-llama/continuous_`echo $1`_`echo $2`_`echo $3`_`echo $4`_drop \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points True \
    --pseudo_labels $([ "$1" = "pseudo" ] && echo "True" || echo "False") \
    --uncertainty_percentage `echo $2` \
    --keep_model False \
    --drop_certain_negatives True \
    --seed $4

python training_scripts/train_uncertainty.py \
    --slice 7 \
    --startup-size 4 \
    --data_path data_`echo $3`/dataset_variants.json \
    --output_dir outputs/meta-llama/continuous_`echo $1`_`echo $2`_`echo $3`_`echo $4`_drop \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points True \
    --pseudo_labels $([ "$1" = "pseudo" ] && echo "True" || echo "False") \
    --uncertainty_percentage `echo $2` \
    --keep_model False \
    --drop_certain_negatives True \
    --seed $4
EOT
