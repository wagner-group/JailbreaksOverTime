#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --job-name=continuous_selftraining_"$1"_"$2"_"$3"_"$4"_"$5"
#SBATCH --output=/data/julien_piet/llm-attack-detect/logs/%x.log
#SBATCH --error=/data/julien_piet/llm-attack-detect/logs/%x.log
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16

### ARGS
# 1: uncertainty percentage (0-1) -> Those that are certain will be dropped
# 2: startup size (number of samples to use for the first training)
# 3: slice size
# 4: data variant (orig, new, alt)
# 5: seed

# Activate virtual environment
source env/bin/activate

# Run the training script
echo python training_scripts/train_uncertainty.py \
    --slice $3 \
    --startup-size $2 \
    --data_path data_`echo $4`/dataset_variants.json \
    --output_dir outputs/meta-llama/continuous_selftraining_"$1"_"$2"_"$3"_"$4"_"$5" \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points False \
    --pseudo_labels False \
    --self-training True \
    --uncertainty_percentage `echo $1` \
    --keep_model False \
    --seed $5

python training_scripts/train_uncertainty.py \
    --slice $3 \
    --startup-size $2 \
    --data_path data_`echo $4`/dataset_variants.json \
    --output_dir outputs/meta-llama/continuous_selftraining_"$1"_"$2"_"$3"_"$4"_"$5" \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points False \
    --pseudo_labels False \
    --self-training True \
    --uncertainty_percentage `echo $1` \
    --keep_model False \
    --seed $5
EOT
