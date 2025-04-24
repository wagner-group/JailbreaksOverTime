#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --job-name=continuous_pseudo_"$1"_"$2"_"$3"_"$4"_"$5"_"$6"
#SBATCH --output=/data/julien_piet/llm-attack-detect/logs/%x.log
#SBATCH --error=/data/julien_piet/llm-attack-detect/logs/%x.log
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16

### ARGS
# 1: uncertainty percentage (0-1)
# 2: data variant (orig, new, alt)
# 3: toxic threshold high
# 4: benign threshold low
# 5: drop uncertain (True/False)
# 6: seed

# Activate virtual environment
source env/bin/activate

# Run the training script
echo python training_scripts/train_uncertainty.py \
    --slice 7 \
    --startup-size 4 \
    --data_path data_`echo $2`/dataset_variants.json \
    --output_dir outputs/meta-llama/continuous_pseudo_"$1"_"$2"_"$3"_"$4"_"$5"_"$6" \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points True \
    --pseudo_labels True \
    --uncertainty_percentage `echo $1` \
    --keep_model False \
    --jailbreak_threshold $3 \
    --non_jailbreak_threshold $4 \
    --drop_ambiguous $5 \
    --seed $6

python training_scripts/train_uncertainty.py \
    --slice 7 \
    --startup-size 4 \
    --data_path data_`echo $2`/dataset_variants.json \
    --output_dir outputs/meta-llama/continuous_pseudo_"$1"_"$2"_"$3"_"$4"_"$5"_"$6" \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --include_certain_points True \
    --pseudo_labels True \
    --uncertainty_percentage `echo $1` \
    --keep_model False \
    --jailbreak_threshold $3 \
    --non_jailbreak_threshold $4 \
    --drop_ambiguous $5 \
    --seed $6
EOT
