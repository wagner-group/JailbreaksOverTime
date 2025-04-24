#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=base_"$1"_"$2"
#SBATCH --output=/data/julien_piet/llm-attack-detect/logs/%x.log
#SBATCH --error=/data/julien_piet/llm-attack-detect/logs/%x.log

# Activate virtual environment
source env/bin/activate

# Run the training script
python training_scripts/train_base.py \
    --slice 7 \
    --startup-size 4 \
    --raw_data_path data_`echo $1`/dataset_variants.json \
    --base_output_dir outputs/meta-llama/base_`echo $1`_`echo $2` \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --use_pseudo_labels False \
    --keep_model False \
    --training_seed $2

EOT
