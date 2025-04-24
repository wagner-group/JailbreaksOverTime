# JailbreaksOverTime

### Setup 
Create a python3.12 virtual environment, install the requirements, export your OpenAI Key, and you're good to go.

### Running the active monitor
`llmad --datapath data/jailbreaksovertime.json --model_path mistralai/Mistral-7B-Instruct-v0.3 --model_name mistral`

Output is saved with data at `data/jailbreaksovertime_outputs.json`

### Training the continuous detector
Requires 90GB+ of GPU VRAM to train. Use the `train_uncertainty.py` script as follows:

run ```bash finetune_scripts/self_training 1 4 7 1``` to replicate results from paper (Llama3-3B-Chat base model, 1 month initial training, retrain every week)

### Dataset
The JailbreaksOverTime data is contained in data/jailbreaksovertime.json



