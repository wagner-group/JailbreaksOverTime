import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer
from vllm import LLM

from llmad import run_model as run_llm

system_messages = {
    "llama-2": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make "
    "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't "
    "know the answer to a question, please don't share false information.",
    "vicuna": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed,"
    " and polite answers to the user's questions.",
    "llama": "You are a helpful assistant",
}


def run(prompts, model_name, model_path):
    # Load model
    model = LLM(
        "/scratch/public_models/huggingface/" + model_path,
        gpu_memory_utilization=0.85,
        max_model_len=2304,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Apply Vicuna format
    prompts = [format_prompt(prompt, model_name) for prompt in prompts]

    # Run model
    outputs = run_llm(
        model,
        prompts,
        tokenizer=tokenizer,
        backend="vllm",
        guard=False,
        max_tokens=256,
        if_format_prompt=False,
        logprobs=1,
    )

    # Extract text
    outputs = [
        output.outputs[0].text.strip() if output and output.outputs else None
        for output in outputs
    ]

    # Delete models
    del model
    del tokenizer

    return outputs


def format_prompt(prompt, model_name="vicuna"):
    conv = get_conversation_template(model_name)
    conv.messages = []  # Clear previous messages to avoid contamination
    if model_name in system_messages:
        # Add system message if available
        conv.system_message = system_messages[model_name]
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


# the actual classifier
def run_model(data, model_name, model_path):
    # Extract prompts
    prompts = [d["prompt"] for d in data]

    # Run Vicuna
    outputs = run(prompts, model_name, model_path)

    return data, outputs
