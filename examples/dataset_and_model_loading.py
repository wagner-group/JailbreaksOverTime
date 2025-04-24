from llmad import load_dataset, load_tokenizer_and_model, run_model

# Load model
tokenizer, model = load_tokenizer_and_model("llamaguard")

# Load dataset
dataset = load_dataset("lmsys-toxic")

# Generate prompts
prompts = [
    f"User: {line['user_input']}\n Agent: {line['model_output']}"
    for line in dataset["train"]
][:5]

toxicity = [line["toxicity"] for line in dataset["test"]][:5]

# Run LlamaGuard
outputs, inputs = run_model(
    model,
    prompts,
    tokenizer=tokenizer,
    role="User",
    backend="vllm",
    return_input=True,
)

output_classification = ["unsafe" in o.outputs[0].text.lower() for o in outputs]

fn = sum(
    [tox and not out for tox, out in zip(toxicity, output_classification)]
) / sum(toxicity)
fp = sum(
    [not tox and out for tox, out in zip(toxicity, output_classification)]
) / sum([not t for t in toxicity])

print(f"False negative rate: {fn:.2%}, False positive rate: {fp:.2%}")
