import torch


def run_model_with_input_attentions(
    model,
    text,
    tokenizer,
    **kwargs,
):
    """
    Run the model on the given text and return the output and attention weights.

    Args:
        model: The model to run.
        text: The text to run the model on.
        tokenizer: The tokenizer to use.
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        The output of the model inference and the attention weights.
    """
    if isinstance(text, list):
        raise ValueError("Only a single text input is supported.")

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs, return_dict_in_generate=True, output_attentions=True, **kwargs
    )

    internal_attns = torch.stack([att[0] for att in outputs.attentions[0]])
    internal_attns = internal_attns[:, :, 1:, 1:].float()
    internal_attns = internal_attns / internal_attns.sum(axis=-1).unsqueeze(-1)
    L, H, IS, _ = internal_attns.shape

    generation_attns = [
        torch.stack([att[0].float() for att in outputs.attentions[i]])
        for i in range(1, len(outputs.attentions))
    ]
    generation_attns = [
        (
            val[:, :, :, 1:].float()
            / val[:, :, :, 1:].sum(axis=-1).unsqueeze(-1).float()
        ).float()
        for val in generation_attns
    ]

    user_attns = (
        torch.zeros((L, H, 1 + len(generation_attns), IS))
        .to(model.device)
        .float()
    )

    user_attns[:, :, 0, :] = internal_attns[:, :, -1, :]

    for gen_index, gen_attn in enumerate(generation_attns):
        gen_index += 1
        user_attns[:, :, gen_index, :] = gen_attn[:, :, 0, :IS]
        for comp_index in range(gen_index):
            user_attns[:, :, gen_index, :] += user_attns[
                :, :, comp_index, :
            ] * gen_attn[:, :, 0, IS + comp_index].unsqueeze(-1)

        user_attns[:, :, gen_index, :] /= (
            user_attns[:, :, gen_index, :].sum(axis=-1).unsqueeze(-1)
        )

    attns = user_attns  # Shape is (L, H, Generated tokens, Prompt tokens)

    _, _, GT, IS = attns.shape

    heatmap = (attns.reshape(-1, GT, IS).sum(axis=0) / (H * L)).mean(axis=0)

    weights = sorted(
        zip(
            list(heatmap.cpu().float().numpy()),
            list(tokenizer.batch_decode(inputs["input_ids"][0, 1:])),
            range(input_length - 1),
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    filtered_weights = [
        w
        for w in weights
        if w[2]
        not in {
            0,
            1,
            2,
            input_length - 2,
            input_length - 3,
            input_length - 4,
            input_length - 5,
        }
    ]

    return (
        outputs,
        attns,
        heatmap,
        inputs["input_ids"][0, 1:],
        weights,
        filtered_weights,
    )
