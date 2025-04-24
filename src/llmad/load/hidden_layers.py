import gc
import hashlib
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .load import load_dataset, load_tokenizer_and_model, run_model

# Load toxic chat dataset
tc_dataset = load_dataset("lmsys-toxic")


def get_hidden_states(
    prompts,
    model_name="llama2base",
    cache_path=".cache/hidden_states",
    device="cuda:0",
):
    # Load from cache
    hidden_states = [None for _ in prompts]
    if cache_path is not None:
        cache_path = os.path.join(cache_path, model_name)
        index_path = os.path.join(cache_path, "index.json")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        else:
            with open(index_path, "r", encoding="utf-8") as f:
                cache_index = json.load(f)
            for prompt_idx, prompt in enumerate(prompts):
                prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
                if prompt_hash in cache_index:
                    hidden_states_file = cache_index[prompt_hash]
                    hidden_states[prompt_idx] = np.load(hidden_states_file)

        if all(hs is not None for hs in hidden_states):
            return hidden_states

    # Filter out prompts that are already cached
    orig_indices = [i for i in range(len(prompts)) if hidden_states[i] is None]
    prompts = [prompts[i] for i in orig_indices]

    # Load model
    tokenizer, model = load_tokenizer_and_model(
        model_name,
        backend="transformers",
        devices=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer.pad_token = tokenizer.eos_token
    batch_size = 2
    # layer = 28

    batches = [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]
    i = 0
    for batch in tqdm(batches, desc="Processing batches"):
        # Get last hidden states
        res, seq = run_model(
            model,
            batch,
            backend="transformers",
            tokenizer=tokenizer,
            guard=False,
            return_input=True,
            tokenizer_params={
                "padding": True,
                "truncation": True,
                "max_length": 2048,
            },
            output_hidden_states=True,
            max_new_tokens=1,
            return_dict_in_generate=True,
        )
        lengths = seq["input_ids"].ne(tokenizer.pad_token_id).sum(-1)
        if len(res.hidden_states) == 1:
            hs = res.hidden_states[0]
        else:
            hs = res.hidden_states
        local_hidden_states = [
            hs[layer][range(len(lengths)), lengths - 1]
            .float()
            .cpu()
            .detach()
            .numpy()
            for layer in range(0, len(hs), 2)
        ]
        local_hidden_states = np.stack(
            local_hidden_states, axis=1
        )  # Stacking matrices of size B x H to B x L x H
        B, _, _ = local_hidden_states.shape
        for batch_index in range(B):
            hidden_states[orig_indices[i]] = local_hidden_states[batch_index]
            prompt_hash = hashlib.md5(prompts[i].encode("utf-8")).hexdigest()
            filename = os.path.join(cache_path, f"{prompt_hash}.npy")
            cache_index[prompt_hash] = filename
            np.save(filename, local_hidden_states[batch_index])
            i += 1
        del res
        gc.collect()

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(cache_index, f)

    return hidden_states
