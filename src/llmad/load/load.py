import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime

import datasets
import pandas as pd
import torch
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from .format import format_prompt, format_prompt_guard


def load_dataset(path, subset="", orig=False):
    """
    Load the dataset from the given path.

    Args:
        path (str): The path to the dataset. If path is one of ['lmsys-chat', 'openai-moderation', 'lmsys-toxic'], the corresponding public dataset will be loaded from its default location.
        subset (str, optional): The subset of the dataset to load. Defaults to "".
        orig (bool, optional): Whether to load the original dataset for jailbreak_llms. Defaults to False.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    if path == "lmsys-chat":
        path = "/data1/public_datasets/lmsys-chat-1m"
    elif path == "openai-moderation":
        path = "/data1/public_datasets/openai-moderation-api-evaluation"
    elif path == "lmsys-toxic":
        path = "/data1/public_datasets/toxic-chat"
        subset = "toxicchat0124"
    elif path == "jailbreak_llm":
        raw_data = []

        if orig:
            path = "/data1/public_datasets/jailbreak_llms/data/prompts"

            @dataclass
            class Entry:
                platform: str
                source: str
                prompt: str
                jailbreak: bool
                created_at: int
                date: str

                def __repr__(self) -> str:
                    return f"<{self.platform} {self.source} {self.jailbreak} {self.date}>"

            for file in os.listdir(path):
                if file.endswith(".csv"):
                    file_path = os.path.join(path, file)
                    print(file)
                    # jailbreaks = "jailbreak" in file
                    with open(file_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        data = list(reader)
                        entries = []
                        fields = []
                        for i, d in enumerate(data):
                            if i == 0:
                                fields = d
                                jb_index = fields.index("jailbreak")
                                created_at_index = fields.index("created_at")
                                date_index = fields.index("date")
                                platform_index = fields.index("platform")
                                source_index = fields.index("source")
                                prompt_index = fields.index("prompt")
                                continue

                            if isinstance(d[jb_index], str):
                                jb = d[jb_index] == "True"
                            else:
                                jb = bool(d[jb_index])
                            if d[created_at_index].strip() == "":
                                ca = -1
                            else:
                                try:
                                    format_string = "%Y-%m-%dT%H:%M:%S.%fZ"
                                    ca = datetime.strptime(
                                        d[created_at_index], format_string
                                    ).timestamp()
                                except ValueError:
                                    try:
                                        format_string = "%Y-%m-%dT%H:%M:%SZ"
                                        ca = datetime.strptime(
                                            d[created_at_index], format_string
                                        ).timestamp()
                                    except ValueError:
                                        try:
                                            ca = datetime.fromisoformat(
                                                d[created_at_index]
                                            ).timestamp()
                                        except ValueError:
                                            try:
                                                format_string = (
                                                    "%Y-%m-%dT%H:%M:%S.%f+00:00"
                                                )
                                                ca = datetime.strptime(
                                                    d[created_at_index],
                                                    format_string,
                                                ).timestamp()
                                            except:
                                                ca = float(d[created_at_index])
                            entries.append(
                                Entry(
                                    d[platform_index],
                                    d[source_index],
                                    d[prompt_index],
                                    jb,
                                    ca,
                                    d[date_index],
                                )
                            )

                        raw_data.extend(entries)
        else:
            path = "/data1/public_datasets/jailbreak_llm_mod.json"
            with open(path, "r", encoding="utf-8") as infile:
                raw_data = json.load(infile)

        pandas_dataframe = pd.DataFrame(raw_data)
        return datasets.Dataset.from_pandas(pandas_dataframe)

    if subset == "":
        dataset = datasets.load_dataset(path)
    else:
        dataset = datasets.load_dataset(path, subset)

    return dataset


def _get_models(base_path):
    models = []
    # Ensure base path exists
    if os.path.exists(base_path):
        for root, dirs, _ in os.walk(base_path):
            # Calculate depth from base_path
            depth = root[len(base_path) :].count(os.sep)
            if depth < 3:  # Check if depth is less than 3
                # Add relative paths of directories at current depth
                models.extend(
                    [
                        os.path.relpath(os.path.join(root, d), start=base_path)
                        for d in dirs
                    ]
                )
            elif depth == 3:
                # At depth 3, we add the directories but do not dive deeper
                for d in dirs:
                    models.append(
                        os.path.relpath(os.path.join(root, d), start=base_path)
                    )
                # No need to explore subdirectories further at depth 3
                dirs[:] = []  # This clears the 'dirs' list in-place
    return models


def load_tokenizer_and_model(
    path, backend="vllm", return_path=False, devices=None, **kwargs
):
    """
    Load tokenizer and model from the given path.

    Args:
        path (str): The path to the tokenizer and model. If path is one of ['llama2', 'mistral', 'llamaguard'], the corresponding public model will be loaded from its default location.
        backend (str, optional): The backend to use for loading the model.
            Supported backends are "transformers" and "vllm".
            Defaults to "vllm".
        return_path (boolean, optional): Return the path to the model instead of the model itself.

    Returns:
        tuple: A tuple containing the tokenizer and model. If return_path is True, the path to the model is returned instead.
    """

    if devices is not None and backend == "vllm":
        raise ValueError("Cannot specify devices for VLLM backend.")
    elif devices is not None and not isinstance(devices, list):
        devices = [devices]

    base_path = "/scratch/public_models/huggingface"
    if os.path.exists(base_path):
        models = _get_models(base_path)
    else:
        models = []

    # Resolve path
    if path == "llama2":
        path = "meta-llama/Llama-2-7b-chat-hf"
    elif path == "llama2base":
        path = "meta-llama/Llama-2-7b-hf"
    elif path == "llama3":
        path = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif path == "llama3base":
        path = "meta-llama/Meta-Llama-3-8B"
    elif path == "mistral":
        path = "mistralai/Mistral-7B-Instruct-v0.3"
    elif path == "llamaguard":
        path = "meta-llama/LlamaGuard-7b"
    elif path == "llamaguard-v2":
        path = "meta-llama/Meta-Llama-Guard-2-8B"
    elif path == "wildguard":
        path = "allenai/wildguard"

    if path in models:
        path = os.path.join(base_path, path)

    if return_path:
        return path

    tokenizer = AutoTokenizer.from_pretrained(path)

    if backend == "transformers":
        if devices is None or len(devices) == 1:
            model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
            if devices is not None:
                model = model.to(devices[0])
        else:
            model = _load_distributed_hf(path, devices, **kwargs)

    elif backend == "vllm":
        model = LLM(model=path, **kwargs)
    else:
        raise ValueError(f"Backend {backend} not supported.")
    return tokenizer, model


def _load_distributed_hf(model_name, devices):

    # Setup model config and memory map
    gpu_count = len(devices)
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
        total_size = model.num_parameters() * 2 / 10**9
        max_memory = {
            i: (
                "{}GiB".format(
                    math.ceil(1 / (gpu_count - 1) + total_size / gpu_count)
                )
                if i in devices
                else "0GiB"
            )
            for i in devices
        }
        max_memory[min(devices)] = "{}GiB".format(
            math.ceil(total_size / gpu_count - 1)
        )

        print(devices)
        print(total_size)
        print(max_memory)

        # Infer not split module:
        if "t5" in model_name:
            no_split_module_classes = ["T5Block"]
        elif (
            "vicuna" in model_name.lower()
            or "koala" in model_name.lower()
            or "llama" in model_name.lower()
        ):
            no_split_module_classes = ["LlamaDecoderLayer"]
        elif "mistral" in model_name:
            no_split_module_classes = ["MistralDecoderLayer"]
        else:
            raise NotImplementedError(
                "No distributed inference support for this model."
            )

        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=no_split_module_classes,
            dtype=torch.bfloat16,
            max_memory=max_memory,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    return model


def run_model(
    model,
    text,
    backend="vllm",
    tokenizer=None,
    role="User",
    categories=None,
    guard=False,
    return_input=False,
    tokenizer_params=None,
    if_format_prompt=True,
    truncate=True,
    **kwargs,
):
    """
    Run LlamaGuard on the given text.

    Args:
        model: The model to be used for inference.
        text: The input text or a list of input texts.
        backend: The backend to be used for running the model. Default is "vllm".
        tokenizer: The tokenizer to be used for tokenizing the input text. Required for "transformers" backend.
        role: The role or a list of roles associated with the input text. Default is "User".
        categories: The categories or a list of policy categories. Default is the one in format.DEFAULT_CATEGORIES.
        guard: Whether to use the guard prompt. Default is False.
        return_input: Whether to return the input text. Default is False.
        format_prompt: Whether to format the prompt. Default is True.
        **kwargs: Additional keyword arguments to be passed to the model (see run_vllm or run_hf for details)

    Returns:
        The output of the model inference.
    """
    if isinstance(text, str):
        text = [text]

    if tokenizer is None and backend == "transformers":
        raise ValueError("Tokenizer is required for transformers backend.")

    if (
        isinstance(role, list)
        and not len(role) == len(text)
        or isinstance(categories, list)
        and not len(categories) == len(text)
    ):
        raise ValueError(
            "If role or categories are lists, they must have the same length as the text."
        )

    prompts = []
    for i, t in enumerate(text):
        if isinstance(role, list):
            r = role[i]
        else:
            r = role

        if isinstance(categories, list):
            c = categories[i]
        else:
            c = categories

        if guard:
            prompts.append(format_prompt_guard(r, t, c))
        elif if_format_prompt:
            kwargs_local = {
                k: v
                for k, v in kwargs.items()
                if k == "e_inst" or k == "b_inst"
            }
            prompts.append(format_prompt(t, **kwargs_local))
        else:
            prompts.append(t)

    outputs = None
    if backend == "vllm":
        # Run model
        outputs = run_vllm(prompts, model, truncate=truncate, **kwargs)
    elif backend == "transformers":
        if tokenizer_params is None:
            tokenizer_params = {}
        inputs = tokenizer(prompts, return_tensors="pt", **tokenizer_params)
        inputs = inputs.to(model.device)
        outputs = run_hf(inputs, model, **kwargs)
    else:
        raise ValueError(f"Backend {backend} not supported.")

    if return_input:
        return outputs, (prompts if backend == "vllm" else inputs)
    else:
        return outputs


def run_vllm(
    inputs,
    model,
    temperature=0.7,
    top_p=0.7,
    logprobs=25,
    truncate=True,
    **kwargs,
):
    """
    Generate output using VLLM

    Args:
        inputs (str): The input text to generate from.
        model: The VLLM model to use for text generation.
        temperature (float): The temperature parameter for controlling the randomness of the generated text.
        top_p (float): The top-p parameter for controlling the diversity of the generated text.
        logprobs (int): The number of log probabilities to return for each token in the generated text.
        **kwargs: Additional keyword arguments for the VLLM model.

    Returns:
        str: The generated text.
    """
    sp = SamplingParams(
        temperature=temperature, top_p=top_p, logprobs=logprobs, **kwargs
    )
    if truncate:
        tokenizer = model.get_tokenizer()
        prompts_token_ids = tokenizer(inputs)["input_ids"]
        truncate_len = model.llm_engine.model_config.max_model_len - kwargs.get(
            "max_tokens", 0
        )
        truncated_prompts_token_ids = [
            p[-truncate_len:] for p in prompts_token_ids
        ]
        return model.generate(
            prompt_token_ids=truncated_prompts_token_ids, sampling_params=sp
        )

    else:
        return model.generate(inputs, sp)


def run_hf(inputs, model, **kwargs):
    """
    Generate output using HuggingFace

    Args:
        inputs: The inputs to be passed to the model.
        model: The model to be run.
        **kwargs: Additional keyword arguments to be passed to the model.

    Returns:
        The outputs of the model.
    """
    outputs = model.generate(**inputs, **kwargs)
    return outputs
