import json
import random
import time
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional

import torch
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="output")
    remove_unused_columns: bool = False
    use_pseudo_labels: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="mistral")
    pad_token: str = field(
        default="<|finetune_right_pad_id|>", metadata={"help": "Padding token."}
    )
    unk_token: str = field(
        default="<|reserved_special_token_0|>",
        metadata={"help": "Unknown token."},
    )
    data_path: Optional[str] = field(default="data/jailbreaks.json")
    balance: Optional[bool] = field(default=True)

    # Set default values for the following arguments
    gradient_accumulation_steps: int = field(default=8)
    tf32: bool = field(default=True)
    transformer_layer_cls_to_wrap: str = field(default="LlamaDecoderLayer")
    fsdp: str = field(default="full_shard auto_wrap")
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    warmup_ratio: float = field(default=0.03)
    weigth_decay: float = field(default=0.0)
    learning_rate: float = field(default=2e-6)
    save_strategy: str = field(default="no")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    eval_strategy: str = field(default="no")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)
    bf16: bool = field(default=True)
    num_train_epochs: int = field(default=10)
    training_seed: int = field(default=-1)


def prepare_cli(
    args: CustomTrainingArguments,
    script_path: str = "training_scripts/torchrun_entrypoint.py",
) -> str:
    """
    Prepare a complete CLI command with torchrun configuration.

    Args:
        args: CustomTrainingArguments instance
        script_path: Path to the training script

    Returns:
        str: Complete CLI command string
    """
    # Get number of available GPUs
    node_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Generate a seed from the current time
    seed = int(time.time_ns()) % (2**32 - 1)
    local_random = random.Random(seed)
    master_port = local_random.randint(10000, 20000)

    # Build base command
    base_cmd = f"torchrun --nproc_per_node={node_count} --master_port={master_port} {script_path}"

    # Get only fields explicitly defined in CustomTrainingArguments
    custom_fields = set(CustomTrainingArguments.__annotations__.keys())

    cli_args = []
    for field_obj in fields(args):
        # Only process fields explicitly defined in CustomTrainingArguments
        if field_obj.name not in custom_fields:
            continue

        if field_obj.name == "fsdp":
            field_value = " ".join(option._value_ for option in args.fsdp)
        else:
            field_value = getattr(args, field_obj.name)

        field_name = field_obj.name

        # Skip None values
        if field_value is None:
            continue

        # Convert field name from snake_case to kebab-case
        cli_name = field_name.replace("_", "-")

        # Handle boolean flags
        if isinstance(field_value, bool):
            if field_value:
                cli_args.append(f"--{cli_name}")
        # Handle other types
        else:
            # Quote string values
            if isinstance(field_value, str):
                field_value = f'"{field_value}"'
            cli_args.append(f"--{cli_name} {field_value}")

    return f"{base_cmd} {' '.join(cli_args)}"
