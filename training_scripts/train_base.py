import json
import os
import random
import time
from dataclasses import dataclass, field

import transformers

from llmad import TrainingArguments, prepare_cli
from llmad.finetune.create_datasets import run_local, save_dataset
from llmad.finetune.test import TestingArguments, adjust, test


@dataclass
class JailbreakArguments:
    slice: int = field(default=7)
    startup_size: int = field(default=4)
    raw_data_path: str = field(default="data/jailbreaks.json")
    data_split: float = field(default=0.9)
    base_output_dir: str = field(default="output/")
    target_fp: float = field(default=-1)
    keep_model: bool = field(default=False)
    run_base: bool = field(default=True)
    run_split: bool = field(default=False)


def main():
    """Run training from the command line"""
    parser = transformers.HfArgumentParser(
        (
            JailbreakArguments,
            TrainingArguments,
        )
    )
    jailbreak_args, training_args = parser.parse_args_into_dataclasses()

    # Setup cache
    if training_args.training_seed < 0:
        training_args.training_seed = int(time.time_ns()) % (2**32 - 1)

    random.seed(training_args.training_seed)

    if jailbreak_args.run_base:
        data, cutoff = run_local(
            jailbreak_args.raw_data_path,
            jailbreak_args.slice,
            jailbreak_args.startup_size,
            seed=training_args.training_seed,
            base=True,
            continuous=False,
        )

        train, test_data = data[0], data[1]

        # Create folder
        os.makedirs(
            jailbreak_args.base_output_dir,
            exist_ok=True,
        )

        # Save train, eval and test sets
        save_dataset(
            [
                os.path.join(jailbreak_args.base_output_dir, "train.json"),
            ],
            train,
            cutoff,
            eval=1 - jailbreak_args.data_split,
            seed=training_args.training_seed,
        )
        save_dataset(
            [
                os.path.join(jailbreak_args.base_output_dir, "test.json"),
            ],
            test_data,
            cutoff,
        )

        # Save manifest
        manifest = {
            "train_size": len(train),
            "eval_size": len(test_data),
            "train_start": min(b["timestamp"] for b in train),
            "train_end": max(b["timestamp"] for b in train),
            "eval_start": min(b["timestamp"] for b in test_data),
            "eval_end": max(b["timestamp"] for b in test_data),
        }
        with open(
            os.path.join(jailbreak_args.base_output_dir, "manifest.json"),
            "w",
            encoding="utf-8",
        ) as outfile:
            json.dump(manifest, outfile)

        # Run training
        training_args.data_path = os.path.join(
            jailbreak_args.base_output_dir, "train.json"
        )
        training_args.output_dir = jailbreak_args.base_output_dir

        for i in range(5):
            try:
                cli = prepare_cli(training_args)
                print(cli)
                os.system(cli)
                break
            except RuntimeError:
                continue

        # Create testing args
        testing_args = TestingArguments(
            manifest_path=os.path.join(
                jailbreak_args.base_output_dir, "manifest.json"
            ),
            model_path=training_args.output_dir,
            results_path=os.path.join(
                jailbreak_args.base_output_dir, "eval_results"
            ),
            target_fp=jailbreak_args.target_fp,
        )
        os.makedirs(testing_args.results_path, exist_ok=True)

        test(training_args, testing_args)

        # Clean up
        if not jailbreak_args.keep_model:
            for file in os.listdir(training_args.output_dir):
                if file.endswith(".safetensors"):
                    os.remove(os.path.join(training_args.output_dir, file))


if __name__ == "__main__":
    main()
