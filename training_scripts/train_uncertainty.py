import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

from llmad import TrainingArguments, prepare_cli
from llmad.finetune.create_datasets import run_local, save_dataset
from llmad.finetune.test import (
    TestingArguments,
    test,
    uncertainty_sampling,
)


@dataclass
class UncertaintySamplingArgs:
    slice: int = field(default=7)
    startup_size: int = field(default=4)
    target_fp: float = field(default=-1)
    uncertainty_percentage: float = field(default=0.1)
    uncertainty_count: int = field(default=-1)
    seed: int = field(default=-1)
    include_certain_points: bool = field(default=True)
    pseudo_labels: bool = field(default=False)
    model_max_length: int = field(default=1024)
    batch_size: int = field(default=4)
    data_split: float = field(default=0.9)
    model_name_or_path: Optional[str] = field(default="mistral")
    data_path: Optional[str] = field(default="data/jailbreaks.json")
    output_dir: str = field(default="output/")
    keep_model: bool = field(default=False)
    jailbreak_threshold: float = field(default=0.6)
    non_jailbreak_threshold: float = field(default=0.6)
    force_pipeline: bool = field(default=False)
    drop_ambiguous: bool = field(default=False)
    self_training: bool = field(default=False)


def analyze_predictions(predictions, real):
    uid_to_pred = {d["uid"]: d for d in predictions}
    uid_to_real = {d["uid"]: d for d in real}
    confidences = {d["uid"]: abs(d["confidence"] - 0.5) for d in predictions}
    sorted_confidences = sorted(confidences.items(), key=lambda item: item[1])
    uid_to_percentile = {
        uid: (i + 1) / len(sorted_confidences)
        for i, (uid, _) in enumerate(sorted_confidences)
    }

    false_positives = {
        k
        for k, v in uid_to_pred.items()
        if v["pred"] and not uid_to_real[k]["label"]
    }

    false_negatives = {
        k
        for k, v in uid_to_pred.items()
        if not v["pred"] and uid_to_real[k]["label"]
    }

    fp_percentiles = {uid_to_percentile[k] for k in false_positives}
    fn_percentiles = {uid_to_percentile[k] for k in false_negatives}

    print(
        f"Error analysis:\n{len(fp_percentiles)} false positives, {len(fn_percentiles)} false negatives. FP confidence percentiles:"
    )
    for k, p in sorted(false_positives.items(), key=lambda item: item[1]):
        print(
            "\tUID {}: {} confidence percentile".format(
                k,
                p,
            )
        )
    print("FN confidence percentiles:")
    for k, p in sorted(false_negatives.items(), key=lambda item: item[1]):
        print(
            "\tUID {}: {} confidence percentile".format(
                k,
                p,
            )
        )


def main():
    """Run training with uncertainty sampling from the command line"""
    parser = transformers.HfArgumentParser((UncertaintySamplingArgs,))
    local_args = parser.parse_args_into_dataclasses()[0]
    local_args.keep_model = True

    if local_args.seed < 0:
        local_args.seed = int(time.time_ns()) % (2**32 - 1)

    random.seed(local_args.seed)
    torch.manual_seed(local_args.seed)

    cache_name = (
        ".data_cache_"
        + local_args.output_dir.split("/")[-1]
        + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=6))
        + "/"
    )

    # Load data
    raw_data, cutoffs = run_local(
        local_args.data_path,
        local_args.slice,
        local_args.startup_size,
    )

    # Make cached dir for current data and model
    os.makedirs(cache_name, exist_ok=True)

    # Initialize data (find what the last build model is)
    current_chunk = 0
    while os.path.exists(
        os.path.join(local_args.output_dir, str(current_chunk))
    ):
        current_chunk += 1

    current_chunk = max(current_chunk - 1, 0)
    if current_chunk == 0:
        # Actual dataset
        data = raw_data[0][:]
    else:
        # Load data
        with open(
            os.path.join(
                local_args.output_dir, str(current_chunk - 1), "data.json"
            ),
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    # Incrementally train model
    for chunk in range(current_chunk, len(raw_data) - 1):

        # Create folder
        os.makedirs(
            os.path.join(local_args.output_dir, str(chunk)), exist_ok=True
        )

        # Build train and eval sets
        save_dataset(
            [
                os.path.join(cache_name, "train.json"),
                os.path.join(local_args.output_dir, str(chunk), "train.json"),
            ],
            data,
            cutoffs[0],
            eval=1 - local_args.data_split,
            seed=local_args.seed,
        )

        # Save a copy of the raw data
        with open(
            os.path.join(local_args.output_dir, str(chunk), "data.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f)

        # Save manifest
        manifest = {
            "train_size": len(data),
            "eval_size": len(raw_data[chunk + 1]),
            "train_start": min(b["timestamp"] for b in data),
            "train_end": max(b["timestamp"] for b in data),
            "eval_start": min(b["timestamp"] for b in raw_data[chunk + 1]),
            "eval_end": max(b["timestamp"] for b in raw_data[chunk + 1]),
        }
        with open(
            cache_name + "manifest.json", "w", encoding="utf-8"
        ) as outfile:
            json.dump(manifest, outfile)

        # Prepare arguments
        training_args = TrainingArguments(
            model_max_length=local_args.model_max_length,
            per_device_train_batch_size=local_args.batch_size,
            output_dir=os.path.join(local_args.output_dir, str(chunk)),
            model_name_or_path=local_args.model_name_or_path,
            data_path=os.path.join(cache_name, "train.json"),
            balance=True,
            training_seed=local_args.seed,
        )

        results_path = os.path.join(training_args.output_dir, "eval_results")
        testing_args = TestingArguments(
            manifest_path=cache_name + "manifest.json",
            model_path=training_args.output_dir,
            results_path=results_path,
            target_fp=local_args.target_fp,
        )

        os.makedirs(results_path, exist_ok=True)

        # Train model
        if not os.path.exists(
            os.path.join(training_args.output_dir, "config.json")
        ):
            print("##### Training increment #", chunk)
            ### Torch run command here
            trained = False
            for i in range(5):
                cli = prepare_cli(training_args)
                try:
                    os.system(cli)
                except RuntimeError:
                    continue
                if os.path.exists(
                    os.path.join(training_args.output_dir, "config.json")
                ):
                    trained = True
                    break

            if not trained:
                raise RuntimeError("Training failed")

        # Save test file and remove training data (to make sure it is not used for eval)
        save_dataset(
            [
                cache_name + "test.json",
                os.path.join(local_args.output_dir, str(chunk), "test.json"),
            ],
            raw_data[chunk + 1],
            cutoffs[0],
        )

        # # Adjust threshold
        # threshold = adjust(training_args, testing_args)
        threshold = 0.5

        # Test on next chunk of data
        raw_predictions = test(training_args, testing_args, threshold=threshold)
        analyze_predictions(raw_predictions, raw_data[chunk + 1])

        if len(raw_predictions) != len(raw_data[chunk + 1]):
            print(
                f"Warning: {len(raw_predictions)} predictions for {len(raw_data[chunk + 1])} examples"
            )

        # Run uncertainty sampling
        if (
            local_args.uncertainty_percentage != 0
            and local_args.uncertainty_percentage != 1
        ):
            if local_args.uncertainty_count == -1:
                count = int(
                    math.ceil(
                        local_args.uncertainty_percentage * len(raw_predictions)
                    )
                )
                uncertain_uids = uncertainty_sampling(
                    raw_predictions, threshold, count
                )
            else:
                uncertain_uids = uncertainty_sampling(
                    raw_predictions, threshold, local_args.uncertainty_count
                )
        elif local_args.uncertainty_percentage == 0:
            uncertain_uids = []
        else:
            uncertain_uids = [d["uid"] for d in raw_predictions]

        uncertain_uids_set = set(uncertain_uids)

        # Make a prediction dict for fast access
        prediction_dict = {d["uid"]: d for d in raw_predictions}

        # Update data
        # Step 1: copy existing data, and reindex it
        new_data = deepcopy(raw_data[chunk + 1])
        uid_to_data = {d["uid"]: d for d in new_data}

        print("Number of uncertain uids: ", len(uncertain_uids))
        print("Number of new data points: ", len(new_data))

        # Step 2: Change labels depending on strategy
        uids_to_drop = []
        for i, d in enumerate(new_data):
            uid = d["uid"]
            # If we want to self-train, we use the previous model's predictions
            if local_args.self_training:
                if (
                    uid in uncertain_uids_set
                    or local_args.include_certain_points
                ):
                    d["label"] = 1 if prediction_dict[uid]["pred"] else 0
                    d["guess"] = 1
                else:
                    uids_to_drop.append(uid)
            # In other cases, we need to handle the certain points
            elif uid not in uncertain_uids_set:
                if local_args.include_certain_points:
                    d["label"] = 1 if prediction_dict[uid]["pred"] else 0
                    d["guess"] = 1
                else:
                    uids_to_drop.append(uid)
            # If we get here, we are not using self-training, and we are handling uncertain points
            # First branch handles pseudo-labels
            elif local_args.pseudo_labels:
                toxicity_percentage = uid_to_data[uid]["toxic_ratio"]
                d["guess"] = 0
                if toxicity_percentage >= local_args.jailbreak_threshold:
                    d["label"] = 1
                elif toxicity_percentage < local_args.non_jailbreak_threshold:
                    d["label"] = 0
                elif local_args.drop_ambiguous:
                    uids_to_drop.append(uid)
                else:
                    d["label"] = 1 if prediction_dict[uid]["pred"] else 0
                    d["guess"] = 1
            # Second branch handles the case where we don't use pseudo-labels, but use real labels instead
            else:
                d["guess"] = 0
            new_data[i] = d

        for uid in reversed(uids_to_drop):
            del uid_to_data[uid]
        new_data = list(uid_to_data.values())

        print("Number of new data points after editing: ", len(new_data))
        print("Number of guesses: ", sum(d["guess"] for d in new_data))

        # Step 3: Optionnaly remove certain examples
        if not local_args.include_certain_points:
            new_data = [d for d in new_data if d["uid"] in uncertain_uids_set]

        # Step 4: Add to main dataset
        data.extend(new_data)

        # Cleanup
        shutil.rmtree(cache_name)
        os.makedirs(cache_name, exist_ok=True)

        if not local_args.keep_model:
            for file in os.listdir(training_args.output_dir):
                if file.endswith(".safetensors"):
                    os.remove(os.path.join(training_args.output_dir, file))


if __name__ == "__main__":
    main()
