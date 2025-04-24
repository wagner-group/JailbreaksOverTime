import datetime
import json
import os
from dataclasses import dataclass

import transformers

from llmad.finetune.cli_tools import (
    CustomTrainingArguments as TrainingArguments,
)
from llmad.finetune.train import (
    load_and_prepare_data,
)


@dataclass
class TestingArguments:
    manifest_path: str
    model_path: str
    results_path: str
    target_fp: float


from .train import predict


def get_stats(raw):
    true_positives, false_positives, true_negatives, false_negatives = (
        0,
        0,
        0,
        0,
    )
    for datapoint in raw:
        label = datapoint["label"]
        pred = datapoint["pred"]
        # Calculate metrics
        if label == 1:  # Positive sample
            if pred == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:  # Negative sample
            if pred == 1:
                false_positives += 1
            else:
                true_negatives += 1

    # Calculate rates
    fpr = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0
    )
    fnr = (
        false_negatives / (false_negatives + true_positives)
        if (false_negatives + true_positives) > 0
        else 0
    )

    return (fpr, fnr)


def adjust(training_args, testing_args):

    # Load the pretrained model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        truncation_side="left",
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        testing_args.model_path,
    )
    model.eval()

    # Augment tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = training_args.pad_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = training_args.unk_token

    # Load and prepare data
    eval_dataset, _, _, class_values, class_indices = load_and_prepare_data(
        training_args,
        tokenizer,
        pseudo_labeling=training_args.use_pseudo_labels,
        path=training_args.data_path.replace("train", "eval"),
    )

    class_indices = {k: class_values[v] for k, v in class_indices.items()}

    print("Adjusting model")
    raw_outputs = predict(model, eval_dataset, class_indices, 0.5)

    # Compute min threshold for target_fp
    benign_confidence_scores = sorted(
        [x["confidence"] for x in raw_outputs if x["label"] == 0]
    )
    # Get the 1-target_fp quantile from benign_confidence_scores
    threshold = benign_confidence_scores[
        int((1 - testing_args.target_fp) * len(benign_confidence_scores))
    ]

    # Cap threshold to 0.9 (to avoid degenerate cases)
    threshold = min(threshold, 0.9)

    # Save threshold
    with open(f"{testing_args.model_path}/threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    print(f"Adjusted model with threshold {threshold}")

    return threshold


def test(training_args, testing_args, threshold=-1):
    last_dir_name = os.path.basename(os.path.normpath(testing_args.model_path))
    with open(testing_args.manifest_path, "r") as f:
        manifest = json.load(f)
        if "train_size" not in manifest.keys():
            manifest = manifest[last_dir_name]

    start_date_human = datetime.datetime.fromtimestamp(manifest["train_start"])
    end_date_human = datetime.datetime.fromtimestamp(manifest["train_end"])
    eval_start_date_human = datetime.datetime.fromtimestamp(
        manifest["eval_start"]
    )
    eval_end_date_human = datetime.datetime.fromtimestamp(manifest["eval_end"])

    print(
        f"Testing model trained on data from {start_date_human} to {end_date_human}. Eval data from {eval_start_date_human} to {eval_end_date_human}"
    )

    # Load data
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        truncation_side="left",
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        testing_args.model_path,
    )

    print(f"Loaded model")

    # Augment tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = training_args.pad_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = training_args.unk_token

    print(f"Loaded data")

    dataset, _, class_weights, class_values, class_indices = (
        load_and_prepare_data(
            training_args,
            tokenizer,
            pseudo_labeling=False,
            path=training_args.data_path.replace("train", "test"),
        )
    )

    if threshold == -1:
        if (
            os.path.exists(f"{testing_args.model_path}/threshold.json")
            and testing_args.target_fp >= 0
        ):
            with open(f"{testing_args.model_path}/threshold.json", "r") as f:
                threshold = json.load(f)["threshold"]
        else:
            threshold = 0.5

    class_indices = {k: class_values[v] for k, v in class_indices.items()}

    raw = predict(model, dataset, class_indices, threshold)
    (fpr, fnr) = get_stats(raw)

    os.makedirs(testing_args.results_path, exist_ok=True)
    full_results_path = os.path.join(
        testing_args.results_path, "full_results.json"
    )
    summary_results_path = os.path.join(
        testing_args.results_path, "summary_results.json"
    )
    training_cutoff = manifest["train_end"]
    training_size = manifest["train_size"]
    with open(full_results_path, "w", encoding="utf-8") as f:
        save_data = [
            [
                training_cutoff,
                training_size,
                threshold,
                r["timestamp"],
                r["label"],
                r["pred"],
                r["confidence"],
                r["uid"],
            ]
            for r in raw
        ]
        f.write(
            "cutoff\ttraining_size\tthreshold\ttimestamp\tlabel\tpred\tconfidence\tuid\n"
        )
        f.write("\n".join("\t".join(map(str, row)) for row in save_data) + "\n")
    with open(summary_results_path, "w", encoding="utf-8") as f:
        f.write("cutoff\ttraining_size\tfpr\tfnr\n")
        f.write(f"{training_cutoff}\t{training_size}\t{fpr}\t{fnr}\n")

    return raw


def uncertainty_sampling(raw_outputs, threshold, num_samples):
    # Compute normalized uncertainty
    uncertainty = [
        1
        - (
            abs(x["confidence"] - threshold)
            / (threshold if x["confidence"] < threshold else 1 - threshold)
        )
        for x in raw_outputs
    ]

    # uncertainty = [1 - abs(x["confidence"] - 0.5) / 0.5 for x in raw_outputs]

    uncertain_indices = sorted(
        range(len(uncertainty)), key=lambda i: uncertainty[i], reverse=True
    )[:num_samples]

    uncertain_uids = [raw_outputs[i]["uid"] for i in uncertain_indices]

    return uncertain_uids


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (TrainingArguments, TestingArguments)
    )
    training_args, testing_args = parser.parse_args_into_dataclasses()
    test(training_args, testing_args)
