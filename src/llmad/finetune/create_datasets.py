"""Transforms the raw data into the datasets used in the experiments."""

import argparse
import json
import math
import os
import random


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_local(
    path,
    slice=7,
    startup_size=4,
    seed=42,
    base=False,
    split=False,
    continuous=True,
):

    if not base and not split and not continuous:
        raise ValueError(
            "You must specify at least one of base, split, or continuous"
        )
    if base and split or base and continuous or split and continuous:
        raise ValueError(
            "You cannot specify more than one of base, split, or continuous"
        )

    all_data = load_data(path)
    slice *= 60 * 60 * 24

    if continuous or base:
        # Slice data into chunks
        min_time, max_time = (
            min(all_data, key=lambda x: x["timestamp"])["timestamp"],
            max(all_data, key=lambda x: x["timestamp"])["timestamp"],
        )
        chunk_count = math.ceil((max_time - min_time) / slice)
        data_chunks = [[] for _ in range(chunk_count)]
        for data_entry in all_data:
            chunk = math.floor((data_entry["timestamp"] - min_time) / slice)
            data_chunks[chunk].append(data_entry)

        # Group first chunks for startup phase
        grouped_chunks = [
            [d for data_chunk in data_chunks[:startup_size] for d in data_chunk]
        ]
        for chunk in range(startup_size, len(data_chunks)):
            grouped_chunks.append(data_chunks[chunk])

        # Cutoffs
        cutoffs = []
        for chunk in grouped_chunks:
            cutoffs.append(max(d["timestamp"] for d in chunk))

        if base:
            grouped_chunks[1] = [
                d for chunk in grouped_chunks[1:] for d in chunk
            ]
            cutoffs = cutoffs[0]

        return grouped_chunks, cutoffs

    else:
        # Slice data into chunks
        datasets = []
        for split in [0.5, 0.9]:
            random.seed(seed)
            train_idx = random.sample(
                list(range(len(all_data))), k=int(len(all_data) * split)
            )
            train_idx_set = set(train_idx)
            test_idx = [
                d for d in range(len(all_data)) if d not in train_idx_set
            ]

            train, test = [all_data[i] for i in train_idx], [
                all_data[i] for i in test_idx
            ]
            datasets.append((train, test))

        return datasets, None


def save_dataset(paths, dataset, cutoff, eval=0, seed=42):
    if not isinstance(paths, list):
        paths = [paths]
    if not eval:
        formatted_dataset = {
            "inputs": [d["prompt"] for d in dataset],
            "outputs": ["yes" if d["label"] else "no" for d in dataset],
            "pseudo_labels": [
                (
                    ("yes" if d["label"] else "no")
                    if d["timestamp"] <= cutoff
                    else (
                        "yes"
                        if "jailbreak_label" in d and d["jailbreak_label"]
                        else "no"
                    )
                )
                for d in dataset
            ],
            "timestamps": [d["timestamp"] for d in dataset],
            "uids": [d["uid"] for d in dataset],
        }
        for path in paths:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(formatted_dataset, f)
    else:
        all_indices = list(range(len(dataset)))
        random.seed(seed)
        random.shuffle(all_indices)
        train_cutoff = int(len(dataset) * (1 - eval))

        train_indices, eval_indices = (
            all_indices[:train_cutoff],
            all_indices[train_cutoff:],
        )
        train_dataset = [dataset[i] for i in train_indices]
        eval_dataset = [dataset[i] for i in eval_indices]
        save_dataset(
            [p.replace("train", "eval") for p in paths], eval_dataset, cutoff
        )
        save_dataset(paths, train_dataset, cutoff)


def run(
    path,
    slice=7,
    startup_size=4,
    output_path=None,
    return_relative_paths=False,
    seed=42,
    eval_size=0.05,
):
    all_data = load_data(path)
    output_path = output_path or f"data/jailbreaks/"
    os.makedirs(output_path, exist_ok=True)
    slice *= 60 * 60 * 24

    random.seed(seed)

    # Slice data into chunks
    min_time, max_time = (
        min(all_data, key=lambda x: x["timestamp"])["timestamp"],
        max(all_data, key=lambda x: x["timestamp"])["timestamp"],
    )
    chunk_count = math.ceil((max_time - min_time) / slice)
    data_chunks = [[] for _ in range(chunk_count)]
    for data_entry in all_data:
        chunk = math.floor((data_entry["timestamp"] - min_time) / slice)
        data_chunks[chunk].append(data_entry)

    rel_paths = []
    manifest = {}
    # Create base dataset
    base_train = [d for chunk in data_chunks[:startup_size] for d in chunk]
    base_eval = [d for chunk in data_chunks[startup_size:] for d in chunk]
    output_dir = os.path.join(output_path, "base")
    cutoff = max(b["timestamp"] for b in base_train)

    os.makedirs(output_dir, exist_ok=True)
    save_dataset(
        os.path.join(output_dir, "train.json"),
        base_train,
        cutoff,
        eval=eval_size,
    )
    save_dataset(os.path.join(output_dir, "test.json"), base_eval, cutoff)
    rel_paths.append("base/train.json")
    manifest["base"] = {
        "train_size": len(base_train),
        "eval_size": len(base_eval),
        "train_start": min(b["timestamp"] for b in base_train),
        "train_end": max(b["timestamp"] for b in base_train),
        "eval_start": min(b["timestamp"] for b in base_eval),
        "eval_end": max(b["timestamp"] for b in base_eval),
    }

    # Create incremental datasets
    for i in range(startup_size, chunk_count):
        incremental_train = [d for chunk in data_chunks[:i] for d in chunk]
        incremental_eval = [
            d for chunk in data_chunks[i : i + 1] for d in chunk
        ]
        output_dir = os.path.join(output_path, f"incremental_{i}")
        os.makedirs(output_dir, exist_ok=True)
        save_dataset(
            os.path.join(output_dir, "train.json"),
            incremental_train,
            cutoff,
            eval=eval_size,
        )
        save_dataset(
            os.path.join(output_dir, "test.json"), incremental_eval, cutoff
        )
        rel_paths.append(f"incremental_{i}/train.json")
        manifest[f"incremental_{i}"] = {
            "train_size": len(incremental_train),
            "eval_size": len(incremental_eval),
            "train_start": min(b["timestamp"] for b in incremental_train),
            "train_end": max(b["timestamp"] for b in incremental_train),
            "eval_start": min(b["timestamp"] for b in incremental_eval),
            "eval_end": max(b["timestamp"] for b in incremental_eval),
        }

    # Create random datasets
    for split in [0.5, 0.9]:
        random.seed(seed)
        train_idx = random.sample(
            list(range(len(all_data))), k=int(len(all_data) * split)
        )
        train_idx_set = set(train_idx)
        test_idx = [d for d in range(len(all_data)) if d not in train_idx_set]

        train, test = [all_data[i] for i in train_idx], [
            all_data[i] for i in test_idx
        ]
        output_dir = os.path.join(output_path, f"split_{split}")
        os.makedirs(output_dir, exist_ok=True)
        save_dataset(
            os.path.join(output_dir, "train.json"),
            train,
            cutoff,
            eval=eval_size,
        )
        save_dataset(os.path.join(output_dir, "test.json"), test, cutoff)
        rel_paths.append(f"split_{split}/train.json")
        manifest[f"split_{split}"] = {
            "train_size": len(train),
            "eval_size": len(test),
            "train_start": min(b["timestamp"] for b in all_data),
            "train_end": max(b["timestamp"] for b in all_data),
            "eval_start": min(b["timestamp"] for b in all_data),
            "eval_end": max(b["timestamp"] for b in all_data),
        }

    # Save manifest
    with open(
        os.path.join(output_path, "manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f)

    if return_relative_paths:
        return rel_paths


def main():
    parser = argparse.ArgumentParser(description="Create jailbreak datasets")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the raw data file",
        default="data/jailbreaks.json",
    )
    parser.add_argument(
        "--slice",
        type=int,
        help="Time slice in days",
        default=14,
    )
    parser.add_argument(
        "--startup-size",
        type=int,
        help="Number of initial slices to use for training",
        default=2,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output directory",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=42,
    )
    args = parser.parse_args()

    run(
        args.path,
        args.slice,
        args.startup_size,
        args.output_path,
        seed=args.seed,
    )
