import argparse
import hashlib
import json
import os

import dill

from .confirm_jailbreak import confirm_jailbreak
from .detect_toxicity import detect_toxicity
from .isolate_jailbreak import isolate_jailbreak
from .run_model import run_model


def load_data(datapath):
    with open(datapath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_payloads(payloads_filename):
    with open(payloads_filename, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_md5_hash(data):
    md5 = hashlib.md5()
    md5.update(json.dumps(data, sort_keys=True).encode("utf-8"))
    return str(md5.hexdigest()).replace("0x", "")


def cached_func(func, id, cache):
    cache_loc = os.path.join(cache, id)
    if os.path.exists(cache_loc):
        print("Loading from cache...")
        with open(cache + id, "rb") as f:
            return dill.load(f)
    else:
        result = func()
        with open(cache_loc, "wb") as f:
            dill.dump(result, f)
        return result


def run(
    datapath,
    model_path,
    model_name,
    payloads_filename,
    cache,
    skip_filter=False,
):

    # Load data
    data = load_data(datapath)
    data_hash = compute_md5_hash(data)

    # Create unique ID for this combination of parameters
    run_id = compute_md5_hash(
        {
            "data_hash": data_hash,
            "model_path": model_path,
            "model_name": model_name,
        }
    )

    # Run model
    print("Running model.")
    data, outputs = cached_func(
        lambda: run_model(data, model_name, model_path),
        f"generation_{run_id}",
        cache,
    )

    # Save to data
    for d, o in zip(
        data,
        outputs,
    ):
        d["output"] = o

    # Extract prompts
    prompts = [d["prompt"] for d in data]

    # Find toxic outputs
    if not skip_filter:
        print("Detecting toxicity.")
        run_id = compute_md5_hash({"prompts": prompts, "outputs": outputs})
        toxic_labels = cached_func(
            lambda: detect_toxicity(prompts, outputs),
            f"toxdetection_{run_id}",
            cache,
        )

        for d, p, t in zip(data, prompts, toxic_labels):
            if p.strip() != d["prompt"].strip():
                breakpoint()
            d["toxic"] = t
    else:
        for d in data:
            d["toxic"] = True

    # Isolate potential jailbreaks
    print("Isolating jailbreaks.")
    run_id = compute_md5_hash({"data": data})
    suspected_jailbreaks = cached_func(
        lambda: isolate_jailbreak(data),
        f"jailbreakisolation_{run_id}",
        cache,
    )

    for d in data:
        d["jailbreak"] = suspected_jailbreaks[d["uid"]]

    # Confirm jailbreaks
    print("Confirming jailbreaks.")
    payloads = load_payloads(payloads_filename)
    run_id = compute_md5_hash({"data": data, "payloads": payloads})

    _, _, toxic_ratios, labels = cached_func(
        lambda: confirm_jailbreak(data, payloads, model_name, model_path),
        f"jailbreakconfirmation_{run_id}",
        cache,
    )

    for d in data:
        uid = d["uid"]
        if uid in labels:
            d["jailbreak_label"] = 1 if labels[uid] else 0
            d["toxic_ratio"] = toxic_ratios[uid]
        else:
            d["jailbreak_label"] = 0
            d["toxic_ratio"] = 0

    # Save data
    if not skip_filter:
        output_datapath = datapath.replace(".json", "_output.json")
    else:
        output_datapath = datapath.replace(".json", "_output_no_filter.json")
    with open(output_datapath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run the jailbreak detection pipeline."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="Path to the input data file.",
        default="data/dataset_variants.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Model huggingface name",
        default="lmsys/vicuna-7b-v1.3",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Shortname for model to load fastchat template.",
        default="vicuna",
    )
    parser.add_argument(
        "--payloads_filename",
        type=str,
        help="Filename of the payloads to use for jailbreak confirmation.",
        default="data/payloads.json",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="./cache/",
        help="Cached responses directory",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip the filter step.",
        default=False,
    )

    args = parser.parse_args()

    os.makedirs(args.cache, exist_ok=True)

    run(
        args.datapath,
        args.model_path,
        args.model_name,
        args.payloads_filename,
        args.cache,
        skip_filter=args.skip_filter,
    )


if __name__ == "__main__":
    main()
