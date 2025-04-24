import argparse
import hashlib
import json
import os

import dill

from .confirm_jailbreak import confirm_jailbreak
from .isolate_jailbreak import isolate_jailbreak


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


def run(datapath, payloads_filename, cache):

    # Load data
    data = load_data(datapath)
    for d in data:
        d["toxic"] = d["label"]

    # Isolate potential jailbreaks
    print("Isolating jailbreaks.")
    run_id = compute_md5_hash({"data": data})
    suspected_jailbreaks = cached_func(
        lambda: isolate_jailbreak(data),
        f"scoringjailbreakisolation_{run_id}",
        cache,
    )

    for d in data:
        d["jailbreak"] = suspected_jailbreaks[d["uid"]]

    print("Scoring jailbreaks.")
    payloads = load_payloads(payloads_filename)
    run_id = compute_md5_hash({"data": data, "payloads": payloads})

    _, _, toxic_ratios, labels = cached_func(
        lambda: confirm_jailbreak(
            data,
            payloads,
            model_name="mistral",
            model_path="mistral-7b-instruct-v0.1",
        ),
        f"scoringjailbreakconfirmation_{run_id}",
        cache,
    )

    for d in data:
        uid = d["uid"]
        del d["toxic"]
        del d["jailbreak"]
        if uid in labels:
            d["jailbreak_score"] = toxic_ratios[uid]
        else:
            d["jailbreak_score"] = 0

    # Save data
    output_datapath = datapath.replace(".json", "_scored.json")
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
        default="data/final_dataset.json",
    )
    parser.add_argument(
        "--payloads_filename",
        type=str,
        help="Filename of the payloads to use for jailbreak confirmation.",
        default="data/alternate_payloads.json",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=".cache_scoring/",
        help="Cached responses directory",
    )

    args = parser.parse_args()

    os.makedirs(args.cache, exist_ok=True)

    run(
        args.datapath,
        args.payloads_filename,
        args.cache,
    )


if __name__ == "__main__":
    main()
