import json
import os
from datetime import datetime


def save_evaluation(
    model_name,
    dataset,
    split,
    metrics,
    extra_info=None,
    save_dir="core/results"
):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    record = {
        "timestamp": timestamp,
        "model": model_name,
        "dataset": dataset,
        "split": split,
        "metrics": metrics,
        "extra_info": extra_info or {}
    }

    # save individual JSON file per run
    filename = f"{model_name}_{dataset}_{split}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w") as f:
        json.dump(record, f, indent=4)

    # append to a master log file
    master_path = os.path.join(save_dir, "all_results.jsonl")
    with open(master_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Results saved to {filepath}")
    return filepath