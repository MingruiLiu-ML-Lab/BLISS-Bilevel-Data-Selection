import numpy as np
import datasets
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from huggingface_hub import login
# login("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")
os.environ['HF_DATASETS_CACHE'] = 'data/'


def get_candidate_dataset(args):
    # Hard coding to be fixed
    data_files = [
        f"data/train-{str(i).zfill(5)}-of-00891*"
        for i in range(int(args.ckpt / 250), int(args.ckpt / 250) + 160)
    ]
    return datasets.load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=64,
        data_files=data_files,
        verification_mode="no_checks",
        cache_dir='data/',
    )["train"]

def select(dataset_size, selection_size, args):
    if args.method=="bilevel":
        dataset = datasets.concatenate_datasets(
            [
                datasets.load_from_disk(
                    f"data/c4/{args.model_name}/{args.method}/data_influence_model-prediction-ckpt-{args.ckpt}/{i}"
                )
                for i in range(int(args.data_shards))
            ]
        )
        metrics = np.array(dataset["prediction"]).reshape(-1)
    else:
        metrics = np.zeros(dataset_size)
    print(">> Metrics shape:", metrics.shape)
    # metrics = metrics/args.temp 
    # Gumbel-Top-$k$ algorithm
    # rng = np.random.default_rng()
    # gumbel_noise = rng.gumbel(size=len(metrics))
    # metrics += gumbel_noise
    indices = np.argpartition(-metrics, selection_size)[:selection_size]
    # indices = np.argsort(-metrics)[:selection_size]
    return indices, metrics[indices]

def get_indices(dataset_size, selection_size, args):
    print(f">> Selecting {selection_size} indices for", args.method)
    ls, metrics = select(dataset_size, selection_size, args)
    indices = list(map(int, ls))
    return indices, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-31m-1024")
    parser.add_argument("--method", type=str, default="bilevel")
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.5)  
    parser.add_argument("--data_shards", type=float, default=8)
    parser.add_argument("--current_time", type=str, default=None)

    args = parser.parse_args()
    args.ckpt = int(args.round * 40000)
    print(args)
    out_dir = Path("out")
    out_dir = Path(f"data/c4/{args.model_name}/{args.method}/selected_data-ckpt-{args.ckpt}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = get_candidate_dataset(args)
    dataset_size = len(ds)
    print(f">> Dataset size: {dataset_size}")
    # Hard coding to be fixed
    selection_size = dataset_size // 5
    indices, sorted_metrics = get_indices(dataset_size, selection_size, args)
    selected_ds = ds.select(indices)
    selected_ds.save_to_disk(
        out_dir,
        num_proc=os.cpu_count() // 2,
    )
    print(f"Save selected data to data/c4/{args.model_name}/{args.method}/selected_data-ckpt-{args.ckpt}")
    np.save("sorted_indices.npy", indices)
    np.save("sorted_metrics.npy", sorted_metrics)
    
