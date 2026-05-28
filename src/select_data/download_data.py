from datasets import load_dataset
import os
os.environ['HF_DATASETS_CACHE'] = 'data/'
os.environ['HF_HOME'] = 'data/'
from huggingface_hub import login
login("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")
for i in range(0, 5):
    ckpt = int(i * 40000)
    data_files = [
        f"data/train-{str(i).zfill(5)}-of-00891*"
        for i in range(int(ckpt / 250), int(ckpt / 250) + 160)
    ]
    num_proc = os.cpu_count() // 2
    dataset = load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=num_proc,
        data_files=data_files,
        verification_mode="no_checks",
        cache_dir='data/',
    )["train"]

    print(f"Round {i}: Total number of examples:", len(dataset))

data_files = [f"data/train-{str(i).zfill(5)}-of-00891*" for i in range(800, 900)]

dataset = load_dataset(
    "loganengstrom/dsdm-candidate-c4",
    num_proc=os.cpu_count() // 2,
    data_files=data_files,
    verification_mode="no_checks",
    cache_dir='data/',
)["train"]
print(f"Data for bilevel training: total number of examples:", len(dataset))

