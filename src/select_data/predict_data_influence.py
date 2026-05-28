from transformers import AutoTokenizer
import torch.nn as nn
from datasets import load_dataset
import argparse
import torch
import os
from datetime import datetime
from pathlib import Path
import sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# import lightning as L
from lit_gpt import Config
from lit_gpt.model import GPTRegression
import lightning as L
from huggingface_hub import login
# login("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")
class ModelAnnotator:
    def __init__(self, model_name, device_batch_size, score_model_path):
        self.model_name = model_name
        self.device_batch_size = device_batch_size
        self.score_model_path = score_model_path
        checkpoint_score = torch.load(score_model_path)  
        config = Config.from_name(f"{model_name}")

        self.score_model = GPTRegression(config)
        self.score_model.lm_head =  nn.Sequential(
                            nn.Linear(config.n_embd, 1, bias=True),
                            nn.Sigmoid()
                        )
        print(f'Loading score model from {score_model_path}')
        self.score_model.load_state_dict(checkpoint_score)
        self.score_model.half()
        self.score_model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")
        self.score_model.to(self.device)

    def __getstate__(self):
        return {
            "model_name": self.model_name,
            "device_batch_size": self.device_batch_size,
            "score_model_path": self.score_model_path,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    @torch.no_grad()
    def __call__(self, example, indices):
        output = {"index": indices}
        inputs = torch.tensor(example["input_ids"], device=self.device)
        sample_weights = self.score_model(inputs).squeeze(-1)
        output["prediction"] = sample_weights.detach().float().cpu().numpy()
        # print(output["prediction"])
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-31m-1024")
    parser.add_argument("--method", type=str, default="bilevel")
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--iter_num", type=int, default=3000)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=512)
    parser.add_argument("--time", type=str, default=None)

    args = parser.parse_args()

    args.ckpt = int(args.round * 40000)
    print(args)
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_time = args.time
    output_dir = Path(f"data/c4/{args.model_name}/{args.method}/data_influence_model-prediction-ckpt-{args.ckpt}")
    model_path = f'out/bilevel_predict_model/{args.model_name}/score_model_iter-{args.iter_num:06d}-ckpt-{args.ckpt:06d}.pth'
    if args.shard[0] == 0:    
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Make a data saving directory: {output_dir}')
    data_files = [
        f"data/train-{str(i).zfill(5)}-of-00891*"
        for i in range(int(args.ckpt / 250), int(args.ckpt / 250) + 160)
    ]
    num_proc = os.cpu_count() //2
    dataset = load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=num_proc,
        data_files=data_files,
        verification_mode="no_checks",
        cache_dir='data/',
    )["train"]
    src_dataset = dataset.shard(args.shard[1], args.shard[0], contiguous=True)
    dataset = src_dataset

    print("Total number of examples:", len(dataset))

    dataset = dataset.map(
        ModelAnnotator(args.model_name, args.device_batch_size, model_path),
        batched=True,
        with_indices=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {output_dir}")
    dataset.save_to_disk(os.path.join(output_dir, f"{args.shard[0]}"))
