from lightning.fabric.utilities import ThroughputMonitor, measure_flops
from datasets import Dataset, Features, Sequence, Value, load_dataset
from sympy.printing.codeprinter import requires
from transformers import BertForMaskedLM, BertModel, GPT2LMHeadModel, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.trainer_pt_utils import IterableDatasetShard
from lightning.fabric.strategies import FSDPStrategy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector
# from distributed_softmax_loss import distributed_softmax_loss
from typing import Optional, Union
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
# from backpack import backpack, extend
# from backpack.extensions import BatchGrad
import torch.nn as nn
from torch.autograd import grad
import lightning as L
import datasets
import argparse
import torch
import wandb
import math
import time
import sys
import os
import itertools
torch.backends.cuda.enable_flash_sdp(False)
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, GPTRegression, Block
from lit_gpt.model_eval import GPT as GPT_pre
from lit_gpt.utils import (
    chunked_cross_entropy,
    estimate_flops,
    get_default_supported_precision,
    num_parameters,
)
from huggingface_hub import login
# login("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")
os.environ['HF_DATASETS_CACHE'] = 'data/'


def get_distributed_hypergrad(args, fabric, score_local, grad_fx_local, grad_gy_local, z):
    # G_yz of each sample on the local device
    gyz = [sum([(z_*g_y).sum() for (z_, g_y) in zip(z, grad_gy_local[i])]) for i in range(args.micro_batch_size)]  # shape: [micro_batch_size]
    # gather local_score -> scores_global
    score_local = torch.hstack(score_local)
    gathered_scores = fabric.all_gather(score_local)  
    scores_global = gathered_scores.view(-1) # micro_batch_size * num_gpu
    # softmax across all client samples (n)
    max_val = torch.max(scores_global)
    exp_scores_global = torch.exp(scores_global - max_val)
    exp_scores_local = torch.exp(score_local - max_val)
    # exp_scores_global = torch.exp(scores_global )
    # exp_scores_local = torch.exp(score_local )
    denom = exp_scores_global.sum()
    softmax_local = exp_scores_local / denom  # shape [micro_batch_size]
    result_1 = []   
    result_2 = [] 
    weighted_fx_local = []
    weighted_gy_local = None  
    # accumulate each components over samples on the local device 
    for i in range(args.micro_batch_size):    
        if i > 0:
            weighted_gy_local += softmax_local[i] * gyz[i]  # sum_i P_i * grad_gyz_i, a scalar
        else:
            weighted_gy_local = softmax_local[i] * gyz[i]   # P_i * grad_gyz_i, a scalar
        for j, f_x in enumerate(grad_fx_local[i]):
            if i > 0:
                result_1[j] -= softmax_local[i] * f_x * gyz[i] # - sum_i P_i * grad_fx_i * grad_gyz_i  
                weighted_fx_local[j] += softmax_local[i] * f_x #   sum_i P_j * grad_fx_j   
            else:
                result_1.append(-softmax_local[i] * f_x * gyz[i])  # P_i * grad_fx_i * grad_gyz_i
                weighted_fx_local.append(softmax_local[i] * f_x)   # P_j * grad_fx_j  

    # calculate the hypergradient layer by layer          
    for j in range(len(weighted_fx_local)):
        # sum up P_j * grad_fx_j across clients 
        weighted_fx_sum = fabric.all_reduce(weighted_fx_local[j], reduce_op="sum")  # sum_j P_j * grad_fx_j 
        # hypergradient
        result_2.append(result_1[j] + weighted_fx_sum * weighted_gy_local)  # result_1 + sum_i P_i * sum_j P_j * grad_fx_j * grad_gyz_i  

    return result_2, exp_scores_global, denom, exp_scores_global/denom, max_val
        


def kl_div_token_level(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence between two sets of logits of shape [B, L, V].

    Args:
        logits_a (torch.Tensor): 
            The logits from model A, shape [batch_size, seq_len, vocab_size].
        logits_b (torch.Tensor): 
            The logits from model B, shape [batch_size, seq_len, vocab_size].

    Returns:
        torch.Tensor:
            A scalar tensor representing the averaged KL divergence across 
            the batch dimension and sequence dimension.
    """
    # 1) Convert logits of model A to log probabilities
    #    shape remains [B, L, V]
    log_p = F.log_softmax(logits_a, dim=-1)

    # 2) Convert logits of model B to probabilities
    #    shape remains [B, L, V]
    q = F.softmax(logits_b, dim=-1)

    # 3) Compute KL divergence (P || Q) for each position [B, L, V]
    #    reduction='none' keeps the result shape [B, L, V]
    #    log_target=False indicates the second argument is not log probabilities
    kl = F.kl_div(log_p, q, reduction='none', log_target=False)

    # 4) Sum over the vocabulary dimension to get KL per token
    #    shape: [B, L]
    kl_per_token = kl.sum(dim=-1)

    # 5) Average over the sequence length dimension to get KL per sequence
    #    shape: [B]
    kl_per_seq = kl_per_token.mean(dim=-1)

    # 6) Finally, average over the batch dimension
    #    shape: scalar
    kl_mean = kl_per_seq.mean(dim=0)

    return kl_mean



def train_collate_fn(batch):
    return torch.tensor([sample["input_ids"] for sample in batch], device="cuda")


def val_collate_fn(batch):
    input_ids = [
        torch.tensor(sample["input_ids"], device="cuda") for sample in batch
    ]
    labels = [torch.tensor(sample["labels"], device="cuda") for sample in batch]

    x = pad_sequence(input_ids, batch_first=True, padding_value=0)
    y = pad_sequence(labels, batch_first=True, padding_value=-1)

    max_seq_length = 1024
    if max_seq_length:
        x = x[:, :max_seq_length]
        y = y[:, :max_seq_length]

    return x, y

def setup_distributed(fabric):
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=fabric.global_rank, world_size=fabric.world_size)

def setup(args) -> None:
    precision = args.precision or get_default_supported_precision(training=True)
    print(precision)

    if args.fsdp:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
    fabric = L.Fabric(
        devices=args.devices,
        num_nodes=1,
        strategy=strategy,
        precision=precision,
        loggers=None,
    )
    global wandb_run_name
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_run_name = f"{args.model_name}-{args.method}-s{args.ckpt}-{date_time}"
    global out_dir
    out_dir = args.out_path
    global start_iter
    start_iter = args.ckpt
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    if args.decay:
        global max_iters
        max_iters = 200 * gradient_accumulation_steps
        global stable_iters
        stable_iters = args.ckpt
        global save_interval
        save_interval = 200
    # out/c4/pythia-410m/random/iter-040000-ckpt.pth
    fabric.launch(
        main,
        resume = (
            Path(f"out/c4/pythia-160m/random/iter-040000-ckpt.pth")  # out/c4/pythia-31m/bilevel-selected-data/one-model/iter-020000-ckpt.pth   out/c4/pythia-31m/random/iter-040000-ckpt.pth
        ),
        model_name=args.model_name,
        args=args,
    )

def main(
    fabric: L.Fabric,
    resume: Union[bool, Path],
    model_name: str,
    args,
) -> None:
    # setup_distributed(fabric)
    global model_save_dir 
    model_save_dir = Path(f'out/bilevel_predict_model/{model_name}')
    if fabric.global_rank == 0:
        model_save_dir.mkdir(parents=True, exist_ok=True)
    if args.fsdp:
        fabric.seed_everything(
            1337, workers=True
        )  # same seed for every process to init model (FSDP)
    else:
        fabric.seed_everything(workers=True)  # each process gets a different seed (DDP)

    out_dir = 'data/bilevel'
    t0 = time.perf_counter()

    # Loading proxy model
    config = Config.from_name(f"{model_name}")
    config_pre = Config.from_name(f"{args.reference_model_name}")
    fabric.print(f"Loading model with {config.__dict__}")
    # load the warm-up model
    if args.fsdp:
        with fabric.init_module(empty_init=True):
            proxy_model = GPT(config)
            pre_model = GPT_pre(config_pre)
            score_model = GPTRegression(config)
    else:
        with fabric.init_module(empty_init=False):
            proxy_model = GPT(config)
            pre_model = GPT_pre(config_pre)
            score_model = GPTRegression(config)
    pre_model.apply(pre_model._init_weights)
    proxy_model.apply(proxy_model._init_weights)
    score_model.apply(score_model._init_weights)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(proxy_model):,}")

    pre_model = fabric.setup(pre_model)

    # Replace the head of the score model with 3 linear layers
    score_model.lm_head = nn.Sequential(
        nn.Linear(config.n_embd, 1, bias=True),
        nn.Sigmoid()
    )
    if args.round > 1:
        # iter_num = args.max_steps
        model_ckpt = int(40000*(args.round - 1))
        # proxy_model_checkpoint_path = f"out/bilevel_predict_model/{model_name}/proxy_model_iter-{iter_num:06d}-ckpt-{model_ckpt:06d}.pth"
        score_model_checkpoint_path = f"out/bilevel_predict_model/{model_name}/score_model_iter-{args.max_steps:06d}-ckpt-{model_ckpt:06d}.pth"
        # fabric.print(f"Resuming training from {proxy_model_checkpoint_path}")
        # checkpoint_proxy = torch.load(proxy_model_checkpoint_path, weights_only=True)
        # proxy_model.load_state_dict(checkpoint_proxy)
        fabric.print(f"Resuming training from {score_model_checkpoint_path}")
        checkpoint_score = torch.load(score_model_checkpoint_path, weights_only=True)  
        score_model.load_state_dict(checkpoint_score)

    proxy_model = fabric.setup(proxy_model)
    # Initailize the lower_level_optimizer
    lower_level_optimizer = torch.optim.AdamW(
        proxy_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        foreach=False,
    )
    lower_level_optimizer = fabric.setup_optimizers(lower_level_optimizer)
    score_model = fabric.setup(score_model)
    # Initialize the optimizer for the score model
    upper_level_optimizer = torch.optim.AdamW(
        score_model.parameters(),
        lr=args.learning_rate_influence,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        foreach=False,
    )
    upper_level_optimizer = fabric.setup_optimizers(upper_level_optimizer)
    # Loading the training data
    train_data = load_datasets(args.seed)
    train_data = IterableDatasetShard(
        train_data,
        batch_size=args.micro_batch_size,
        num_processes=fabric.world_size,
        process_index=fabric.global_rank,
    )

    state = {
        "models": {
        "pre_model": pre_model,
        "proxy_model": proxy_model,
        "score_model": score_model
        },
        "optimizers":{
            "lower_level_optimizer": lower_level_optimizer,
            "upper_level_optimizer": upper_level_optimizer
        },
        "iter_num": 0,
        "step_count": 0,
    }

    # Loading the warm-up pretrained reference model 
    if args.ckpt == 40000:
        pre_resume = f"out/c4/{args.reference_model_name}/random/iter-{args.ckpt:06d}-ckpt.pth"
    else:
        pre_resume = f"out/c4/{args.reference_model_name}/{args.method}/iter-{args.ckpt:06d}-ckpt.pth"
    fabric.print(f"Resuming training from {pre_resume}")
    state_temp_pre = {"model": state["models"]["pre_model"]}
    fabric.load(pre_resume, state_temp_pre)
    state["models"]["pre_model"] = state_temp_pre["model"]

    # Loading the proxy/score model
    fabric.print(f"Resuming training from {resume}")
    state_temp = {"model": state["models"]["proxy_model"]}
    fabric.load(resume, state_temp)
    state["models"]["proxy_model"] = state_temp["model"]

    if args.round == 1:
        filtered_state_dict = {k: v for k, v in state_temp["model"].state_dict().items() if not k.startswith('lm_head')}
        state["models"]["score_model"].load_state_dict(filtered_state_dict, strict=False)     
    # wandb logging
    if args.wandb_log and fabric.global_rank == 0:
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, config=vars(args), dir=out_dir
        )

    train_time = time.perf_counter()
    train(args, fabric, state, train_data, model_name)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(args,
    fabric: L.Fabric,
    state: dict,
    train_data,
    model_name
) -> None:
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_time = args.time
    score_model = state["models"]["score_model"]
    proxy_model = state["models"]["proxy_model"]
    pre_model = state["models"]["pre_model"]
    pre_model.eval()
    upper_level_optimizer = state["optimizers"]["upper_level_optimizer"]
    lower_level_optimizer = state["optimizers"]["lower_level_optimizer"]
    lower_level_scheduler = get_cosine_schedule_with_warmup(
        optimizer=lower_level_optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps*args.epochs
    )
    upper_level_scheduler = get_cosine_schedule_with_warmup(
        optimizer=upper_level_optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps*args.epochs
    )

    throughput = ThroughputMonitor(fabric, window_size=50)
    total_t0 = time.perf_counter()
    score_sum = None
    score_max_val = None
    state["iter_num"] = 0
    z = [1/param.numel() * torch.ones(param.size()).half().to(fabric.device) for param in proxy_model.parameters()]
    args.num_samples = args.devices * args.micro_batch_size * args.max_steps
    with tqdm(total=(args.max_steps*args.epochs+1)) as pbar:
        for epoch in range(args.epochs):
            inner_step = 0
            val_dataloader = DataLoader(
                torch.load("data/lambada_openai/train.pt"),
                batch_size=args.micro_batch_size,
                collate_fn=val_collate_fn,
            )
            train_dataloader = DataLoader(
                train_data,
                batch_size=args.micro_batch_size,
                collate_fn=train_collate_fn,
            )
            train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
            val_iter = itertools.cycle(val_dataloader)
            train_iter = iter(train_dataloader)
            for input_ids in train_dataloader:
                if inner_step >= args.max_steps + 1:
                    break
                inner_step +=1
                pbar.update(1)
                iter_num = state["step_count"]
                iter_t0 = time.perf_counter()

                # Multiple steps for update lower-level variable
                for _ in range(args.inner_steps): 
                    inner_input_ids = next(train_iter)
                    inner_logits_pre = pre_model(inner_input_ids)
                    inner_logits_proxy, inner_emb = proxy_model(inner_input_ids)
                    if args.bilevel:
                        score_output = score_model(inner_input_ids).squeeze(-1)
                        # local_score = torch.exp(score_output)
                        local_score = torch.exp(score_output - score_max_val) if score_max_val is not None else torch.exp(score_output) 
                        inner_sample_weights = local_score/score_sum if score_sum is not None else local_score/local_score.sum()
                        # inner_sample_weights = torch.softmax(local_score, dim=-1)
                   
                    else:
                        inner_sample_weights = torch.softmax(torch.ones(args.micro_batch_size).cuda(inner_logits_proxy.device), dim=0)
                    # inner_sample_weights = args.a * score_model(inner_emb[:, -1,:]).squeeze(-1)
                    inner_weighted_loss_tr, inner_unweighted_loss = chunked_cross_entropy(
                        inner_logits_proxy[:, :-1, :].contiguous(),
                        targets=inner_input_ids[:, 1:].contiguous(), 
                        # logits_pre=inner_logits_pre[:, :-1, :].contiguous(),
                        weight=inner_sample_weights,
                        chunk_size=0
                        ) 
                    param_norm = args.reg *sum([y.norm(2).pow(2) for y in proxy_model.parameters()])
                    kl_div = kl_div_token_level(inner_logits_proxy, inner_logits_pre.detach())
                    inner_weighted_loss_tr += param_norm  + args.gamma * kl_div#+ args.gamma * entropy(inner_sample_weights)   # + args.lamb * (inner_sample_weights.sum() - 1).pow(2)
                    inner_weighted_loss_tr.backward()
                    fabric.clip_gradients(proxy_model, lower_level_optimizer, max_norm=args.grad_clip)
                    lower_level_optimizer.step()
                    lower_level_optimizer.zero_grad()
                    upper_level_optimizer.zero_grad()

                # Compute the graident of F_y
                val_input_ids, val_labels = next(val_iter)
                val_logits_proxy, _ = proxy_model(val_input_ids)
                val_loss_ce, _ = chunked_cross_entropy(
                    val_logits_proxy[:, :-1, :].contiguous(),
                    targets=val_labels[:, 1:].contiguous(),
                    chunk_size=0
                )
                F_y = torch.autograd.grad(val_loss_ce, proxy_model.parameters(), retain_graph=True)

                # Compute and update z
                logits_proxy, emb = proxy_model(input_ids)
                logits_pre = pre_model(input_ids)

                if args.bilevel:      
                    score_output = score_model(input_ids).squeeze(-1) 
                    # local_score = torch.exp(score_output)
                    local_score = torch.exp(score_output - score_max_val) if score_max_val is not None else torch.exp(score_output)  
                    sample_weights = local_score/score_sum if score_sum is not None else local_score/local_score.sum()
                else:
                    sample_weights = torch.softmax(torch.ones(args.micro_batch_size).cuda(logits_proxy.device), dim=0)
                    score = sample_weights

                weighted_loss_tr, unweighted_loss = chunked_cross_entropy(
                    logits_proxy[:, :-1, :].contiguous(),
                    targets=input_ids[:, 1:].contiguous(),
                    # logits_pre=logits_pre[:, :-1, :].contiguous(),
                    weight=sample_weights,
                    chunk_size=0
                ) 
                
                param_norm = args.reg * sum([y.norm(2).pow(2) for y in proxy_model.parameters()])
                kl_div = kl_div_token_level(logits_proxy, logits_pre.detach())
                weighted_loss_tr += param_norm + args.gamma * kl_div # + args.gamma * entropy(sample_weights) # + args.lamb * (sample_weights.sum() - 1).pow(2)
                G_y = torch.autograd.grad(weighted_loss_tr, proxy_model.parameters(), retain_graph=True, create_graph=True)
                G_y_norm = sum([g.norm() for g in G_y if g is not None])


                for loop in range(args.z_loops):    
                    G_yyz = torch.autograd.grad(G_y, proxy_model.parameters(), grad_outputs=z, retain_graph=True)
                    # update the linear system solution
                    g_norm = []
                    for idx, (g_yyz, f_y) in enumerate(zip(G_yyz, F_y)):
                        g = g_yyz - f_y + args.lamb * z[idx]
                        if g.norm() > args.grad_clip:
                           g = g/g.norm()
                        z[idx] = z[idx] -  args.lr_z * g
                        g_norm.append(g.norm().item())

                    g_norm_sum = sum(g_norm)
                    fabric.print(f'g_norm: {loop}, {g_norm_sum}')
                del G_y, G_yyz, F_y
                torch.cuda.empty_cache()
                z_norm = sum([z_.norm() for z_ in z if z is not None])
                # Compute the gradient of G_xyz
                # input_ids = next(iter(train_dataloader))
                # input_ids, labels = next(val_iter) 
                for i in range(len(z)):
                    fabric.all_reduce(z[i], reduce_op="sum")
                    z[i] = z[i]/fabric.world_size
                fx_per_sample_grads = []
                gy_per_sample_grads = []
                score_local = []
                for input in input_ids:
                    input = input.unsqueeze(0)
                    score = score_model(input).squeeze(-1)
                    score_local.append(score)
                    f_x = torch.autograd.grad(score, score_model.parameters())
                    fx_per_sample_grads.append(f_x)  
                    tr_logits_proxy, _ = proxy_model(input)
                    ce_loss, _ = chunked_cross_entropy(
                                tr_logits_proxy[:, :-1, :].contiguous(),
                                targets=input[:, 1:].contiguous(),
                                chunk_size=0
                            )
                    g_y = torch.autograd.grad(ce_loss, proxy_model.parameters())
                    gy_per_sample_grads.append(g_y)  

                hypergrad, scores_global, score_sum, softmax_score, score_max_val = get_distributed_hypergrad(args, fabric, score_local, fx_per_sample_grads, gy_per_sample_grads, z)

                # Re-initialize the solution to Linear system
                z = [1/param.numel() * torch.ones(param.size()).to(fabric.device) for param in proxy_model.parameters()]
                # Compute the hypergradient 
                x_norm = None
                for p, hg in zip(score_model.parameters(), hypergrad):
                    if p.requires_grad:
                        # Collect the distributed gradient
                        fabric.all_reduce(hg.data, reduce_op="sum")
                        p.grad = hg.data
                        if x_norm == None:
                            x_norm = p.grad.norm()
                        else:
                            x_norm += p.grad.norm()
                        if p.grad.norm() > args.grad_clip:
                            p.grad /= p.grad.norm()
                lower_level_optimizer.zero_grad()
                upper_level_optimizer.step()
                fabric.print(f'{state["step_count"]} score: {[f"{x:.6f}" for x in scores_global.tolist()]}')
                fabric.print(f'{state["step_count"]} weight: {[f"{x:.6f}" for x in softmax_score.tolist()]}')
                torch.cuda.empty_cache()
                lower_level_optimizer.zero_grad()
                upper_level_optimizer.zero_grad()
                fabric.print(f'\n[tr_loss/val_loss/f1-f_r/grad_y_norm/z_norm]: {weighted_loss_tr.item():.4f}/{val_loss_ce.item():.4f}/{unweighted_loss.mean().item():.4f}/{G_y_norm:.4f}/{z_norm:.4f}')

                if args.wandb_log and fabric.global_rank == 0:
                    wandb.log(
                        {
                            "step": state["step_count"],
                            "train_loss": weighted_loss_tr.item(),
                            "val_loss": val_loss_ce.item(),
                            "lr_proxy": lower_level_scheduler.get_last_lr()[0],
                            "lr_score": upper_level_scheduler.get_last_lr()[0],
                            "z_norm": z_norm,
                            "unweighted_tr_loss": unweighted_loss.mean().item(),
                            "grad_y_norm": G_y_norm,
                            "grad_z_norm": g_norm_sum, 
                            "grad_x_norm": x_norm.item(),
                            "kl_div": kl_div,
                            "reg_term": param_norm,
                        }
                    )
                lower_level_scheduler.step()
                upper_level_scheduler.step()

                if iter_num % args.log_interval == 0:
                    loss_item = val_loss_ce.item()  # expensive device-to-host synchronization
                    t1 = time.perf_counter()
                    throughput.update(
                        time=t1 - total_t0,
                        batches=iter_num,
                        samples=iter_num * args.micro_batch_size,
                        lengths=iter_num * args.micro_batch_size * proxy_model.max_seq_length,
                        flops=0 * args.log_interval,
                    )
                    throughput.compute_and_log(step=iter_num)
                    fabric.print(
                        f"iter {iter_num} step {state['step_count']}: loss {loss_item:.4f}, iter time:"
                        f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' }"
                    )
                if state["step_count"] % args.save_interval == 0 or state["step_count"] == args.max_steps:
                    # if not os.path.exists(f"out/bilevel_predict_model/{model_name}"):
                        # os.makedirs(f"out/bilevel_predict_model/{model_name}")
                    iter_num = state["step_count"]
                    proxy_model_checkpoint_path = f"{model_save_dir}/proxy_model_iter-{iter_num:06d}-ckpt-{args.ckpt:06d}.pth"
                    score_model_checkpoint_path = f"{model_save_dir}/score_model_iter-{iter_num:06d}-ckpt-{args.ckpt:06d}.pth"
                    fabric.print(f"Saving checkpoint to {str(proxy_model_checkpoint_path)!r}")
                    fabric.print(f"Saving checkpoint to {str(score_model_checkpoint_path)!r}") 
                    fabric.save(proxy_model_checkpoint_path, proxy_model.state_dict())
                    fabric.save(score_model_checkpoint_path, score_model.state_dict())

                state["step_count"] += 1


def load_datasets(seed):
    print(f'Loading data ... ') 
    # data_files = [f"data/train-{str(i).zfill(5)}-of-00891*" for i in range(800, 900)]
    data_files = [
        f"data/train-{str(i).zfill(5)}-of-00891*"
        for i in range(int(args.ckpt / 250), int(args.ckpt / 250) + 160)
    ]
    train_dataset = load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=os.cpu_count() // 2,
        data_files=data_files,
        verification_mode="no_checks",
        cache_dir='data/',
    )["train"]
    train_dataset = train_dataset.shuffle(seed=seed)
    print(f'data size: {len(train_dataset)}')
    return train_dataset


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Distributed Training with Fabric")
    # wandb
    parser.add_argument("--wandb_log", type=bool, default=True, help="Log the training process")
    parser.add_argument("--wandb_project", type=str, default="bilevel_section", help="Name of project")
    parser.add_argument("--wandb_run_name", type=str, default="bilevel_section", help="Running name of the project")

    # General hyperparameters
    parser.add_argument("--devices", type=int, default=4, help="Number of devices to use for training")
    parser.add_argument("--model_name", type=str, default="pythia-31m-1024", help="Name of the model")
    parser.add_argument("--reference_model_name", type=str, default="pythia-1b", help="Name of the reference model")
    parser.add_argument("--method", type=str, default="bilevel", help="Training method")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint iteration to resume from")
    parser.add_argument("--round", type=int, default=1, help="pretraining round")
    parser.add_argument("--precision", type=str, default=None, help="Training precision")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--data_path", type=str, default="data/c4/pythia-410m/random/0", help="Path to the training data")
    parser.add_argument("--out_path", type=str, default="out", help="Path to save outputs")
    parser.add_argument("--decay", action="store_true", help="Use learning rate decay")    
    parser.add_argument("--seed", type=int, default=0, help="random seed for data shuffle")
    parser.add_argument("--num_samples", type=int, default=320000, help="The number of training samples")
    parser.add_argument("--fsdp", type=bool, default=False, help="Use fsdp or not")
    parser.add_argument("--bilevel", type=bool, default=True, help="Use bilevel optimization or not")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=3000, help="Maximum number of training steps")
    parser.add_argument("--epochs", type=int, default=1, help="The number of training epochs")
    parser.add_argument("--inner_steps", type=int, default=1, help="The number of inner training steps")
    parser.add_argument("--z_loops", type=int, default=3, help="The number of loops for z")
    parser.add_argument("--lr_z", type=float, default=1e-2, help="The learning rate for z")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint saving interval")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for proxy model")
    parser.add_argument("--learning_rate_influence", type=float, default=1e-5, help="Learning rate for influence score model")
    parser.add_argument("--batch_size", type=int, default=64, help="Total batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size per device")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=1e-4, help="coefficient of regularization term") # 1e-4
    parser.add_argument("--lamb", type=float, default=1e-2, help="coefficient of regularization term")
    parser.add_argument("--reg", type=float, default=1e-6, help="coefficient of regularization term") # 1e-6
    parser.add_argument("--score_reg", type=float, default=1e-1, help="coefficient of regularization term")
    parser.add_argument("--gamma_score", type=float, default=0.5, help="momentum parameter")
    parser.add_argument("--a", type=float, default=1.0, help="scaling of the score model output")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="score threshold")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warm up steps")
    parser.add_argument("--warmup", action="store_true", help="Enable warmup or not")
    parser.add_argument("--time", type=str, default="2025-01-15_05-00-00", help="Saving model checkpoint time")

    args = parser.parse_args()
    if args.warmup:
        args.warmup_steps = args.max_steps * args.epochs * 0.1
    else:
        args.warmup_steps = 0
    if args.time == None:
       args.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    if args.ckpt == None:
        args.ckpt = int(args.round * 40000)
    print(args)
    setup(args)