This is code for ICML 2026 [paper](https://arxiv.org/pdf/2510.06048) "BLISS: A Lightweight Bilevel Influence Scoring Method for Data Selection in Language Model Pretraining". The training/validation data processing, model pretraining, and evaluation are built based on [MATES](https://github.com/cxcscmu/MATES).  

## 1 Environment

**Python version**

The code is tested on Python 3.9.

**Install basic dependencies**

```bash
pip install -r requirements.txt
```

## 2 Dataset

We use a tokenized version of the [C4 dataset](https://huggingface.co/datasets/loganengstrom/dsdm-candidate-c4) in our code. Please ensure your disk has at least 500 GB of storage for this dataset. To get the training data for the initial warmup 10k steps, please run:

```bash
python src/select_data/select_data.py --model_name pythia-1b --method random
```

- The selected data will be saved in `data/c4/pythia-1b/random/selected_data-ckpt-0`.
- You should replace the huggingface token in line 18 in `src/select_data/select_data.py` with yours, or you can comment it. 

For preprocessing our validation data LAMBADA, please run:

```bash
python src/select_data/prepare_lambada.py
```

- The processed data will be saved in `data/lambada_openai`.

## 3 Experiments

Our main experiments use 8 GPUs for parallelization.


### 3.1 Warm up proxy/score model 
Warm up the proxy/score model. We first warm up a pythia-31m-2048 model, which will be used as a proxy and score model later. This step is conducted once,
```bash
model_name=pythia-31m-2048 \
method=random \
ckpt=0 \
data_ckpt=0 \
round=0 \
decay=false \
devices=8 \
data_model_name=pythia-1b \
bash scripts/pretrain.sh
```
- The warm-up model will be saved in `out/c4/pythia-31m-2048/random/iter-040000-ckpt.pth`


### 3.2 Warm-up the reference model (reference \theta_r) by the randomly selected data

In the initial warmup 10k steps for the reference model (\theta_r), you can run:

```bash
model_name=pythia-1b \
method=random \
ckpt=0 \
data_ckpt=0 \
round=0 \
decay=false \
data_model_name=pythia-1b \
bash scripts/pretrain.sh
```

- `ckpt=0` denotes we are training from scratch.
- `method=random` means we randomly select from training set for model warmup. 
- The warmup reference model will be saved in `out/c4/$model_name/$method/iter-040000-ckpt.pth` (We use `gradeint accumulation` and update model parameters every 4 steps. With the updated steps=10k, the total iteration is 40k).

### 3.3 Data seletion, pretraining (multiple rounds) 
Our pretraining is run round by round to facilitate the model-aware data selection. Each round consists 
(1) bilevel optimzation for the score model,  (2) predicting the influence score of training data and selecting data. (3) pretraining the reference model by the selected data for 10k steps again. (4) evaluating the reference model on the downstream tasks. You can run one round training step by step like following instructions or do it by

```bash
round=1 gamma=5e-1 reg=1e-7 bash scripts/run.sh
```
- round is set from 1 to 4. 

The detailed instructions for one round training are summarized as follows.

#### 3.3.1 Bilevel optimization for the proxy (\theta_p) /score (\theta_s) model
update pretrained pythia model (proxy model) and influence score model on training and validation set:
```bash
python src/select_data/bilevel_selection.py --devices 8 --round 1 --gamma 1e-2
```

#### 3.3.2 Predict the influence score on training set and select the data
Use the trained influence score model to predict the influence scores of training set:
```bash
round=1 bash scripts/predict_data_influence.sh
```
- Note: set the round
- The selected data will saved in `data/c4/pythia-31m-2048/bilevel/selected_data-ckpt-{args.ckpt}`

#### 3.3.3 Retrain the reference model (\theta_r) on the selected training data.
```bash
model_name=pythia-1b \
method=bilevel \
ckpt=40000\
data_ckpt=40000 \
round=1 \
decay=false \
data_model_name=pythia-31m-2048 \ 
bash scripts/pretrain.sh
```
- `ckpt=40000, 80000, 120000, 160000` is set for the 1st, 2nd, 3rd, 4th round, respectively.
- `method=bilevel` is set since 1st round.  
- The retrained reference model will be saved in `out/c4/$model_name/$method/iter-080000-ckpt.pth`

#### 3.3.4 Evaluation

1️⃣ Evaluate the pretrained model.
It is advised to run the evaluation after the decay stage for intermediate checkpoints for better stability.

```bash
model_name=pythia-1b \
method=bilevel \
ckpt=80000 \
data_ckpt=40000\
round=1\
decay=true \
data_model_name=pythia-31m-2048 \ 
bash scripts/pretrain.sh
```
- The retrained reference model with 800-step lr decay will be saved in `out/c4/$model_name/$method/iter-080800-ckpt.pth`. 
- In this lr decay step, we load the same data checkpoint as in the retraining step.

2️⃣ We provide a simple evaluation example here and you can modify the parameters based on your needs.

```bash
model_name=pythia-1b \
method=bilevel \
ckpt=80800 \
bash scripts/eval.sh
```
- After running the evaluation script, you can find the results in the `results/c4/$model/$method/iter-$ckpt-ckpt/results.json`.

## Citation

If you found this repository helpful, please cite our paper:

```
@inproceedings{hao2026bliss,
title={BLISS: A Lightweight Bilevel Influence Scoring Method for Data Selection in Language Model Pretraining},
author={Jie Hao, Rui Yu, Wei Zhang, Huixia Wang, Jie Xu, Mingrui Liu},
booktitle={43th International Conference on Machine Learning},
year={2026}
}

```
