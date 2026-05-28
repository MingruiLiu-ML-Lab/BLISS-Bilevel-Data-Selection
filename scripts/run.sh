set -e
set -o pipefail

round=$round
#gamma=$gamma
#reg=$reg
num_gpu=8
# download all the training data
echo "STEP 2.1: download data"
python src/select_data/download_data.py

# prepare the warm-up training data by selecting randomly
echo "STEP 2.2: prepare the training data by selecting randomly"
python src/select_data/select_data.py --model_name pythia-1b --method random

echo "STEP 2.3: prepare the validation data"
python src/select_data/prepare_lambada.py

# Warm up the proxy/score model. We first warm up a pythia-160m model, which will be used as a proxy and score model later. This step is conducted once,
echo "STEP 3.1: Warm up the proxy model"
model_name=pythia-160m method=random ckpt=0 data_ckpt=0 round=0 decay=false devices=$num_gpu data_model_name=pythia-1b bash scripts/pretrain.sh

echo "STEP 3.2: Warm up the reference model"
model_name=pythia-1b method=random ckpt=0 data_ckpt=0 round=0 decay=false devices=$num_gpu data_model_name=pythia-1b bash scripts/pretrain.sh

# 3.3.1 Bilevel optimization for proxy and score model
echo "STEP 3.3.1: Bilevel optimization for proxy and score model"
python src/select_data/bilevel_selection.py --model_name pythia-160m-1024 --round $round --devices $num_gpu --gamma 1e-2 --reg 1e-7  --micro_batch_size 4 --max_steps 6000 --warmup --inner_steps 5
wait

# 3.3.2 Predict the influence score of training data by the trained score model
echo "STEP 3.3.2: Predict the influence score of training data by the trained score model"
for s in $(seq 0 $(($num_gpu-1))); do
    echo $s
    CUDA_VISIBLE_DEVICES=$s python src/select_data/predict_data_influence.py --model_name pythia-160m-1024 --shard $s $num_gpu --round $round --iter_num 6000 &
done
wait
# Select top-20% data based on the score ranking of training data
python src/select_data/select_data.py  --data_shards $num_gpu --round $round --model_name pythia-160m-1024
wait

# 3.3.3 Retrain LLM by the selected top-20% data, using the constant learning rate
echo "STEP 3.3.3: Retrain LLM by the selected top-20% data, using the constant learning rate"
ckpt=$(($round * 40000))
data_ckpt=$ckpt
model_name=pythia-1b method=bilevel decay=false ckpt=$ckpt data_ckpt=$data_ckpt round=$round devices=$num_gpu data_model_name=pythia-160m-1024 bash scripts/pretrain.sh
wait

# 3.3.4 (1) Retrain LLM by the selected top-20% data, using learning rate decay
echo "STEP 3.3.4 (1): Retrain LLM by the selected top-20% data, using learning rate decay"
ckpt=$(($round * 40000 + 40000))
data_ckpt=$(($round * 40000))
model_name=pythia-1b method=bilevel decay=true ckpt=$ckpt data_ckpt=$data_ckpt round=$round devices=$num_gpu data_model_name=pythia-160m-1024 bash scripts/pretrain.sh
wait

# 3.3.4 (2) Evaluate the trained LLM on the downstream tasks
echo "STEP 3.3.4 (2): Evaluate the trained LLM on the downstream tasks"
ckpt=$(($round * 40000 + 40800))
formatted_ckpt=$(printf "%06d" $ckpt)
model_name=pythia-1b method=bilevel ckpt=$formatted_ckpt bash scripts/eval.sh
