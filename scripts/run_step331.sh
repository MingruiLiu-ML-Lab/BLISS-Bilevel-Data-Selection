round=1
gamma=0.1
reg=1e-7
num_gpu=8
# 3.3.1 Bilevel optimization for proxy and score model
echo "STEP 3.3.1: Bilevel optimization for proxy and score model with lr=1e-4"
python src/select_data/bilevel_selection.py --round $round --devices $num_gpu --gamma $gamma --reg $reg  --micro_batch_size 4 --max_steps 3000 --learning_rate_influence 1e-4 --learning_rate 1e-4
wait

echo "STEP 3.3.1: Bilevel optimization for proxy and score model with lr=1e-6"
python src/select_data/bilevel_selection.py --round $round --devices $num_gpu --gamma $gamma --reg $reg  --micro_batch_size 4 --max_steps 3000 --learning_rate_influence 1e-6 --learning_rate 1e-6
wait

echo "STEP 3.3.1: Bilevel optimization for proxy and score model with lr_in=1e-4, lr_out=1e-5"
python src/select_data/bilevel_selection.py --round $round --devices $num_gpu --gamma $gamma --reg $reg  --micro_batch_size 4 --max_steps 3000 --learning_rate_influence 1e-5 --learning_rate 1e-4
wait

echo "STEP 3.3.1: Bilevel optimization for proxy and score model with warmup"
python src/select_data/bilevel_selection.py --round $round --devices $num_gpu --gamma $gamma --reg $reg  --micro_batch_size 4 --max_steps 3000 --warmup  --learning_rate_influence 1e-5 --learning_rate 1e-5
wait