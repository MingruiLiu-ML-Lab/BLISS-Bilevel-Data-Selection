num_gpu=4
for s in $(seq 0 $(($num_gpu-1))); do
    echo $s
    CUDA_VISIBLE_DEVICES=$s python src/select_data/predict_data_influence.py --shard $s $num_gpu --round $round &
done
wait
python src/select_data/select_data.py  --data_shards $num_gpu --round $round
