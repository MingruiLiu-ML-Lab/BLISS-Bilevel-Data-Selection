python src/pretrain/pretrain.py \
    --devices $devices \
    --model_name $model_name \
    --method $method \
    --ckpt $ckpt \
    --round $round \
    --data_ckpt $data_ckpt \
    --data_path data/c4/$data_model_name/$method \
    --out_path out/c4/$model_name/$method \
    --decay $decay