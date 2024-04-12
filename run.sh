#!/bin/bash
python main.py \
    --model_name_or_path ../../t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang fr \
    --source_prefix "translate English to Franch: " \
    --dataset_name ../../data/kde4_en_fr \
    --cache_dir ./cache \
    --output_dir ./output \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --logging_steps 5 \
    --evaluation_strategy "epoch" \
    --eval_steps 50 \
    --overwrite_output_dir False \
    --predict_with_generate