#!/bin/bash
torchrun \
    --nproc_per_node 4 \
    --master_port=29503 \
    main.py \
    --model_name_or_path ../../t5-small \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 True \
    --source_lang en \
    --target_lang fr \
    --source_prefix "translate English to Franch: " \
    --dataset_name ../../data/kde4_en_fr \
    --cache_dir ./cache \
    --output_dir ./output_sparse_multConv \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --logging_steps 100 \
    --save_steps 5000 \
    --overwrite_output_dir False \
    --predict_with_generate