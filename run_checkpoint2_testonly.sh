#!/bin/bash

# 测试结果
python code/infet_trans_val_swinvit384_V2.py \
    --snapshot_path "/root/workspace/Mycode1/checkpoints/checkpoint2" \
    --test_dir "/root/workspace/TestSetB" \
    --weight_name checkpoint2.pkl \
    --batch_size 16 \
    --verbose
