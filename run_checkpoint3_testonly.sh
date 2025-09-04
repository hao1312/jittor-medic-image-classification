#!/bin/bash
# 测试结果
python code/train_checkpoint3.py \
    --dataroot '/root/workspace/TestSetB' \
    --modelroot '/root/workspace/Mycode1/checkpoints/checkpoint3' \
    --testonly \
    --loadfrom '/root/workspace/Mycode1/checkpoints/checkpoint3/checkpoint3.pkl' \
    --result_path '/root/workspace/Mycode1/checkpoints/checkpoint3/result.txt' \
    --batch_size 8
