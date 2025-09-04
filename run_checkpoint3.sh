#!/bin/bash

EXP_NAME="train_checkpoint3_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=16

# ==== 训练日志输出 ====

# ==== 启动训练 ====
echo "开始训练任务: ${EXP_NAME}"

python code/train_checkpoint3.py \
    --modelroot '/root/workspace/Mycode1/results/checkpoint3' \
    --batch_size ${BATCH_SIZE}


# 测试结果
python code/train_checkpoint3.py \
    --dataroot '/root/workspace/TestSetB' \
    --modelroot '/root/workspace/Mycode1/results/checkpoint3' \
    --testonly \
    --loadfrom '/root/workspace/Mycode1/results/checkpoint3/best.pkl' \
    --result_path '/root/workspace/Mycode1/results/checkpoint3/result.txt' \
    --batch_size ${BATCH_SIZE}
