#!/bin/bash

# Swin Transformer + RW-LDAM-DRW + EMA 模型训练测试脚本

# ==== 基础路径配置 ====
DATASET_ROOT="/root/workspace/TrainSet/"
RESULTS_DIR="/root/workspace/Mycode1/results"
EXP_NAME="train_checkpoint1_$(date +%Y%m%d_%H%M%S)"
TRANSFORM_CFG="./cfgs/enhanced_custom_transforms_384.yml"
BATCH_SIZE=22

# ==== 训练日志输出 ====
LOG_FILE="${RESULTS_DIR}/${EXP_NAME}/training.log"

# ==== 启动训练 ====
echo "开始训练任务: ${EXP_NAME}"
echo "保存路径: ${RESULTS_DIR}/${EXP_NAME}"
echo "训练日志: ${LOG_FILE}"

python code/train_checkpoint1.py \
    --transform_cfg ${TRANSFORM_CFG} \
    --root_path ${DATASET_ROOT} \
    --res_path ${RESULTS_DIR} \
    --exp ${EXP_NAME} \
    --fold 0 \
    --total_folds 4 \
    --num_classes 6 \
    --epochs 100 \
    --batch_size ${BATCH_SIZE} \
    --base_lr 0.0001 \
    --final_lr 5e-6 \
    --warmup_epochs 5 \
    --cosine \
    --deterministic \
    --seed 42 \
    --max_m 0.5 \
    --s 30 \
    --reweight_epoch 60 \
    --reweight_type inverse \
    --use_ema \
    --ema_decay 0.999 \
    2>&1 | tee "${LOG_FILE}"

python code/train_checkpoint1.py \
    --transform_cfg ${TRANSFORM_CFG} \
    --root_path ${DATASET_ROOT} \
    --res_path ${RESULTS_DIR} \
    --exp ${EXP_NAME} \
    --fold 1 \
    --total_folds 4 \
    --num_classes 6 \
    --epochs 100 \
    --batch_size ${BATCH_SIZE} \
    --base_lr 0.0001 \
    --final_lr 5e-6 \
    --warmup_epochs 5 \
    --cosine \
    --deterministic \
    --seed 42 \
    --max_m 0.5 \
    --s 30 \
    --reweight_epoch 60 \
    --reweight_type inverse \
    --use_ema \
    --ema_decay 0.999 \
    2>&1 | tee "${LOG_FILE}"

python code/train_checkpoint1.py \
    --transform_cfg ${TRANSFORM_CFG} \
    --root_path ${DATASET_ROOT} \
    --res_path ${RESULTS_DIR} \
    --exp ${EXP_NAME} \
    --fold 2 \
    --total_folds 4 \
    --num_classes 6 \
    --epochs 100 \
    --batch_size ${BATCH_SIZE} \
    --base_lr 0.0001 \
    --final_lr 5e-6 \
    --warmup_epochs 5 \
    --cosine \
    --deterministic \
    --seed 42 \
    --max_m 0.5 \
    --s 30 \
    --reweight_epoch 60 \
    --reweight_type inverse \
    --use_ema \
    --ema_decay 0.999 \
    2>&1 | tee "${LOG_FILE}"

python code/train_checkpoint1.py \
    --transform_cfg ${TRANSFORM_CFG} \
    --root_path ${DATASET_ROOT} \
    --res_path ${RESULTS_DIR} \
    --exp ${EXP_NAME} \
    --fold 3 \
    --total_folds 4 \
    --num_classes 6 \
    --epochs 100 \
    --batch_size ${BATCH_SIZE} \
    --base_lr 0.0001 \
    --final_lr 5e-6 \
    --warmup_epochs 5 \
    --cosine \
    --deterministic \
    --seed 42 \
    --max_m 0.5 \
    --s 30 \
    --reweight_epoch 60 \
    --reweight_type inverse \
    --use_ema \
    --ema_decay 0.999 \
    2>&1 | tee "${LOG_FILE}"

# ==== 结束信息 ====
if [ $? -eq 0 ]; then
    echo "✅ 训练完成，模型已保存在: ${SNAPSHOT_PATH}"
else
    echo "❌ 训练失败，请查看日志: ${LOG_FILE}"
fi

# 融合模型
python code/model_soups-2.py --results_path "${RESULTS_DIR}/${EXP_NAME}" --model_type best_ema


# 测试结果
python code/infet_trans_val_swinvit384_V2.py \
    --snapshot_path "${RESULTS_DIR}/${EXP_NAME}" \
    --test_dir "/root/workspace/TestSetB" \
    --weight_name uniform_soup.pkl \
    --batch_size 16 \
    --verbose
