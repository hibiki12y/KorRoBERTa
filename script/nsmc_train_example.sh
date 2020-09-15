#!/bin/bash
DEVICE_IDS=0

MAX_EPOCH=20
PER_GPU_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1

PEAK_LR=1e-05
MAX_GRAD_NORM=0.0

SAVE_CHECKPOINTS_DIR=checkpoints/yaho
SAVE_CHECKPOINTS_STEPS=5000


python train.py --device_ids ${DEVICE_IDS} \
                --num_train_epochs ${MAX_EPOCH} \
                --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --learning_rate ${PEAK_LR} \
                --max_grad_norm ${MAX_GRAD_NORM} \
                --save_checkpoints_dir ${SAVE_CHECKPOINTS_DIR} \
                --save_checkpoints_steps ${SAVE_CHECKPOINTS_STEPS};