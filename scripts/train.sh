#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
readonly MASTER_ADDR="127.0.0.1"
readonly MASTER_PORT="10356"
readonly MICRO_BATCH_SIZE=1
readonly LR=1e-5
readonly GRADIENT_ACCUMULATION_STEPS=8
PWD="$(pwd)" && readonly PWD
PROJECT_DIR="$(dirname "${PWD}")" && readonly PROJECT_DIR
readonly RUN_PY="${PROJECT_DIR}/src/train.py"
readonly DS_CONFIG="${PROJECT_DIR}/scripts/ds_config.json" 
readonly DATA_PATH="${PROJECT_DIR}/data/training_data/" # Modify
readonly MODEL_NAME_OR_PATH="${PROJECT_DIR}/llm/LongAlpaca-7B"
readonly TOKENIZER_NAME_OR_PATH="${PROJECT_DIR}/llm/LongAlpaca-7B"
cat <<EOT >"${DS_CONFIG}"
{
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": ${LR},
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
   "bf16": {
    "enabled": "auto",
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 1e9,
    "reduce_scatter": true,
    "reduce_bucket_size": 1e9,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_clipping": "auto",
  "wall_clock_breakdown": false,
  "prescale_gradients": false
}
EOT
echo ${MODEL_NAME_OR_PATH}
PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" deepspeed --include localhost:"${CUDA_VISIBLE_DEVICES}"\
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${RUN_PY}" \
  --tokenizer_path ${TOKENIZER_NAME_OR_PATH} \
  --data_path ${DATA_PATH} \
  --output_dir "${PROJECT_DIR}/data/checkpoints"\
  --num_train_epochs 2 \
  --bf16 True \
  --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
  --per_device_eval_batch_size ${MICRO_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --evaluation_strategy "steps" \
  --eval_num 1 \
  --use_flash_attn True \
  --save_strategy "steps" \
  --save_steps 100 \
  --eval_steps 100 \
  --save_total_limit 5 \
  --learning_rate ${LR} \
  --weight_decay 0. \
  --warmup_steps 20 \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 32768 \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --deepspeed "${PROJECT_DIR}/scripts/ds_config.json" \
  --lr_scheduler_type "cosine"
