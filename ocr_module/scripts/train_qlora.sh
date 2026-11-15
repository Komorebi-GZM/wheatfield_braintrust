#!/usr/bin/env bash
# QLoRA fine-tuning entry (Windows / WSL / Linux). Requires ms-swift installed.
# Adjust env vars for your host. This file does NOT download anything by itself,
# but `swift sft` will pull the base model the first time — run it only when ready.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-SeerRay-Lab/Xiaomi-OCR-0}"
TRAIN_DATA="${TRAIN_DATA:-datasets/handwriting_notes/splits/train_swift.jsonl}"
VAL_DATA="${VAL_DATA:-datasets/handwriting_notes/splits/val_swift.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/qlora-v1}"
TEMPLATE="${TEMPLATE:-qwen3_5}"
MAX_PIXELS="${MAX_PIXELS:-1024000}"                    # ~1024^2 long side budget

# Validation uses the Swift-format export of the fixed real-page split.

swift sft \
  --model "${BASE_MODEL}" \
  --model_type qwen3_5 \
  --template "${TEMPLATE}" \
  --tuner_type lora \
  --quant_method bnb --quant_bits 4 \
  --torch_dtype bfloat16 \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --max_length 8192 \
  --max_pixels "${MAX_PIXELS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --gradient_checkpointing true \
  --freeze_vit true \
  --logging_steps 1 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --save_total_limit 2 \
  --output_dir "${OUT_DIR}" \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --dataloader_num_workers 2

echo "[train_qlora] done -> ${OUT_DIR}"
echo "[train_qlora] LoRA adapter is under ${OUT_DIR} (push as git-lfs/HF, not plain git)"
