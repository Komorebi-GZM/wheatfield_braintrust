#!/usr/bin/env bash
# QLoRA fine-tuning entry (Windows / WSL / Linux). Requires ms-swift installed.
# Adjust env vars for your host. This file does NOT download anything by itself,
# but `swift sft` will pull the base model the first time — run it only when ready.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-SeerRay-Lab/Xiaomi-OCR-0}"   # fallback: Qwen/Qwen2.5-VL-3B-Instruct
TRAIN_DATA="${TRAIN_DATA:-data/splits/train_swift.jsonl}"
VAL_DATA="${VAL_DATA:-data/splits/val.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/qlora-v1}"
TEMPLATE="${TEMPLATE:-qwen2_5_vl}"                     # if base unsupported, try qwen2_vl
MAX_PIXELS="${MAX_PIXELS:-1024000}"                    # ~1024^2 long side budget

# NOTE: val.jsonl is reference-format; for a train-time eval set you can export
# a swift-format val too (see prepare_dataset.py) and pass it instead.

swift sft \
  --model "${BASE_MODEL}" \
  --template "${TEMPLATE}" \
  --train_type lora \
  --quant_method bnb --quant_bits 4 \
  --torch_dtype bfloat16 \
  --dataset "${TRAIN_DATA}" \
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
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --output_dir "${OUT_DIR}" \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --dataloader_num_workers 2

echo "[train_qlora] done -> ${OUT_DIR}"
echo "[train_qlora] LoRA adapter is under ${OUT_DIR} (push as git-lfs/HF, not plain git)"
