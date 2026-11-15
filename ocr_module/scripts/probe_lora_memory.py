"""Disposable QLoRA step with zero learning rate and no weight save."""
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from swift.model import get_model_processor
from swift.template import get_template

path = snapshot_download('SeerRay-Lab/Xiaomi-OCR-0', local_files_only=True)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
model, processor = get_model_processor(path, model_type='qwen3_5', quantization_config=quant,
                                       device_map={'': 0}, torch_dtype=torch.bfloat16)
template = get_template(processor, template_type='qwen3_5', max_length=8192, max_pixels=1024000)
template.set_mode('train')
template.model = model
samples = [json.loads(line) for line in Path('datasets/handwriting_notes/splits/train_swift.jsonl').read_text(encoding='utf-8').splitlines()]
sample = next(x for x in samples if x['images'][0].endswith('IMG_20261104_200903.jpg'))
encoded = template.encode(sample)
batch = template.data_collator([encoded])
batch = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in batch.items()}
model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05,
                                         target_modules='all-linear', task_type='CAUSAL_LM'))
for name, parameter in model.named_parameters():
    if '.visual.' in name:
        parameter.requires_grad_(False)
model.train()
print('Starting forward', flush=True)
loss = model(**batch).loss
print('Forward loss', float(loss.detach()), flush=True)
loss.backward()
print('Backward complete; trainable tensors with gradient', sum(p.grad is not None for p in model.parameters() if p.requires_grad), flush=True)
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0, weight_decay=0.0)
optimizer.step()
torch.cuda.synchronize()
print('Zero-learning-rate optimizer step completed; state entries', len(optimizer.state), flush=True)
optimizer.zero_grad(set_to_none=True)
model.eval()
validation = [json.loads(line) for line in Path('datasets/handwriting_notes/splits/val_swift.jsonl').read_text(encoding='utf-8').splitlines()]
for row in validation:
    item = template.encode(row)
    val_batch = template.data_collator([item])
    val_batch = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in val_batch.items()}
    with torch.no_grad():
        val_loss = model(**val_batch).loss
    print('Validation', Path(row['images'][0]).name, 'loss', float(val_loss), flush=True)
    del val_batch, val_loss
print('peak allocated bytes', torch.cuda.max_memory_allocated(), 'reserved bytes', torch.cuda.max_memory_reserved(),
      'elapsed seconds', round(time.monotonic() - started, 1), flush=True)
