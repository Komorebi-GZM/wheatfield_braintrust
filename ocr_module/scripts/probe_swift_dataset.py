import json
from pathlib import Path
from huggingface_hub import snapshot_download
from swift.model import get_model_processor
from swift.template import get_template

model_path = snapshot_download('SeerRay-Lab/Xiaomi-OCR-0', local_files_only=True)
_, processor = get_model_processor(model_path, model_type='qwen3_5', load_model=False)
template = get_template(processor, template_type='qwen3_5', max_length=8192, max_pixels=1024000)
template.set_mode('train')
for split in ('train', 'val'):
    path = Path(f'datasets/handwriting_notes/splits/{split}_swift.jsonl')
    for line in path.read_text(encoding='utf-8').splitlines():
        sample = json.loads(line)
        encoded = template.encode(sample)
        print(split, Path(sample['images'][0]).name,
              'grid', encoded['image_grid_thw'].tolist(),
              'tokens', len(encoded['input_ids']),
              'supervised', sum(x != -100 for x in encoded['labels']), flush=True)
