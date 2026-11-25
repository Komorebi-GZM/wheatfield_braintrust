"""Run one model (base or +adapter) over a manifest, one record per page.

Design rules for reproducible comparisons:
- every run writes a protocol sidecar next to the output; if a previous sidecar
  exists with a different fingerprint, the run refuses to touch that file so that
  predictions from different configurations can never be mixed;
- results are appended page-by-page with UTF-8 and newline='\n', so an
  interrupted run can be resumed by re-running the same command;
- each record keeps the raw generation *and* the post-processed markdown, plus the
  full evidence block (image hash, prompt hash, grid, tokens, stop reason, ...).

Usage:
    python scripts/infer_test_baseline.py \
        --manifest datasets/handwriting_notes/splits/test.jsonl \
        --output outputs/baseline/test_pred_v2.jsonl \
        --max-pixels 1024000 --max-new-tokens 1024

    python scripts/infer_test_baseline.py \
        --output outputs/finetuned/test_pred_v2.jsonl \
        --adapter outputs/qlora-v1/v2-20261115-114258/checkpoint-4
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.engine import OCREngine, PROTOCOL_VERSION  # noqa: E402
from ocr.prompts import INSTRUCTION  # noqa: E402

DATA_VERSION = Path('datasets/handwriting_notes/data_version_v2.json')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


POSTPROCESS_VERSION = 'v2-normalize'   # wrapper strip + math delimiter unification


def fingerprint(args, engine_desc):
    data_sha = None
    if DATA_VERSION.exists():
        data_sha = json.load(open(DATA_VERSION, encoding='utf-8'))['annotation']['sha256']
    return {
        'protocol_version': PROTOCOL_VERSION,
        'postprocess_version': POSTPROCESS_VERSION,
        'base_model': args.base_model or engine_desc['base_model'],
        'adapter_dir': os.path.abspath(args.adapter) if args.adapter else None,
        'adapter_weights_sha256': (engine_desc.get('adapter') or {}).get('adapter_weights_sha256'),
        'adapter_weights_verified': (engine_desc.get('adapter') or {}).get('weights_verified'),
        'max_pixels': args.max_pixels,
        'max_new_tokens': args.max_new_tokens,
        'repetition_penalty': engine_desc.get('repetition_penalty'),
        'min_new_tokens': engine_desc.get('min_new_tokens'),
        'use_system_prompt': args.use_system_prompt,
        'load_in_4bit': True,
        'instruction': engine_desc.get('instruction', INSTRUCTION),
        'instruction_sha256': engine_desc.get('instruction_sha256')
        or hashlib.sha256(INSTRUCTION.encode('utf-8')).hexdigest(),
        'manifest': str(args.manifest),
        'manifest_sha256': sha256_file(args.manifest),
        'data_version_annotation_sha256': data_sha,
    }


def main():
    ap = argparse.ArgumentParser(description='Run OCR on a fixed manifest (one record per page).')
    ap.add_argument('--manifest', default='datasets/handwriting_notes/splits/test.jsonl')
    ap.add_argument('--output', default='outputs/baseline/test_pred_v2.jsonl')
    ap.add_argument('--adapter', default=None, help='Optional PEFT adapter directory')
    ap.add_argument('--base-model', default=None, help='Optional local or Hub base model')
    ap.add_argument('--max-pixels', type=int, default=1024000)
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--use-system-prompt', action='store_true',
                    help='Off by default: training never used a system message')
    ap.add_argument('--instruction', default=None,
                    help='Override the user-turn instruction (prompt-factor experiments). '
                         'Defaults to the frozen protocol instruction in ocr/prompts.py.')
    ap.add_argument('--repetition-penalty', type=float, default=None,
                    help='Decoding knob. Omit to use the engine default (protocol v4: 1.1); '
                         'pass 1.0 to reproduce the frozen v3 protocol.')
    ap.add_argument('--min-new-tokens', type=int, default=None,
                    help='Decoding knob. Omit to use the engine default (protocol v4: 0, i.e. unset).')
    ap.add_argument('--dry-run', action='store_true', help='Print the plan and exit')
    args = ap.parse_args()

    manifest = Path(args.manifest)
    output = Path(args.output)
    sidecar = output.with_suffix(output.suffix + '.protocol.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]

    engine_kwargs = dict(max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens,
                         use_system_prompt=args.use_system_prompt)
    if args.repetition_penalty is not None:
        engine_kwargs['repetition_penalty'] = args.repetition_penalty
    if args.min_new_tokens is not None:
        engine_kwargs['min_new_tokens'] = args.min_new_tokens
    if args.instruction:
        engine_kwargs['instruction'] = args.instruction
    if args.adapter:
        engine_kwargs['adapter_dir'] = args.adapter
    if args.base_model:
        engine_kwargs['base_model'] = args.base_model
    engine = OCREngine(**engine_kwargs)

    if args.dry_run:
        print(json.dumps({'manifest': str(manifest), 'n_pages': len(rows),
                          'output': str(output), 'engine': engine.describe()},
                         ensure_ascii=False, indent=2))
        return

    engine._load()  # adapter verification happens here
    desc = engine.describe()
    if args.adapter and not (desc.get('adapter') or {}).get('weights_verified'):
        raise SystemExit('REFUSING to run: adapter weights failed verification, see the log above')
    fp = fingerprint(args, desc)

    if sidecar.exists():
        old = json.load(open(sidecar, encoding='utf-8'))
        if old.get('fingerprint') != fp:
            raise SystemExit(
                f'REFUSING to append to {output}: its protocol sidecar has a different '
                f'fingerprint.\n  existing: {json.dumps(old.get("fingerprint"), ensure_ascii=False)}\n'
                f'  current : {json.dumps(fp, ensure_ascii=False)}\n'
                'Use a new --output path for a different configuration.')
    else:
        with open(sidecar, 'w', encoding='utf-8', newline='\n') as f:
            json.dump({'created_at': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
                       'fingerprint': fp, 'engine': desc}, f, ensure_ascii=False, indent=2)

    completed = {}
    if output.exists():
        completed = {json.loads(l)['image'] for l in output.read_text(encoding='utf-8').splitlines() if l.strip()}
        if not completed.issubset({r['image'] for r in rows}):
            raise SystemExit('Existing predictions do not belong to this manifest')

    for row in rows:
        image = row['image']
        if image in completed:
            print(f'skip (already done) {Path(image).name}', flush=True)
            continue
        print(f'Running {Path(image).name} ...', flush=True)
        result = engine.recognize_page(image)
        record = {'image': image, 'markdown': result['markdown'],
                  'raw_output': result.get('raw_output'), 'run': engine.last_run}
        with output.open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        r = engine.last_run
        print(f"  done {Path(image).name}: {r['perf']['elapsed_seconds']:.1f}s "
              f"gen={r['generation']['generated_tokens']}tok finish={r['generation']['finish_reason']} "
              f"grid={r['image']['image_grid_thw']} rep={r['repetition']['detected']}", flush=True)


if __name__ == '__main__':
    main()
