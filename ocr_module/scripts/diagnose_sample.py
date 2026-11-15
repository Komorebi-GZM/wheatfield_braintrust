"""Sample the whole non-test split with both models under one fixed protocol.

Purpose: answer "does the generation cap cover a normal full page?" and "how often
does each model degenerate?" using ONLY train/val pages (never the test set), so
the cap is chosen from non-test evidence.

Both models run through the identical engine, processor, prompt and cap; the only
difference is whether the LoRA adapter is attached. Base is loaded once and the
adapter is attached in place, so nothing else can drift between the two arms.

Usage:
    python scripts/diagnose_sample.py --split val --models base,ft \
        --max-pixels 1024000 --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.engine import OCREngine  # noqa: E402

CKPT = 'outputs/qlora-v1/v2-20261115-114258/checkpoint-4'


def summarise(arm, page, r, ref_tokens):
    g, im, rep, pp = r['generation'], r['image'], r['repetition'], r['postprocess']
    return {
        'arm': arm,
        'page': os.path.basename(page),
        'image': page,
        'ref_supervised_tokens': ref_tokens,
        'grid': im['image_grid_thw'],
        'n_patches': im['n_patches'],
        'n_visual_tokens': im['n_visual_tokens'],
        'prompt_tokens': r['prompt']['prompt_tokens'],
        'max_new_tokens': g['max_new_tokens'],
        'generated_tokens': g['generated_tokens'],
        'finish': g['finish_reason'],
        'stopped_on_eos': g['stopped_on_eos'],
        'hit_cap': g['hit_length_cap'],
        'rep_detected': rep['detected'],
        'rep_line_run': rep['line_run']['max_consecutive'],
        'rep_template_run': rep['template_line_run']['max_consecutive'],
        'rep_ngram_coverage': rep['ngram']['coverage'],
        'chars': pp['markdown_chars'],
        'wrapper': pp['wrapper_stripped'],
        'sec': round(r['perf']['elapsed_seconds'], 1),
        'gpu_mib': r['perf']['gpu_memory_used_after_mib'],
        'peak_alloc_mib': (round(r['perf']['peak_cuda_allocated_bytes'] / 2**20, 1)
                           if r['perf']['peak_cuda_allocated_bytes'] else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['train', 'val'])
    ap.add_argument('--models', default='base,ft')
    ap.add_argument('--max-pixels', type=int, default=1024000)
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--out', default='outputs/diagnosis')
    ap.add_argument('--limit', type=int, default=0, help='0 = all pages in the split')
    args = ap.parse_args()

    manifest = Path(f'datasets/handwriting_notes/splits/{args.split}.jsonl')
    rows = [json.loads(l) for l in manifest.read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    sup = json.load(open('outputs/protocol/supervised_tokens.json', encoding='utf-8'))['per_page']
    arms = [a.strip() for a in args.models.split(',') if a.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    engine = OCREngine(max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens)
    print(f'--- loading base (max_pixels={args.max_pixels}, cap={args.max_new_tokens})', flush=True)
    engine._load()

    def run_arm(arm):
        for row in rows:
            page = row['image']
            ref_tok = sup.get(os.path.basename(page), {}).get('supervised_tokens')
            result = engine.recognize_page(page)
            rec = {'case': f'{args.split}_{arm}_{Path(page).stem}', 'arm': arm,
                   'image': page, 'markdown': result['markdown'],
                   'raw_output': result.get('raw_output'), 'run': engine.last_run}
            p = out_dir / f'{rec["case"]}.json'
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
            s = summarise(arm, page, engine.last_run, ref_tok)
            results.append(s)
            print(json.dumps(s, ensure_ascii=False), flush=True)

    if 'base' in arms:
        run_arm('base')
    if 'ft' in arms:
        print('--- attaching adapter', flush=True)
        meta = engine.attach_adapter(args.ckpt)
        print('weights_verified =', meta['weights_verified'], flush=True)
        if not meta['weights_verified']:
            raise SystemExit('adapter verification failed; aborting before any generation')
        run_arm('ft')

    summary = {
        'split': args.split,
        'max_pixels': args.max_pixels,
        'max_new_tokens': args.max_new_tokens,
        'adapter': os.path.abspath(args.ckpt) if 'ft' in arms else None,
        'n_pages': len(rows),
        'results': results,
    }
    sp = out_dir / f'summary_{args.split}_{args.max_pixels}_{args.max_new_tokens}.json'
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n=== summary ===')
    hdr = ('arm', 'page', 'ref_tok', 'gen', 'finish', 'rep', 'tmpl', 'chars', 'sec')
    print('{:>5} {:<24} {:>7} {:>5} {:>6} {:>4} {:>4} {:>6} {:>6}'.format(*hdr))
    for s in results:
        print('{:>5} {:<24} {:>7} {:>5} {:>6} {:>4} {:>4} {:>6} {:>6}'.format(
            s['arm'], s['page'], str(s['ref_supervised_tokens']), s['generated_tokens'],
            s['finish'], int(s['rep_detected']), s['rep_template_run'], s['chars'], s['sec']))
    print('\nwritten', sp)


if __name__ == '__main__':
    main()
