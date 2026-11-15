"""Small-scale inference diagnosis on NON-test pages (train/val only).

One process, one GPU, cases run serially: the base model is loaded once, the
adapter is attached in-place afterwards, so every comparison changes exactly one
factor at a time.

    python scripts/diagnose_inference.py --cases default

Each case writes outputs/diagnosis/<case_id>.json with the full engine evidence
block plus the raw output, so conclusions can be checked against artifacts.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.engine import OCREngine  # noqa: E402

CKPT = 'outputs/qlora-v1/v2-20261115-114258/checkpoint-4'
# non-test pages only
PAGE_LONG_TRAIN = 'datasets/handwriting_notes/images/IMG_20261104_200903.jpg'   # 611 supervised tokens
PAGE_LONG_VAL = 'datasets/handwriting_notes/images/IMG_20261104_235932.jpg'     # 529 supervised tokens

# id, model, page, max_pixels (None = leave the processor default, i.e. original size), cap
DEFAULT_CASES = [
    ('d1_base_train_1mp_1024', 'base', PAGE_LONG_TRAIN, 1024000, 1024),
    ('d2_ft_train_1mp_1024', 'ft', PAGE_LONG_TRAIN, 1024000, 1024),
    ('d3_ft_train_orig_1024', 'ft', PAGE_LONG_TRAIN, None, 1024),
    ('d4_ft_val_1mp_1024', 'ft', PAGE_LONG_VAL, 1024000, 1024),
]


def run_case(engine, case_id, page, out_dir):
    result = engine.recognize_page(page)
    rec = {'case': case_id, 'image': page, 'markdown': result['markdown'],
           'raw_output': result.get('raw_output'), 'run': engine.last_run}
    p = os.path.join(out_dir, f'{case_id}.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    r = engine.last_run
    print(json.dumps({
        'case': case_id,
        'grid': r['image']['image_grid_thw'],
        'patches': r['image']['n_patches'],
        'visual_tokens': r['image']['n_visual_tokens'],
        'prompt_tokens': r['prompt']['prompt_tokens'],
        'generated_tokens': r['generation']['generated_tokens'],
        'finish': r['generation']['finish_reason'],
        'eos': r['generation']['stopped_on_eos'],
        'rep': r['repetition']['detected'],
        'rep_line_run': r['repetition']['line_run']['max_consecutive'],
        'chars': r['postprocess']['markdown_chars'],
        'wrapper': r['postprocess']['wrapper_stripped'],
        'sec': r['perf']['elapsed_seconds'],
        'gpu_mib': r['perf']['gpu_memory_used_after_mib'],
    }, ensure_ascii=False), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='default')
    ap.add_argument('--out', default='outputs/diagnosis')
    ap.add_argument('--ids', default=None, help='comma separated case ids to run')
    ap.add_argument('--ckpt', default=CKPT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wanted = set(args.ids.split(',')) if args.ids else None
    cases = [c for c in DEFAULT_CASES if (wanted is None or any(c[0].startswith(w) for w in wanted))]

    # group by max_pixels so the pixel budget is fixed per engine instance
    import torch
    for mp in sorted({c[3] for c in cases}, key=lambda x: (x is None, x)):
        subset = [c for c in cases if c[3] == mp]
        engine = OCREngine(max_pixels=mp if mp else 16_777_216, max_new_tokens=max(c[4] for c in subset))
        # base first (no adapter) so the same process can serve base and ft cases
        print(f'--- loading base (max_pixels={mp})', flush=True)
        engine._load()
        for case_id, model, page, _, cap in subset:
            if model != 'base':
                continue
            engine.max_new_tokens = cap
            run_case(engine, case_id, page, args.out)
        ft = [c for c in subset if c[1] == 'ft']
        if ft:
            print('--- attaching adapter', flush=True)
            meta = engine.attach_adapter(args.ckpt)
            print(json.dumps(meta, ensure_ascii=False), flush=True)
            for case_id, model, page, _, cap in ft:
                engine.max_new_tokens = cap
                run_case(engine, case_id, page, args.out)
        del engine
        torch.cuda.empty_cache()
    print('done')


if __name__ == '__main__':
    main()
