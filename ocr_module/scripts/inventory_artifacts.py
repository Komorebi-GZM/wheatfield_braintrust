"""Inventory predictions + training artifacts into a single JSON report."""
import json, os, sys, hashlib, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr.paths import image_id  # noqa: E402

RUN = 'outputs/qlora-v1/v2-20261115-114258'
TEST = 'datasets/handwriting_notes/splits/test.jsonl'
PRED_FILES = [
    'outputs/baseline/test_pred.jsonl',
    'outputs/baseline/test_pred_512.jsonl',
    'outputs/finetuned/test_pred_512.jsonl',
]


def jl(p):
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    test = jl(TEST)
    test_set = {image_id(r['image']) for r in test}
    ref_len = {image_id(r['image']): len(r['markdown']) for r in test}
    ref_sq = {image_id(r['image']): r['markdown'].count('□') for r in test}

    preds = {}
    store = {}
    for p in PRED_FILES:
        rows = jl(p)
        if rows is None:
            preds[p] = {'exists': False}
            continue
        imgs = [image_id(r['image']) for r in rows]
        store[p] = {i: r['markdown'] for i, r in zip(imgs, rows)}
        preds[p] = {
            'exists': True,
            'sha256': sha(p),
            'bytes': os.path.getsize(p),
            'n_lines': len(rows),
            'n_unique_images': len(set(imgs)),
            'duplicate_images': sorted({i for i in imgs if imgs.count(i) > 1}),
            'missing_pages': sorted(test_set - set(imgs)),
            'extra_pages': sorted(set(imgs) - test_set),
            'complete': sorted(set(imgs)) == sorted(test_set),
            'per_page': [{
                'image': i,
                'pred_chars': len(r['markdown']),
                'ref_chars': ref_len.get(i),
                'ref_unknown_marker': ref_sq.get(i),
                'elapsed_seconds': round(r.get('run', {}).get('elapsed_seconds', 0), 1),
                'original_size_wh': r.get('run', {}).get('original_size_wh'),
                'image_grid_thw': r.get('run', {}).get('image_grid_thw'),
            } for i, r in zip(imgs, rows)],
            'run_fields_recorded': sorted(rows[0].get('run', {}).keys()),
            'run_fields_missing': ['max_new_tokens', 'finish_reason'],
        }

    # identical-output comparison between the two baseline runs
    a, b = store.get(PRED_FILES[0]), store.get(PRED_FILES[1])
    cmp = {}
    if a and b:
        for k in sorted(set(a) | set(b)):
            cmp[k] = {'identical': a.get(k) == b.get(k),
                      'chars_4096': len(a.get(k, '')),
                      'chars_512': len(b.get(k, '')),
                      'tail_4096': a.get(k, '')[-40:],
                      'tail_512': b.get(k, '')[-40:]}

    ckpts = {}
    for name in sorted(os.listdir(RUN)):
        d = os.path.join(RUN, name)
        if not os.path.isdir(d) or not name.startswith('checkpoint'):
            continue
        files = {f: os.path.getsize(os.path.join(d, f)) for f in sorted(os.listdir(d))}
        cfg = json.load(open(os.path.join(d, 'adapter_config.json'), encoding='utf-8'))
        st = json.load(open(os.path.join(d, 'trainer_state.json'), encoding='utf-8'))
        ckpts[name] = {
            'files': files,
            'lora': {k: cfg.get(k) for k in ['r', 'lora_alpha', 'lora_dropout', 'peft_type', 'task_type']},
            'target_modules': cfg.get('target_modules'),
            'base_model_snapshot': cfg.get('base_model_name_or_path'),
            'global_step': st.get('global_step'),
            'epoch': st.get('epoch'),
            'best_metric': st.get('best_metric'),
            'usable_for_inference': os.path.exists(os.path.join(d, 'adapter_model.safetensors')),
        }

    args = json.load(open(os.path.join(RUN, 'args.json'), encoding='utf-8'))
    keys = ['model_type', 'template', 'dataset', 'val_dataset', 'split_dataset_ratio',
            'num_train_epochs', 'per_device_train_batch_size', 'gradient_accumulation_steps',
            'learning_rate', 'lr_scheduler_type', 'warmup_ratio', 'warmup_steps',
            'max_length', 'max_pixels', 'lora_rank', 'lora_alpha', 'lora_dropout',
            'quant_method', 'quant_bits', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant',
            'freeze_vit', 'freeze_aligner', 'gradient_checkpointing', 'bf16',
            'dataloader_num_workers', 'save_strategy', 'eval_strategy', 'save_total_limit',
            'seed', 'load_best_model_at_end']
    train_args = {k: args.get(k) for k in keys}

    log = jl(os.path.join(RUN, 'logging.jsonl'))
    summary = next((x for x in log if 'train_runtime' in x), log[-1] if log else {})
    tail = log[-1] if log else {}
    train_log = [x for x in log if 'loss' in x]
    eval_log = [x for x in log if 'eval_loss' in x]

    eval_dir = 'outputs/baseline/eval'
    metrics = json.load(open(os.path.join(eval_dir, 'metrics.json'), encoding='utf-8')) \
        if os.path.exists(os.path.join(eval_dir, 'metrics.json')) else None

    doc = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'checked_by': '脚本扫描 + 人工核对（未运行模型）',
        'test_manifest': {'file': TEST, 'n_pages': len(test),
                          'pages': [image_id(r['image']) for r in test]},
        'predictions': preds,
        'baseline_4096_vs_512': cmp,
        'training': {
            'run_dir': RUN,
            'exists': os.path.exists(RUN),
            'checkpoints': ckpts,
            'best_checkpoint_recorded': image_id((tail or summary).get('best_model_checkpoint', '')),
            'last_checkpoint_alias_present': os.path.exists(os.path.join(RUN, 'last-checkpoint')),
            'train_args': train_args,
            'log': {
                'n_entries': len(log),
                'train_loss_steps': [{'step': e.get('global_step/max_steps'), 'loss': e.get('loss'),
                                      'token_acc': e.get('token_acc'), 'lr': e.get('learning_rate'),
                                      'memory_GiB': e.get('memory(GiB)')} for e in train_log],
                'eval_loss_steps': [{'step': e.get('global_step/max_steps'), 'eval_loss': e.get('eval_loss'),
                                     'eval_token_acc': e.get('eval_token_acc')} for e in eval_log],
                'train_runtime_s': summary.get('train_runtime'),
                'train_loss_final': summary.get('train_loss'),
                'peak_memory_GiB': (tail or summary).get('memory'),
                'param_info': (tail or {}).get('model_parameter_info'),
            },
            'empty_run_dirs': [d for d in ['outputs/qlora-v1/v0-20261115-114149',
                                           'outputs/qlora-v1/v1-20261115-114219']
                               if os.path.isdir(d) and not os.listdir(d)],
        },
        'evaluation': {
            'baseline_metrics_file': os.path.join(eval_dir, 'metrics.json'),
            'baseline_metrics': metrics,
            'finetuned_eval_dir_exists': os.path.isdir('outputs/finetuned/eval'),
            'note': 'metrics.json 对应 outputs/baseline/test_pred.jsonl（4096 上限）；微调版无任何评测产物',
        },
        'process_check': {
            'python_processes': 0,
            'gpu_used_MiB_at_check': 1354,
            'gpu_total_MiB': 8188,
            'note': '核对时无 python 训练/推理进程；显存占用来自桌面等其它程序，未终止任何进程',
        },
        'verdict': {
            'baseline_complete_pages': 5,
            'finetuned_complete_pages': 1,
            'finetuned_scorable_pages': 0,
            'comparison_possible': False,
            'blocking_reason': '微调版只有 IMG_20261105_000020（参考含 28 个□，评测整页排除），4 个可评分页无预测；evaluate.py 对缺页直接抛错',
        },
    }
    out = 'artifacts_inventory_v2.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(json.dumps({'out': out,
                      'preds': {k: (v.get('n_lines'), v.get('complete')) for k, v in preds.items()},
                      'checkpoints': {k: v['global_step'] for k, v in ckpts.items()},
                      'comparison_possible': doc['verdict']['comparison_possible']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
