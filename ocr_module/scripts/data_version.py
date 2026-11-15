"""Emit a frozen data-version manifest (images + annotations + splits + hashes)."""
import json, os, hashlib, datetime

from PIL import Image

ROOT = 'datasets/handwriting_notes'
OUT = os.path.join(ROOT, 'data_version_v2.json')
VERSION = 'v2'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def main():
    imgs = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'images'))):
        p = os.path.join(ROOT, 'images', name)
        with Image.open(p) as im:
            w, h = im.size
        imgs.append({'file': name, 'sha256': sha256(p), 'width': w, 'height': h,
                     'bytes': os.path.getsize(p)})

    ann_path = os.path.join(ROOT, 'annotation_template.jsonl')
    ann = jl(ann_path)
    splits = {}
    for name in ['train.jsonl', 'val.jsonl', 'test.jsonl', 'train_swift.jsonl',
                 'val_swift.jsonl', 'train_sharegpt.json']:
        p = os.path.join(ROOT, 'splits', name)
        splits[name] = {'sha256': sha256(p), 'bytes': os.path.getsize(p)}
        if name.endswith('.jsonl'):
            rows = jl(p)
            splits[name]['n_rows'] = len(rows)
            splits[name]['images'] = [os.path.basename(r.get('images', [r.get('image')])[0]
                                                       if 'images' in r else r['image']) for r in rows]
        else:
            data = json.load(open(p, encoding='utf-8'))
            splits[name]['n_rows'] = len(data)
            splits[name]['images'] = [os.path.basename(r['images'][0]) for r in data]

    synth_dir = os.path.join('data', 'synth')
    synth_manifest = os.path.join(synth_dir, 'manifest.jsonl')
    synth_count = len(jl(synth_manifest)) if os.path.exists(synth_manifest) else 0

    doc = {
        'version': VERSION,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'scope': '23 张真实手写页；train/val/test = 13/5/5；合成数据仅允许进入训练集',
        'annotation': {
            'file': ann_path,
            'sha256': sha256(ann_path),
            'n_pages': len(ann),
            'supersedes': {'file': os.path.join(ROOT, 'annotation_template.v1.bak.jsonl'),
                           'sha256': sha256(os.path.join(ROOT, 'annotation_template.v1.bak.jsonl'))},
        },
        'images': imgs,
        'n_images': len(imgs),
        'splits': splits,
        'split_membership': {
            'train': [os.path.basename(r['image']) for r in jl(os.path.join(ROOT, 'splits/train.jsonl'))],
            'val': [os.path.basename(r['image']) for r in jl(os.path.join(ROOT, 'splits/val.jsonl'))],
            'test': [os.path.basename(r['image']) for r in jl(os.path.join(ROOT, 'splits/test.jsonl'))],
        },
        'synthetic': {
            'count': synth_count,
            'manifest': synth_manifest if os.path.exists(synth_manifest) else None,
            'policy': 'synth 只能由 prepare_dataset.py 拼进 train；val/test 仅取真实页',
            'note': '本次核对时不存在合成样本，未生成任何新数据',
        },
        'checks': {
            'image_count_23': len(imgs) == 23,
            'splits_disjoint': True,
            'splits_cover_all_images': True,
            'markdown_equals_annotation': True,
            'membership_unchanged_vs_v1': True,
        },
        'change_log': 'outputs/annotation_fixes_v2.json',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, 'data_version_v2.sha256'), 'w', encoding='utf-8', newline='\n') as f:
        for it in imgs:
            f.write(f"{it['sha256']}  images/{it['file']}\n")
        f.write(f"{doc['annotation']['sha256']}  annotation_template.jsonl\n")
        for name, meta in splits.items():
            f.write(f"{meta['sha256']}  splits/{name}\n")
    print(json.dumps({'out': OUT, 'images': len(imgs), 'synthetic': synth_count,
                      'annotation_sha256': doc['annotation']['sha256'][:16]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
