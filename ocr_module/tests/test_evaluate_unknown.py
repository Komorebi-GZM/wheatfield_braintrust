import json
import subprocess
import sys
from pathlib import Path


def test_unknown_reference_is_excluded_but_accounted_for(tmp_path):
    ref = tmp_path / 'ref.jsonl'
    pred = tmp_path / 'pred.jsonl'
    refs = [{'image': 'clear', 'markdown': 'abc'}, {'image': 'blur', 'markdown': 'a□□z'}]
    preds = [{'image': 'clear', 'markdown': 'abc'}, {'image': 'blur', 'markdown': 'unreadable guess'}]
    for path, rows in ((ref, refs), (pred, preds)):
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    out = tmp_path / 'out'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate.py'
    subprocess.run([sys.executable, str(script), '--ref', str(ref), '--pred', str(pred), '--out', str(out)], check=True)
    metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['n_pages'] == 2
    assert metrics['n_scored_pages'] == 1
    assert metrics['text_cer'] == 0
    assert metrics['per_page'][1]['excluded_reason'] == 'unknown_reference'
    assert metrics['per_page'][1]['text_cer'] is None


def test_missing_prediction_fails_instead_of_scoring_empty(tmp_path):
    ref = tmp_path / 'ref.jsonl'
    pred = tmp_path / 'pred.jsonl'
    ref.write_text(json.dumps({'image': 'page', 'markdown': 'abc'}) + '\n', encoding='utf-8')
    pred.write_text('', encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate.py'
    result = subprocess.run([sys.executable, str(script), '--ref', str(ref), '--pred', str(pred), '--out', str(tmp_path / 'out')], capture_output=True)
    assert result.returncode != 0
