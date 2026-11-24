import json
import subprocess
import sys
from pathlib import Path


def test_nonempty_unreviewed_annotations_are_rejected(tmp_path):
    for i in range(3):
        (tmp_path / f'page{i}.jpg').write_bytes(bytes([i]))
    ann = tmp_path / 'real.jsonl'
    ann.write_text(''.join(json.dumps({'image': f'page{i}.jpg', 'markdown': 'text'}) + '\n' for i in range(3)), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_dataset.py'
    result = subprocess.run([sys.executable, str(script), '--real-dir', str(tmp_path), '--real-ann', str(ann), '--synth-manifest', str(tmp_path / 'missing'), '--out-dir', str(tmp_path / 'out'), '--val-pages', '1', '--test-pages', '1'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'review' in result.stderr.lower()
