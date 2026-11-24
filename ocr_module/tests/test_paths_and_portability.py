"""Portability regressions: Windows paths in frozen manifests + OTSL cells."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.paths import image_id, resolve_image_path  # noqa: E402
from ocr.postprocess import otsl_to_markdown  # noqa: E402


def test_image_id_handles_both_separators():
    assert image_id(r"E:\ocr\page1.jpg") == "page1.jpg"
    assert image_id("/tmp/ocr/page2.jpg") == "page2.jpg"
    assert image_id("page3.jpg") == "page3.jpg"


def test_resolve_image_path_uses_manifest_directory(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    page = images / "page.jpg"
    page.write_bytes(b"x")
    manifest = tmp_path / "splits" / "test.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("", encoding="utf-8")
    # Historical manifest stored a Windows absolute path; the local copy must win.
    assert resolve_image_path(r"E:\old\images\page.jpg", manifest=manifest) == str(page)


def test_resolve_image_path_keeps_missing_path_transparent(tmp_path):
    missing = str(tmp_path / "nope.jpg")
    assert resolve_image_path(missing) == missing


def test_otsl_leading_cell_token_does_not_add_empty_column():
    md = otsl_to_markdown("<fcel>量<fcel>符号<nl><fcel>磁通<fcel>Φ<nl>")
    assert md.splitlines()[0] == "| 量 | 符号 |"
    assert "| 磁通 | Φ |" in md


def test_evaluate_matches_windows_and_posix_page_ids(tmp_path):
    ref = tmp_path / "ref.jsonl"
    pred = tmp_path / "pred.jsonl"
    ref.write_text(json.dumps({"image": "splits/page.jpg", "markdown": "abc"}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    pred.write_text(json.dumps({"image": r"E:\old\splits\page.jpg", "markdown": "abc"}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    subprocess.run([sys.executable, str(script), "--ref", str(ref), "--pred", str(pred), "--out", str(out)],
                   check=True, capture_output=True, text=True)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_pages"] == 1
    assert metrics["text_cer"] == 0
