"""Run OCR on a single page image. Example:

    python scripts/infer_one.py data/raw/page1.jpg --adapter outputs/qlora-v1

Prints the structured result JSON (and optionally the markdown).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.engine import OCREngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--base", default="SeerRay-Lab/Xiaomi-OCR-0")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--show-markdown", action="store_true")
    args = ap.parse_args()

    eng = OCREngine(base_model=args.base, adapter_dir=args.adapter,
                    load_in_4bit=not args.no_4bit)
    res = eng.recognize_page(args.image)
    if args.show_markdown:
        print(res.pop("markdown"))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
