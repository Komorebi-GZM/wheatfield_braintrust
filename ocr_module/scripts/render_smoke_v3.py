"""Render the single-page smoke acceptance record from real run artifacts.

Consumes the jsonl produced by scripts/infer_test_baseline.py (which stores the
full `run` evidence block per page) and does the itemised checks that the
acceptance checklist asks for: text / headings / lists / formulas, plus UTF-8
round-trip, original vs processed size, timing and memory.

No model is loaded here; it only reads what the runs already wrote.

Usage:
    python scripts/render_smoke_v3.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.postprocess import extract_formulas, normalize_output  # noqa: E402
from scripts.evaluate import _heading_levels, cer, plain_text  # noqa: E402

ARMS = [
    ("base", "基座（无 adapter）", "outputs/smoke_v3/base_235955.jsonl"),
    ("ft", "checkpoint-26", "outputs/smoke_v3/ft_235955.jsonl"),
]
REF = "datasets/handwriting_notes/splits/val.jsonl"


def read_one(path):
    with open(path, "rb") as f:
        raw = f.read()
    rec = json.loads(raw.decode("utf-8").strip())
    rec["_bytes"] = len(raw)
    rec["_decode_ok"] = True
    rec["_replacement_chars"] = rec["markdown"].count("\ufffd")
    return rec


def list_items(md):
    return [l.strip() for l in md.splitlines() if l.strip().startswith("- ")]


def main():
    ref = {}
    for line in open(REF, encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            ref[os.path.basename(it["image"])] = it["markdown"]

    out = {"page": None, "arms": {}}
    for key, label, path in ARMS:
        rec = read_one(path)
        name = os.path.basename(rec["image"])
        ref_md = ref[name]
        ref_n, _ = normalize_output(ref_md)
        hyp_n, _ = normalize_output(rec["markdown"])
        run = rec["run"]
        img, gen, perf, pp = run["image"], run["generation"], run["perf"], run["postprocess"]

        f_ref, f_hyp = extract_formulas(ref_n), extract_formulas(hyp_n)
        out["page"] = name
        out["arms"][key] = {
            "label": label,
            "prediction_file": path,
            "input": {
                "image_path": run["image_path"],
                "image_sha256": img["sha256"],
                "original_size_wh": img["original_size_wh"],
                "processed_size_wh": img["processed_size_wh"],
                "max_pixels": img["max_pixels"],
                "image_grid_thw": img["image_grid_thw"],
                "patch_size": img["patch_size"],
                "merge_size": img["merge_size"],
                "n_patches": img["n_patches"],
                "n_visual_tokens": img["n_visual_tokens"],
                "prompt_tokens": run["prompt"]["prompt_tokens"],
                "n_image_pad_tokens": run["prompt"]["n_image_pad_tokens"],
                "prompt_sha256": run["prompt"]["prompt_sha256"],
                "system_prompt_used": run["prompt"]["system_prompt_used"],
            },
            "utf8": {
                "file_bytes": rec["_bytes"],
                "markdown_chars": len(rec["markdown"]),
                "decode_ok": rec["_decode_ok"],
                "replacement_chars": rec["_replacement_chars"],
                "readable_markdown": len(rec["markdown"]) > 0 and rec["_replacement_chars"] == 0,
                "wrapper_stripped": pp["wrapper_stripped"],
            },
            "generation": {
                "generated_tokens": gen["generated_tokens"],
                "max_new_tokens": gen["max_new_tokens"],
                "finish_reason": gen["finish_reason"],
                "hit_length_cap": gen["hit_length_cap"],
                "last_token_id": gen["last_token_id"],
                "eos_token_id": gen["eos_token_id"],
                "repetition": run["repetition"]["detected"],
                "rep_rules_flagged": [k for k, v in run["repetition"].items()
                                      if isinstance(v, dict) and v.get("flagged")],
            },
            "perf": {
                "elapsed_seconds": perf["elapsed_seconds"],
                "peak_cuda_allocated_mib": round(perf["peak_cuda_allocated_bytes"] / 2**20, 1),
                "peak_cuda_reserved_mib": round(perf["peak_cuda_reserved_bytes"] / 2**20, 1),
                "gpu_card_used_mib_before": perf["gpu_memory_used_before_mib"],
                "gpu_card_used_mib_after": perf["gpu_memory_used_after_mib"],
                "note": perf["gpu_memory_note"],
            },
            "checks": {
                "text": {
                    "ref_plain_chars": len(plain_text(ref_n)),
                    "hyp_plain_chars": len(plain_text(hyp_n)),
                    "text_cer": round(cer(plain_text(ref_n), plain_text(hyp_n)), 4),
                    "ref_head": plain_text(ref_n)[:70],
                    "hyp_head": plain_text(hyp_n)[:70],
                },
                "headings": {
                    "ref_count": len(_heading_levels(ref_n)),
                    "hyp_count": len(_heading_levels(hyp_n)),
                    "ref_levels": _heading_levels(ref_n)[:12],
                    "hyp_levels": _heading_levels(hyp_n)[:12],
                    "match": _heading_levels(ref_n) == _heading_levels(hyp_n),
                },
                "lists": {
                    "ref_count": len(list_items(ref_n)),
                    "hyp_count": len(list_items(hyp_n)),
                    "hyp_sample": list_items(hyp_n)[:3],
                },
                "formulas": {
                    "ref_count": len(f_ref),
                    "hyp_count": len(f_hyp),
                    "extractable": len(f_hyp) > 0,
                    "ref_sample": f_ref[:2],
                    "hyp_sample": f_hyp[:2],
                },
            },
            "markdown": rec["markdown"],
            "raw_output": rec.get("raw_output"),
        }

    os.makedirs("outputs/smoke_v3", exist_ok=True)
    with open("outputs/smoke_v3/smoke_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    L = [f"# 单页冒烟（当前入口）× 两臂：{out['page']}\n",
         "页面为验证集页，含标题/列表/公式，用于逐项检查；不参与任何选择。\n"]
    for key, _label, _p in ARMS:
        a = out["arms"][key]
        i, u, g, p, c = a["input"], a["utf8"], a["generation"], a["perf"], a["checks"]
        L.append(f"## {key} — {a['label']}\n")
        L.append("| 项 | 值 |")
        L.append("|---|---|")
        L.append(f"| 原图 | `{i['image_path']}` |")
        L.append(f"| 原图 sha256 | `{i['image_sha256'][:32]}…` |")
        L.append(f"| 原始尺寸 (W×H) | {i['original_size_wh'][0]}×{i['original_size_wh'][1]} |")
        L.append(f"| 处理器实际尺寸 | {i['processed_size_wh'][0]}×{i['processed_size_wh'][1]} |")
        L.append(f"| max_pixels | {i['max_pixels']} |")
        L.append(f"| image_grid_thw | {i['image_grid_thw']} |")
        L.append(f"| patch / merge | {i['patch_size']} / {i['merge_size']} |")
        L.append(f"| patch 数 / 视觉 token 数 | {i['n_patches']} / {i['n_visual_tokens']} |")
        L.append(f"| prompt token（其中 image_pad） | {i['prompt_tokens']} / {i['n_image_pad_tokens']} |")
        L.append(f"| system prompt 使用 | {i['system_prompt_used']} |")
        L.append(f"| prompt sha256 | `{i['prompt_sha256'][:32]}…` |")
        L.append(f"| UTF-8 解码 | {'成功' if u['decode_ok'] else '失败'}，替换字符 {u['replacement_chars']} 个 |")
        L.append(f"| markdown 字符数 / 文件字节 | {u['markdown_chars']} / {u['file_bytes']} |")
        L.append(f"| 剥除的外层包装 | {u['wrapper_stripped']} |")
        L.append(f"| 生成 token / 上限 | {g['generated_tokens']} / {g['max_new_tokens']} |")
        L.append(f"| 结束原因 | {g['finish_reason']}（末 token {g['last_token_id']}，EOS={g['eos_token_id']}） |")
        L.append(f"| 重复检测 | {g['repetition']}，触发规则 {g['rep_rules_flagged']} |")
        L.append(f"| 耗时 | {p['elapsed_seconds']:.1f} s |")
        L.append(f"| 进程峰值 allocated / reserved | {p['peak_cuda_allocated_mib']} / {p['peak_cuda_reserved_mib']} MiB |")
        L.append(f"| 整卡占用 前→后 | {p['gpu_card_used_mib_before']} → {p['gpu_card_used_mib_after']} MiB |")
        L.append("")
        L.append("逐项检查：\n")
        L.append("| 元素 | 参考 | 输出 | 结论 |")
        L.append("|---|---|---|---|")
        L.append(f"| 文字 | {c['text']['ref_plain_chars']} 字符 | {c['text']['hyp_plain_chars']} 字符，CER {c['text']['text_cer']} | "
                 f"{'有输出' if c['text']['hyp_plain_chars'] else '无输出'}，可读 |")
        L.append(f"| 标题 | {c['headings']['ref_count']} 个 {c['headings']['ref_levels']} | "
                 f"{c['headings']['hyp_count']} 个 {c['headings']['hyp_levels']} | "
                 f"{'层级序列一致' if c['headings']['match'] else '不一致'}"
                 f"{'' if c['headings']['hyp_count'] else '；输出无标题，无法评价'}"+" |")
        L.append(f"| 列表 | {c['lists']['ref_count']} 项 | {c['lists']['hyp_count']} 项 | "
                 f"{'有列表结构' if c['lists']['hyp_count'] else '输出无列表，无法评价'} |")
        L.append(f"| 公式 | {c['formulas']['ref_count']} 个 | {c['formulas']['hyp_count']} 个 | "
                 f"{'可被提取器识别' if c['formulas']['extractable'] else '未提取到格式化公式'} |")
        L.append("")
        L.append(f"文字开头对照：参考 `{c['text']['ref_head']}` ／ 输出 `{c['text']['hyp_head']}`\n")
        if c["formulas"]["hyp_sample"]:
            L.append(f"输出公式样例：`{c['formulas']['hyp_sample']}`\n")
        if c["lists"]["hyp_sample"]:
            L.append(f"输出列表示例：`{c['lists']['hyp_sample'][0][:80]}`\n")

    with open("outputs/smoke_v3/smoke_v3.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print("\nWROTE outputs/smoke_v3/smoke_v3.{json,md}")


if __name__ == "__main__":
    main()
