"""Build the frozen side-by-side comparison from two prediction files + two metrics.json.

Inputs are produced by scripts/infer_test_baseline.py (predictions with a `run`
evidence block per page) and scripts/evaluate.py (same evaluator version for both
sides). Nothing here re-implements scoring: CER numbers are read straight from
metrics.json so both arms are guaranteed to come from one evaluator.

Outputs:
  outputs/eval_v3/comparison_v3.json   machine-readable, incl. every per-page row
  outputs/eval_v3/comparison_v3.md     the table the report quotes

Usage:
    python scripts/compare_runs.py \
        --base-pred outputs/eval_v3/baseline/test_pred_v3.jsonl \
        --base-metrics outputs/eval_v3/baseline/metrics.json \
        --ft-pred outputs/eval_v3/finetuned/test_pred_v3.jsonl \
        --ft-metrics outputs/eval_v3/finetuned/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.paths import image_id  # noqa: E402


def read_pred(path: str) -> Dict[str, Dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            out[image_id(it["image"])] = it
    return out


def page_row(name: str, rec: Dict, metric: Dict) -> Dict:
    run = rec.get("run") or {}
    gen = run.get("generation") or {}
    rep = run.get("repetition") or {}
    perf = run.get("perf") or {}
    img = run.get("image") or {}
    fm = (metric or {}).get("formula") or {}
    return {
        "page": name,
        "ref_chars": (metric or {}).get("ref_chars"),
        "hyp_chars": (metric or {}).get("hyp_chars"),
        "edit_distance": (metric or {}).get("edit_distance"),
        "text_cer": (metric or {}).get("text_cer"),
        "excluded": (metric or {}).get("excluded_reason"),
        "generated_tokens": gen.get("generated_tokens"),
        "max_new_tokens": gen.get("max_new_tokens"),
        "finish_reason": gen.get("finish_reason"),
        "hit_length_cap": gen.get("hit_length_cap"),
        "stopped_on_eos": gen.get("stopped_on_eos"),
        "repetition": rep.get("detected"),
        "rep_rules": {k: v.get("flagged") for k, v in rep.items()
                      if isinstance(v, dict) and "flagged" in v},
        "grid": img.get("image_grid_thw"),
        "n_visual_tokens": img.get("n_visual_tokens"),
        "wrapper_stripped": (run.get("postprocess") or {}).get("wrapper_stripped"),
        "formula_n_ref": fm.get("n_ref"),
        "formula_n_hyp": fm.get("n_hyp"),
        "formula_exact_match_rate": fm.get("exact_match_rate"),
        "ref_headings": ((metric or {}).get("structure") or {}).get("ref", {}).get("headings"),
        "hyp_headings": ((metric or {}).get("structure") or {}).get("hyp", {}).get("headings"),
        "ref_tables": ((metric or {}).get("structure") or {}).get("ref", {}).get("tables"),
        "hyp_tables": ((metric or {}).get("structure") or {}).get("hyp", {}).get("tables"),
        "ref_list_items": ((metric or {}).get("structure") or {}).get("ref", {}).get("list_items"),
        "hyp_list_items": ((metric or {}).get("structure") or {}).get("hyp", {}).get("list_items"),
        "heading_seq_match": ((metric or {}).get("structure") or {}).get("heading_seq_match"),
        "elapsed_seconds": perf.get("elapsed_seconds"),
        "peak_cuda_allocated_mib": (round(perf["peak_cuda_allocated_bytes"] / 2**20, 1)
                                    if perf.get("peak_cuda_allocated_bytes") else None),
        "gpu_card_used_mib_after": perf.get("gpu_memory_used_after_mib"),
    }


def arm_summary(pred: Dict[str, Dict], metrics: Dict) -> Dict:
    rows = []
    for p in metrics["per_page"]:
        name = image_id(p["image"])
        rows.append(page_row(name, pred[name], p))
    scored = [r for r in rows if not r["excluded"]]
    return {
        "n_pages": metrics["n_pages"],
        "n_scored_pages": metrics["n_scored_pages"],
        "n_excluded": metrics["n_excluded_unknown_reference"],
        "excluded_pages": [r["page"] for r in rows if r["excluded"]],
        "text_cer_macro": metrics["text_cer"],
        "text_cer_micro": metrics.get("text_cer_micro"),
        "total_edit_distance": metrics.get("total_edit_distance"),
        "total_ref_chars": metrics.get("total_ref_chars"),
        "formula_exact_match_rate": metrics["formula_exact_match_rate"],
        "formula_cer": metrics["formula_cer"],
        "n_pages_ref_has_formula_but_hyp_none": metrics.get("n_pages_ref_has_formula_but_hyp_none"),
        "structure_heading_seq_match_rate": metrics["structure_heading_seq_match_rate"],
        "n_hit_cap": sum(1 for r in scored if r["hit_length_cap"]),
        "n_repetition": sum(1 for r in scored if r["repetition"]),
        "cap_and_repetition": sum(1 for r in scored if r["hit_length_cap"] and r["repetition"]),
        "failure_rate_cap_or_rep": (round(sum(1 for r in scored
                                             if r["hit_length_cap"] or r["repetition"]) / len(scored), 4)
                                    if scored else None),
        "total_seconds": round(sum(r["elapsed_seconds"] or 0 for r in rows), 1),
        "max_peak_cuda_allocated_mib": max((r["peak_cuda_allocated_mib"] or 0) for r in rows),
        "generated_tokens": {r["page"]: r["generated_tokens"] for r in rows},
        "ref_has_tables_total": sum(r["ref_tables"] or 0 for r in rows),
        "hyp_has_tables_total": sum(r["hyp_tables"] or 0 for r in rows),
        "ref_headings_total": sum(r["ref_headings"] or 0 for r in rows),
        "hyp_headings_total": sum(r["hyp_headings"] or 0 for r in rows),
        "rows": rows,
    }


def md_table(rows: List[Dict], keys: List[tuple]) -> str:
    head = "| " + " | ".join(k[0] for k in keys) + " |"
    sep = "|" + "|".join("---" for _ in keys) + "|"
    body = []
    for r in rows:
        cells = []
        for _, field, fmt in keys:
            v = r.get(field)
            if v is None:
                cells.append("—")
            elif fmt == "bool":
                cells.append("是" if v else "否")
            elif fmt == "pct" and isinstance(v, (int, float)):
                cells.append(f"{v * 100:.1f}%")
            elif fmt == "num4" and isinstance(v, (int, float)):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-pred", required=True)
    ap.add_argument("--base-metrics", required=True)
    ap.add_argument("--ft-pred", required=True)
    ap.add_argument("--ft-metrics", required=True)
    ap.add_argument("--out", default="outputs/eval_v3")
    args = ap.parse_args()

    bp, fp = read_pred(args.base_pred), read_pred(args.ft_pred)
    bm = json.load(open(args.base_metrics, encoding="utf-8"))
    fm_ = json.load(open(args.ft_metrics, encoding="utf-8"))
    if set(bp) != set(fp):
        raise SystemExit(f"prediction page sets differ: {sorted(set(bp) ^ set(fp))}")

    base = arm_summary(bp, bm)
    ft = arm_summary(fp, fm_)

    # per-page side-by-side
    ft_by = {r["page"]: r for r in ft["rows"]}
    side = []
    for r in base["rows"]:
        o = ft_by[r["page"]]
        side.append({
            "page": r["page"],
            "excluded": r["excluded"],
            "ref_chars": r["ref_chars"],
            "base_cer": r["text_cer"],
            "ft_cer": o["text_cer"],
            "delta_cer": (None if (r["text_cer"] is None or o["text_cer"] is None)
                          else round(o["text_cer"] - r["text_cer"], 4)),
            "base_gen": r["generated_tokens"],
            "ft_gen": o["generated_tokens"],
            "base_finish": r["finish_reason"],
            "ft_finish": o["finish_reason"],
            "base_rep": r["repetition"],
            "ft_rep": o["repetition"],
            "base_sec": r["elapsed_seconds"],
            "ft_sec": o["elapsed_seconds"],
            "base_chars": r["hyp_chars"],
            "ft_chars": o["hyp_chars"],
            "ref_has_formula": r["formula_n_ref"],
            "base_formula_n": r["formula_n_hyp"],
            "ft_formula_n": o["formula_n_hyp"],
        })

    out = {
        "evaluator": "scripts/evaluate.py (same version for both arms)",
        "reference": "datasets/handwriting_notes/splits/test.jsonl",
        "base": base,
        "finetuned": ft,
        "per_page_side_by_side": side,
        "n_pages_delta_negative_fewer_errors": sum(1 for s in side
                                                   if s["delta_cer"] is not None and s["delta_cer"] < 0),
        "n_pages_delta_positive_more_errors": sum(1 for s in side
                                                  if s["delta_cer"] is not None and s["delta_cer"] > 0),
        "n_pages_equal": sum(1 for s in side if s["delta_cer"] == 0),
    }
    os.makedirs(args.out, exist_ok=True)
    jp = os.path.join(args.out, "comparison_v3.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- markdown ----
    L = []
    L.append("# 基座 vs 微调（v2 数据 checkpoint-26）同口径对比\n")
    L.append("| 指标 | 基座 | 微调 | 差值(微调−基座) |")
    L.append("|---|---|---|---|")

    def row(label, a, b, fmt="{:.4f}"):
        if a is None or b is None:
            d = "—"
        elif isinstance(a, str):
            d = f"{b} → {a}"
        else:
            d = fmt.format(b - a)
        L.append(f"| {label} | {a} | {b} | {d} |")

    row("总页数", base["n_pages"], ft["n_pages"], "{:.0f}")
    row("可评分页数", base["n_scored_pages"], ft["n_scored_pages"], "{:.0f}")
    row("排除页数（参考含 □）", base["n_excluded"], ft["n_excluded"], "{:.0f}")
    row("宏平均 CER（每页等权）", base["text_cer_macro"], ft["text_cer_macro"])
    row("字符加权 CER（Σ编辑距离/Σ参考字符）", base["text_cer_micro"], ft["text_cer_micro"])
    row("Σ编辑距离", base["total_edit_distance"], ft["total_edit_distance"], "{:.0f}")
    row("Σ参考字符", base["total_ref_chars"], ft["total_ref_chars"], "{:.0f}")
    row("公式精确匹配率", base["formula_exact_match_rate"], ft["formula_exact_match_rate"])
    row("公式 CER", base["formula_cer"], ft["formula_cer"])
    row("命中生成上限页数", base["n_hit_cap"], ft["n_hit_cap"], "{:.0f}")
    row("检出重复页数", base["n_repetition"], ft["n_repetition"], "{:.0f}")
    row("失败率（触上限或重复）", base["failure_rate_cap_or_rep"], ft["failure_rate_cap_or_rep"])
    row("总耗时(s)", base["total_seconds"], ft["total_seconds"], "{:.1f}")
    row("单页峰值 allocated(MiB)", base["max_peak_cuda_allocated_mib"],
        ft["max_peak_cuda_allocated_mib"], "{:.0f}")
    row("标题序列匹配率", base["structure_heading_seq_match_rate"],
        ft["structure_heading_seq_match_rate"])
    row("参考页中“未提取到格式化公式”的页数",
        base["n_pages_ref_has_formula_but_hyp_none"],
        ft["n_pages_ref_has_formula_but_hyp_none"], "{:.0f}")
    L.append("")
    L.append("排除页名单（双方相同）：" + (", ".join(base["excluded_pages"]) or "无"))
    L.append("")
    L.append("## 逐页明细\n")
    keys_side = [("页", "page", None), ("参考字符", "ref_chars", None),
                 ("基座CER", "base_cer", "num4"), ("微调CER", "ft_cer", "num4"),
                 ("ΔCER", "delta_cer", "num4"),
                 ("基座生成", "base_gen", None), ("微调生成", "ft_gen", None),
                 ("基座结束", "base_finish", None), ("微调结束", "ft_finish", None),
                 ("基座重复", "base_rep", "bool"), ("微调重复", "ft_rep", "bool")]
    L.append(md_table([{**s, "page": s["page"].replace("IMG_", "").replace(".jpg", "")}
                       for s in side], keys_side))
    L.append("")
    L.append("## 生成的详细运行证据（逐页）\n")
    keys_rows = [("页", "page", None), ("视觉token", "n_visual_tokens", None),
                 ("生成token", "generated_tokens", None), ("停止", "finish_reason", None),
                 ("上限", "hit_length_cap", "bool"), ("重复", "repetition", "bool"),
                 ("公式n_ref/n_hyp", None, None), ("耗时s", "elapsed_seconds", None),
                 ("峰值MiB", "peak_cuda_allocated_mib", None)]
    for label, arm in (("基座", base), ("微调", ft)):
        L.append(f"**{label}**\n")
        rows = []
        for r in arm["rows"]:
            rr = dict(r)
            rr["fch"] = f"{r['formula_n_ref']}/{r['formula_n_hyp']}"
            rr["page"] = r["page"].replace("IMG_", "").replace(".jpg", "")
            rows.append(rr)
        keys = [("页", "page", None), ("生成token", "generated_tokens", None),
                ("停止", "finish_reason", None), ("重复", "repetition", "bool"),
                ("公式 ref/hyp", "fch", None), ("耗时s", "elapsed_seconds", None),
                ("峰值MiB", "peak_cuda_allocated_mib", None),
                ("标题 ref/hyp", None, None)]
        rows2 = []
        for r in rows:
            rr = dict(r)
            rr["hh"] = f"{r['ref_headings']}/{r['hyp_headings']}"
            rr["tt"] = f"{r['ref_tables']}/{r['hyp_tables']}"
            rr["ll"] = f"{r['ref_list_items']}/{r['hyp_list_items']}"
            rows2.append(rr)
        keys2 = [k for k in keys if k[1] is not None] + [("标题 ref/hyp", "hh", None),
                                                        ("表格 ref/hyp", "tt", None),
                                                        ("列表 ref/hyp", "ll", None)]
        L.append(md_table(rows2, keys2))
        L.append("")

    mp = os.path.join(args.out, "comparison_v3.md")
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("\n".join(L[:30]))
    print(f"\nWROTE {jp}\nWROTE {mp}")


if __name__ == "__main__":
    main()
