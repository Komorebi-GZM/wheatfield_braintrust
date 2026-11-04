# 手写课程笔记数据集（handwriting_notes）

麦田智囊 OCR 子模块的手写笔记数据集。内容为**手写中文物理/电子学课程笔记**照片。

## 目录

```
handwriting_notes/
├── images/                   # 23 张手机拍摄手写笔记（3072×4096 / 4096×3072，RGB）
├── pdf/                      # 参考打印版/习题
│   ├── 物理。.pdf             # 12 页
│   └── 题型归纳.pdf           # 8 页
├── annotation_template.jsonl # 整页级标注模板（image + 空 markdown，待填）
├── 图片清单.csv               # 文件/尺寸/方向/大小清单
└── 校验报告.md                # 数据校验结论
```

## 状态

- 图像校验：23/23 可读、无损坏、无重复（见 `校验报告.md`）。
- **标注：尚未完成** —— `annotation_template.jsonl` 的 `markdown` 字段为空，需人工按整页级补全（保留标题/列表/表格/公式；不可辨认用 `□`，不臆造）。

## 用于训练

把 `images/` 作为 `ocr_module/data/raw/`，将填好的标注保存为 `ocr_module/data/annotations/real.jsonl`，
再运行 `scripts/prepare_dataset.py`。注意：原图 12MP，训练前建议缩放到长边 1600–2048。

## 隐私说明

本目录含**个人手写笔记原图**，已由仓库 owner 明确同意入库。请勿外传或用于训练数据以外的用途。
