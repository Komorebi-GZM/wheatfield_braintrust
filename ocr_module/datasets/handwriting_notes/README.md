# 手写课程笔记数据集（handwriting_notes）

麦田智囊 OCR 子模块的手写笔记数据集。内容为**手写中文物理/电子学课程笔记**照片。

## 目录

```
handwriting_notes/
├── images/                   # 23 张手机拍摄手写笔记（3072×4096 / 4096×3072，RGB）
├── pdf/                      # 参考打印版/习题
│   ├── 物理。.pdf             # 12 页
│   └── 题型归纳.pdf           # 8 页
├── annotation_template.jsonl # 整页级原图转写与复核状态
├── splits/                   # 固定的真实页训练、验证和测试划分
├── 图片清单.csv               # 文件/尺寸/方向/大小清单
└── 校验报告.md                # 数据校验结论
```

## 状态

- 图像校验：23/23 可读、无损坏、无重复（见 `校验报告.md`）。
- `annotation_template.jsonl` 的 23 页已完成一轮原图对照，无法辨认的内容标为 `□`。远程标注与本地版本有差异，见 `remote_annotation_review.md`；一轮复核不保证标注无误。
- **v2（2026-11-15）**：23 页逐页对照原图二次复核，修正 7 页 10 处（见 `../../annotation_review_v2.md`），v1 保留为 `annotation_template.v1.bak.jsonl`。数据版本与全部 SHA-256 见 `data_version_v2.json` / `data_version_v2.sha256`，修正明细见 `../../outputs/annotation_fixes_v2.json`。
- 固定划分：真实训练 13 页、验证 5 页、测试 5 页。测试集中一页严重失焦，含 `□` 的参考页保留在清单中，但评测脚本不将其计入精确指标。

## 用于训练

运行 `scripts/prepare_dataset.py --real-dir datasets/handwriting_notes --real-ann datasets/handwriting_notes/annotation_template.jsonl --out-dir datasets/handwriting_notes/splits`。训练入口使用该目录的 Swift 格式文件；合成数据必须重新生成并只进入训练集。原图保持原始分辨率，训练处理器的像素预算另行配置。

## 隐私说明

本目录含**个人手写笔记原图**，已由仓库 owner 明确同意入库。请勿外传或用于训练数据以外的用途。
