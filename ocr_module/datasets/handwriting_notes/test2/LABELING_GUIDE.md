# test2 标注指南

> 独立测试集：**从未参与过任何调参/诊断**。现有 5 页 test 已被反复观察（HANDOFF §8），只能当开发集。

## 流程：一个问卷，两条命令

```powershell
cd E:\ocr\wheatfield_braintrust\ocr_module
python scripts/test2_tool.py form      # 生成 test2/问卷.md
#   → 打开 test2/问卷.md 答题
python scripts/test2_tool.py build --mark-reviewed
```

## 六道题

| 题 | 问什么 | 怎么答 |
|---|---|---|
| Q1 | 标题/编号 | 照抄图上编号（如 `1-1 传感器定义`）；没有写「无」 |
| Q2 | 正文 | 按图上顺序一条一行；看不清的字写 `□` |
| Q3 | 表格 | 每行 `| 单元格 | 单元格 |`；没有写「无」 |
| Q4 | 公式 | 行内 `$...$`，独立一行 `$$...$$`；没有写「无」 |
| Q5 | 红笔批注/页边注 | 直接写内容；没有写「无」 |
| Q6 | 插图/框图/示波器截图 | `> [图示：简述]`；没有写「无」 |

## 三条硬规则

1. **标注时不要看旧的 `splits/test.jsonl` / `val.jsonl` / `annotation_template.jsonl`**——看过就不独立了。
2. **不要改笔记**：原文写错、涂改、漏字，照原样转写，不要"顺手修正"。
3. **不要补、不要概括**：只写图上看得见的；看不清用 `□`，整块看不清写 `> [无法辨认：区域描述]`。

## 组装规则（build 自动做）

- Q1 → `## 标题`
- Q2/Q3/Q4/Q5/Q6 → 按顺序各成一段（`无` 与空自动跳过）
- 结果写 `answers/<页>.md`、`splits/test2.jsonl`（`image` 为 `images/<文件名>`）、`test2_manifest.json`（每页 sha256）

## 规模

8–15 页，尽量覆盖纯文字 / 表格 / 公式 / 潦草字迹。低于 8 页只能报个案。
