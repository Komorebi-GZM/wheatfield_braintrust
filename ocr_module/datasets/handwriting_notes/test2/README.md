# test2 —— 独立测试集（待标注）

> 这个集合用来替代已被反复观察的旧 `test`。**标注时不要参考旧的 test/val 标注。**
> 标注方式是**问答式**：每页生成一份填空问卷，你只回答问题，工具组装成标注。

## 目录

```
test2/
├── images/            放新页照片（已 gitignore，只留在本机）
├── answers/
│   ├── <同名>.form.md   问卷 —— 你在这里回答问题
│   └── <同名>.md        组装后的标准标注（由 assemble 生成）
├── LABELING_GUIDE.md  规则、例子、常见错误、自检清单
└── test2_manifest.json  运行 build 后生成（每页图片/答案的 sha256）
```

## 操作

```powershell
cd E:\ocr\wheatfield_braintrust\ocr_module

# 1) 把照片放进 test2/images/  （或直接发我，我来生成问卷）
# 2) 生成问卷
python scripts/test2_tool.py form
# 3) 在 answers/<同名>.form.md 里回答 Q1..Q6（不需要的题写「无」）
# 4) 组装 + 检查
python scripts/test2_tool.py assemble
python scripts/test2_tool.py check
# 5) 生成评测清单
python scripts/test2_tool.py build --mark-reviewed
```

`build` 写出 `datasets/handwriting_notes/splits/test2.jsonl`（`image` 用相对路径 `images/<文件名>`，可被 `resolve_image_path` 解析）与 `test2_manifest.json`。

## 六个问题

Q1 标题/编号 · Q2 正文（一条一行）· Q3 表格（`| a | b |`）· Q4 公式（`$...$` / `$$...$$`）· Q5 红笔批注 · Q6 插图（`> [图示：…]`）。

## 规模建议

8–15 页，覆盖纯文字 / 表格 / 公式 / 潦草字迹四类版式。低于 8 页只能报个案，不能支撑"泛化"结论。
