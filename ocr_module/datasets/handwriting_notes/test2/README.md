# test2 —— 独立测试集（待标注）

> 这个集合用来替代已被反复观察的旧 `test`。**标注时不要参考旧的 test/val 标注。**

## 目录

```
test2/
├── images/            放新页照片（已 gitignore，只留在本机）
├── answers/           每页一个 <同名>.md —— 你要写的地方
├── LABELING_GUIDE.md  格式规则、例子、常见错误、自检清单
└── test2_manifest.json  运行 build 后生成（每页图片/答案的 sha256）
```

## 三步

```powershell
cd E:\ocr\wheatfield_braintrust\ocr_module
python scripts/test2_tool.py prepare     # 为每张图生成 answers/<同名>.md 骨架
#   → 在 answers/ 里逐页写转写
python scripts/test2_tool.py check       # 有缺页/空页/TODO 会报错
python scripts/test2_tool.py build --mark-reviewed
```

`build` 会写出 `datasets/handwriting_notes/splits/test2.jsonl`（`image` 用相对路径 `images/<文件名>`，可直接被 `resolve_image_path` 解析）。

## 规模建议

8–15 页，覆盖纯文字 / 表格 / 公式 / 潦草字迹四类版式。低于 8 页只能报个案，不能支撑"泛化"结论。
