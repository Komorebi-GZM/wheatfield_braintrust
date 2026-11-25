# test2 —— 独立测试集（一个问卷，两条命令）

> 替代已被反复观察的旧 `test`。**标注时不要参考旧 test/val 的标注。**

## 两条命令

```powershell
cd E:\ocr\wheatfield_braintrust\ocr_module

# 1) 把页面照片放进 test2/images/ 后，生成一个问卷
python scripts/test2_tool.py form

# 2) 打开 test2/问卷.md，每题 Q 后面写答案（不需要的写「无」）
#    写完跑这一条，标注 / splits / manifest 全部生成
python scripts/test2_tool.py build --mark-reviewed
# 只想先建已答的页（跳过未答）：
python scripts/test2_tool.py build --mark-reviewed --allow-partial
```

就这两步。`check` 只是体检（不写文件），可跳过。

## 问卷长这样

```markdown
# 第 1 页：第1页.jpg

Q1 标题/编号（照抄图上编号，没有写「无」）
1-1 传感器定义

Q2 正文（按图上顺序一条一行，看不清的字写 □）
- 传感器是能感受被测量并按一定规律转换成可用输出信号的器件
- 通常由敏感元件和转换元件组成

Q3 表格（每行 | 单元格 | 单元格 |，没有写「无」）
| 元件 | 作用 |
| 敏感元件 | 直接感受被测量 |

Q4 公式（行内 $...$，独立一行 $$...$$，没有写「无」）
灵敏度 $S = \Delta y / \Delta x$

Q5 红笔批注/页边注（没有写「无」）
无

Q6 插图/框图/示波器截图（写 > [图示：…]，没有写「无」）
无
```

- 答案可以换行写，遇到下一个 `Q` 或下一页标题自动结束。
- 六个问题：标题 / 正文 / 表格 / 公式 / 批注 / 插图。
- 页数不限；每页六题，`build` 会一次组装成 Markdown 并写出 `splits/test2.jsonl` 与 `test2_manifest.json`。

## 隐私

`images/` 已 gitignore（照片只留本机）；`answers/` 与 `问卷.md` 会入库。
若整批都不想公开，把 `answers/` 和 `问卷.md` 也加进 `.gitignore`，只保留 `test2_manifest.json` 的哈希。

