# 训练计划 v4 —— 需求、边界、计划与本机核验

> 状态：**已核验，未开跑**。本文件只记录"要训什么、不训什么、怎么训、本机是否具备条件"，训练命令尚未执行，等 owner 指令。
> 适用范围：`ocr_module/`（小麦 OCR 训练线，分支 `feat/ocr-xiaomi-qlora`）。
> 基线提交：`58e5360`（含 `dfea55d` 路径/OTSL/评测归一化修复 + `fce9112` 旧脚本页 ID 修复）。

---

## 0. 一页结论

| 判定项 | 结论 | 关键证据 |
|---|---|---|
| 训练需求是否清楚 | 清楚：手写课程笔记 → 结构化 Markdown（标题/列表/表格/LaTeX） | §1 |
| 边界是否锁定 | 锁定：划分冻结 13/5/5、0 合成、不碰 test 调参、不换模型/不改 LoRA 结构 | §2 |
| 计划是否可执行 | 可执行：单条 PowerShell 命令，52 步、约 4 分钟、4 个 epoch checkpoint | §3 |
| 本机环境是否就绪 | **就绪** | §4：RTX 4060 8G、torch 2.6.0+cu124、ms-swift 4.6.0、bnb 0.50.2、peft 0.20.0 |
| 数据/标注是否就绪 | **就绪** | §5：23 页、13/5/5、7 项内容 checks 全 true、18 页全部可编码、0 截断 |
| 权重链路是否就绪 | **就绪** | §6：基座快照完整、checkpoint-26 adapter 的 sha256 与清单一致 |
| 本次是否训练 | **否** | 等 owner 指令；且当前产物已是最佳 ckpt-26，同配置重跑只会复现 |

---

## 1. 需求（做什么）

1. **任务定义**：输入一页手写课程笔记照片（中文、含公式与表格），输出结构化 Markdown——保留标题层级、分点列表、表格、`$...$` / `$$...$$` 公式。
2. **为什么微调**：基座 Xiaomi-OCR-0 对"密集手写 + 中文课程笔记结构"不稳定（长输出退化、结构丢失、公式定界符约定不一致），QLoRA 用于把输出约定拉回标注口径。
3. **交付物定义**：一个 LoRA adapter（约 21.7 MB）+ 权重校验记录 + 同口径评测报告；**权重不入库**，只入库文字产物与哈希。
4. **成功判据（措辞受限）**：在**未参与调参**的页面上，基座与微调用**同一份推理协议**比较；同时报"正文 CER"和"含公式体 CER"；报告可评分页数与每页排除原因。**不得**用单一指标宣称"整体识别更好了"。

---

## 2. 边界（本轮硬约束）

| # | 约束 | 理由 |
|---|---|---|
| 1 | **绝不重新划分**，train/val/test 保持 13/5/5，成员不变 | 重划分会让全部历史指标失效（`data_version_v2.json` 的 `split_membership` 是冻结记录） |
| 2 | **合成数据 0 张**（本次 `data/synth` 不存在）；若将来合成，只允许进 train | val/test 必须全部是真实页 |
| 3 | **test 集不参与任何选择**（不挑 checkpoint、不挑提示词、不挑上限）；它已被历史诊断反复观察，只能当"开发集"报告 | HANDOFF §8 第 2 条已认定"从未见过"的前提不可恢复 |
| 4 | **不换基座、不改 LoRA 结构**（保持 r8/α32/all-linear/冻结 vit+aligner） | 当前瓶颈是"密集手写页退化"，没有容量不足的证据 |
| 5 | **一次只改一个因子**，改了就换新实验目录，旧预测不得拼接 | 保证差值可归因 |
| 6 | 权重、缓存、裁切图、`*.log` 不入库（`.gitignore` 已有白名单） | 仓库为 PUBLIC，原图/标注已在历史提交中，不得再扩大 |
| 7 | 训练/评测产物落在 `outputs/`（Git 忽略），只有报告与清单进提交 | 同上 |

**本轮明确不做**：扩大训练样本、加新测试页标注、改 `max_pixels`、换模板、上更大模型、发布权重。这些都属于下一轮，且需要 owner 决定（见 §10）。

---

## 3. 训练内容与计划

### 3.1 配置（与 `scripts/train_qlora_v2.ps1` 的 `-Inspect` 输出逐项一致）

| 项 | 值 |
|---|---|
| 基座 | `SeerRay-Lab/Xiaomi-OCR-0`（`Qwen3_5ForConditionalGeneration`，0.8B），本地 ModelScope 快照 |
| 量化 | 4-bit bnb：NF4 + 双重量化 + bf16 计算 |
| LoRA | r=8、α=32、dropout 0.05、`all-linear`；冻结 `vit` + `aligner`；可训练 ≈5.41M（0.89%） |
| 数据 | train `splits/train_swift.jsonl`（13 页）、val `splits/val_swift.jsonl`（5 页），`split_dataset_ratio=0` |
| 序列/像素 | `max_length=8192`、`truncation_strategy=delete`、`max_pixels=1024000`（→ 972 视觉 token） |
| 优化 | bs 1 × accum 1 × 4 epoch × 13 页 = **52 个 optimizer step**；lr 1e-4 / cosine / warmup_ratio 0.03 |
| 保存/评估 | 每 epoch 评估+保存，`save_total_limit=4`（→ ckpt-13/26/39/52） |
| 随机性 | seed 42 / data_seed 42 |
| 产物 | `outputs/qlora-v2/v2data-<时间戳>/`（Git 忽略） |

### 3.2 时间与显存预算

| 项 | 实测（上一轮 `v2data-20261115-134816`） |
|---|---|
| 训练时长 | 52 步 / **237.98 s**（≈4 分钟） |
| 训练 loss | 末步 0.603；最近 10 步均值 0.600；`train_loss` 0.7831 |
| 验证 loss | 1.20785（ep1）→ **1.18883（ep2，ckpt-26）** → 1.18942（ep3）→ 1.19147（ep4） |
| 日志显存 | `memory(GiB)` 峰值 **8.4**、均值 6.63 —— **WDDM 统计含共享内存**，不能当独占物理显存 |
| 当前 GPU 余量 | 8188 MiB 总 / 已用 1613 MiB / **空闲 6344 MiB**（桌面占用 1.6 GB） |

> 显存余量偏薄：训练前应关闭占用 GPU 的桌面程序（浏览器硬件加速、录屏等），否则可能落到共享内存拖慢速度（不会直接 OOM，但会显著变慢）。

### 3.3 计划（本轮只到"可执行"为止）

1. **不重训**：现有 `checkpoint-26` 已是最佳（按 val loss 选），同配置重跑只会复现同一结果（seed 固定）。
2. 若 owner 要求重训，命令固定为：`powershell -File scripts/train_qlora_v2.ps1 -OutputRoot outputs/qlora-v2`，跑完先做 adapter 校验（372/372、`max_abs_diff=0`）再谈评测。
3. 真正缺的是**新独立测试页**（当前 5 页已被反复观察），见 §10 第 2 条。

---

## 4. 本机环境核验（2026-11-24 本机实测）

| 项 | 实测值 | 判定 |
|---|---|---|
| OS / GPU | Windows，`NVIDIA GeForce RTX 4060 Laptop GPU` 8188 MiB，驱动 621.24 | ✓ 满足 |
| Python | 3.12.13（`.venv`，`include-system-site-packages = true`，宿主 `D:\conda_envs\py312-dsml`） | ✓ 满足 3.12 要求 |
| torch | 2.6.0+cu124，`cuda_available=True`，`get_device_name=RTX 4060 Laptop` | ✓ 满足 |
| transformers | 5.14.1 | ✓ |
| ms-swift | `ms_swift 4.6.0`（`swift.exe` 在 `.venv\Scripts`） | ✓ ≥4.6，认 `qwen3_5` 模板 |
| bitsandbytes | 0.50.2（CUDA 匹配，Windows 上最大坑已趟过） | ✓ ≥0.50.2 |
| peft / accelerate / datasets | 0.20.0 / 1.14.0 / 4.8.4 | ✓ |
| pytest | 9.2.0；`pytest tests --basetemp=.pytest_tmp` → **38 passed** | ✓ |
| 基座缓存 | `outputs/modelscope_cache/models/SeerRay-Lab--Xiaomi-OCR-0/snapshots/master`，10 文件 **1,766,990,278 B**（1.65 GB，`model.safetensors` 1.63 GB） | ✓ 完整 |
| HF 缓存 | `~/.cache/huggingface/hub` **不存在** | ⚠️ 见 4.1-① |

### 4.1 三条环境发现（写下来避免下一个人踩）

1. **`scripts/probe_swift_dataset.py` 现在跑不通**：它用 `snapshot_download(..., local_files_only=True)` 找 HF 缓存，但本机没有 HF 缓存（模型只在 ModelScope 缓存里）。本次改用"直接指向 ModelScope 快照路径"的等价探测（§5.3），结果正常。若要保留原脚本，需要给它加 `--model` 参数。
2. **`.venv` 是"共享 conda 站点包"的 venv**：`swift` 装在本 venv，而 `torch` / `transformers` 从 `D:\conda_envs\py312-dsml` 导入。**升级该 conda 环境会静默改变训练运行时**，训练前必须重跑 §11 的版本核对。
3. **本机 `core.autocrlf=true`**：`annotation_template.jsonl` 检出为 CRLF，`data_version.py` 按字节算出的哈希与仓库冻结值不同（`62e520…` vs `34f262…`），**但内容没变**。首次遇到属于误报，详细解释见 `windows_preflight.md` 末两条；不要把重新生成的清单覆盖回去。

> 另：本工作区此前被 Windows 文件权限挡住（`E:\ocr` 缺 WRITE_OWNER、子目录缺写权限）。已修复工作区根的权限，但 **`.git` 及子目录对沙箱内进程仍不可写**，因此 git 写操作（fetch/commit/push）需要提权执行；读操作、跑测试、跑脚本不受影响。这条是环境事实，不是训练阻塞项。

---

## 5. 数据与标注核验

### 5.1 结构

| 项 | 实测 |
|---|---|
| 原图 | `datasets/handwriting_notes/images/` **23 张** |
| 标注 | `annotation_template.jsonl` **23 行**（v2 口径；v1 备份 `annotation_template.v1.bak.jsonl`） |
| 划分 | train 13 / val 5 / test 5（`splits/*.jsonl`，含 `train_swift` 13、`val_swift` 5、`train_sharegpt.json`） |
| 合成数据 | `data/synth` **不存在** → 0 张 |
| 第二份标注 | `data/annotations/real.jsonl`（标注 bot 版）**不被任何 split 引用**，未参与训练/评测；抽验存在读错，**不得当真值** |

### 5.2 内容级 checks（`python scripts/data_version.py`，本机实跑）

```
image_count_23              true
splits_disjoint             true
splits_cover_all_images     true
split_images_exist          true
markdown_equals_annotation  true
swift_images_in_annotation   true
membership_unchanged_vs_v1  true
```

> 结论：数据**没有被动过**。判断依据是这 7 项内容级 checks，而不是 sha256 是否相等（见 §4.1-③）。

### 5.3 可编码性（用真实 processor + 模板，本机实跑 18 页）

| 划分 | 页数 | token 范围 | 监督 token 范围 | 截断 | 图像网格 |
|---|---|---|---|---|---|
| train_swift | 13 | 1144 – 1615 | 140 – 611 | **0 页** | `[1,54,72]` / `[1,72,54]` |
| val_swift | 5 | 1380 – 1533 | 376 – 529 | **0 页** | 同上 |

- 视觉侧：`max_pixels=1024000` → 3888 patch → **972 视觉 token**（与训练/推理协议一致）。
- 监督 token 上界 611 与历史记录一致（"非测试页最长参考答案 611 token"）。
- 页面路径：用 `ocr.paths.image_id` / `resolve_image_path` 按真实清单逐条解析，**23/23 命中本地图片，0 缺失**（含历史 `E:\...` 绝对路径）。

### 5.4 标注版本（v2）事实

- 23 页全部目视复核，**20 页维持、7 页修正 10 处**（`outputs/annotation_fixes_v2.json` 为补丁记录，`scripts/apply_annotation_fixes.py` 带锚点断言）。
- 已知残留：`□` 占位符仍存在于部分页（test 里 000020 / 000028 因此被排除评分），属**已记录缺陷**，本轮不修。

---

## 6. 权重链路核验

| 项 | 实测 |
|---|---|
| 基座快照 | 10 文件 / 1,766,990,278 B；`config.json` 架构 `Qwen3_5ForConditionalGeneration` |
| 训练产物 | `outputs/qlora-v2/v2data-20261115-134816/`（`args.json`、`logging.jsonl`、4 个 ckpt、`command.txt`） |
| 最佳 ckpt | `checkpoint-26`（`trainer_state.best_model_checkpoint` 指向它，val loss 最低） |
| adapter | `adapter_model.safetensors` **21,700,560 B**，sha256 `a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6` —— 与 `CHECKPOINT_MANIFEST.md` 记录**逐字节一致** |
| 推理协议 | `inference_protocol_v3_delivery.json`（当前配置指向 ckpt-26）；`v2` 为历史记录（ckpt-4），保持原样 |

---

## 7. 执行步骤（等指令才执行）

```powershell
# 0) 前置：确认 GPU 空闲、环境未漂移
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch,swift;print(torch.__version__, torch.cuda.is_available())"

# 1) 数据完整性（内容级 checks 必须全 true；不要提交重新生成的清单）
cd ocr_module
python scripts/data_version.py

# 2) 训练（52 步，约 4 分钟；产物在 outputs/qlora-v2/v2data-<stamp>/）
powershell -File scripts/train_qlora_v2.ps1 -OutputRoot outputs/qlora-v2

# 3) 训练后：adapter 校验（372/372、max_abs_diff=0、weights_verified=true）
python scripts/verify_adapter.py --adapter outputs/qlora-v2/v2data-<stamp>/checkpoint-<N>

# 4) 只有校验通过，才允许接推理/评测；换任何参数都必须换输出目录
```

**回滚**：训练产物全部在 `outputs/`（Git 忽略），删除对应 run 目录即可；不涉及仓库文件。若误改了 `data_version_v2.json` / `.sha256`，`git checkout -- ocr_module/datasets/handwriting_notes/` 恢复。

---

## 8. 验收标准与判读规则（防止过度宣称）

**训练侧（可达）**
1. 52 步跑完，4 个 ckpt 落盘，`train_runtime` 量级 ≈4 分钟；
2. val `eval_loss` 出现下降并选出一个 ckpt（按 val loss，**不看 test**）；
3. adapter 校验 372/372 命中、`max_abs_diff=0`、`weights_verified=true`。

**评测侧（必须同口径）**
4. 基座与微调用**同一份协议文件**（同一 `max_pixels`、同一 `max_new_tokens`、同一 prompt、同一 `normalize_output`）；
5. 同时报：正文 CER、含公式体 CER、可评分页数、每页排除原因、公式 EM、结构匹配；
6. 允许的结论句式："在 N 页可评分页上，正文 CER 从 A 变到 B；含公式体 CER 从 C 变到 D"；
7. **禁止**的结论句式："微调后 OCR 整体变好 / 泛化更好 / 表格能力提升"——3 页可评分、0 张表、公式 EM=0 都不支持。

**当前基线（`eval_report_v3.md`，3 页可评分）**：正文宏 CER 1.4872 → 0.7725；含公式体 1.1217 → 1.7952（**变差**）；结论为**混合变化**。任何新一轮结果都必须与这组数字同口径对照，并说明差异来源。

---

## 9. 已知坑与规避

| # | 坑 | 规避 |
|---|---|---|
| 1 | Windows 上 ms-swift 的 `update_last_checkpoint_symlink` 会 WinError 1314 → 回滚时 `os.remove` 一个目录 → WinError 5，首个 save 崩 | 脚本已在 run dir 预置空 `last-checkpoint` 目录，让其走"已存在且非符号链接"的早退分支；**不要删这个目录** |
| 2 | `args.json` 里 `warmup_steps=0` 是落盘失真 | 以 `warmup_ratio=0.03` 为准（52 步 × 0.03 ≈ 1.56 步） |
| 3 | `memory(GiB)` / `nvidia-smi` 在 WDDM 下含共享内存，8.4 GiB > 物理 8188 MiB 是正常现象 | 不要把该数字当"独占显存"，也不要写进对外报告 |
| 4 | 训练数据里可能被 `plain_text()` 系统性优待（公式体被删） | 评测同时报"含公式体 CER"；新增"参考有公式而预测没有"的计数 |
| 5 | 提示词/上限一改就换目录 | 旧预测不得拼接；协议文件另存 v4 |
| 6 | `train_sharegpt.json` 是缩进 JSON，不能按行数当样本数 | 13 页就是 13 条 |

---

## 10. 待 owner 确认（本轮不动）

1. **重训与否**：当前 ckpt-26 已是最佳；同配置重跑只会复现。若目的是"验证流水线仍可用"，跑一次 4 分钟可以接受，但要明确这只是复现性验证，不是新结果。
2. **新独立测试页**：现有 5 页已被反复观察，建议按 v2 复核标准新标 8–15 页从未参与调参的真实页，形成 `test2`；旧 test 降级为"开发集"并在报告里标注。**需要人工标注投入。**
3. **行尾统一**：若要 sha256 跨平台可复现，需统一 LF 并重算一次冻结清单——那会改动 `splits/*` 的字节，属数据变更，**必须显式批准**。
4. **权重托管**：adapter 21.7 MB 目前不入库；若要发布，用 HF/ModelScope 托管并只提交哈希与下载方式。

---

## 11. 本次核验命令清单（可复现）

```powershell
git -C E:\ocr\wheatfield_braintrust log --oneline -3          # 58e5360 = 基线
cd E:\ocr\wheatfield_braintrust\ocr_module

# 环境
..\.venv\Scripts\python.exe -V
..\.venv\Scripts\python.exe -m pip list | Select-String 'torch|swift|peft|bitsandbytes|transformers|accelerate|pytest'
..\.venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# 单测（38 项）
..\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.pytest_tmp

# 数据内容级 checks（7 项全 true；不提交重新生成的清单）
..\.venv\Scripts\python.exe scripts/data_version.py

# 页 ID / 路径解析（23/23）
# 编码探测（真实 processor + 模板，18 页，0 截断）

# 权重链路
Get-FileHash outputs\qlora-v2\v2data-20261115-134816\checkpoint-26\adapter_model.safetensors -Algorithm SHA256
# 期望 a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6

# 训练脚本参数（不训练，只打印）
powershell -NoProfile -File scripts/train_qlora_v2.ps1 -Inspect
```

---

**本文件不含任何新实验结果**：上面的 CER 数字全部来自既有 `eval_report_v3.md`，训练/评测本轮均未执行。

---

## 12. 执行记录：2026-11-24 重训（复现性验证）

owner 于 2026-11-24 12:48 下达「开跑」。本轮训练是**复现性验证**，不是新实验：配置与 11-15 那次逐项相同（见 §3.1），预期结果是"同输入同权重"。

```powershell
cd ocr_module
powershell -File scripts/train_qlora_v2.ps1 -OutputRoot outputs/qlora-v2
# 实际调用：swift sft ...（参数与 §3.1 完全一致），run 目录 outputs/qlora-v2/v2data-20261124-124959
```

### 12.1 结果

| 项 | 本次 `v2data-20261124-124959` | 上次 `v2data-20261115-134816` |
|---|---|---|
| 起止 | 12:49:59 → 12:54:42 | — |
| 步数 | 52/52 | 52/52 |
| `train_runtime` | **147.5 s**（2m28s） | 237.98 s |
| `train_loss`（汇总） | **0.7831** | 0.7831 |
| eval_loss（ep1→ep4） | 1.208 / **1.189** / 1.189 / 1.191 | 1.20785 / **1.18883** / 1.18942 / 1.19147 |
| best checkpoint | **checkpoint-26**（按 val loss） | checkpoint-26 |
| 可训练参数 | 5,411,328 / 609.59M（0.8877%） | 同 |
| 显存（WDDM 统计） | 峰值 8.4 GiB，均值约 5.6–6.7 GiB | 峰值 8.4 GiB |
| 产物大小 | 249.7 MB（4 ckpt + 日志，未入库） | — |

### 12.2 复现性（本轮的核心产出）

`checkpoint-26/adapter_model.safetensors` 两次运行 **sha256 完全相同**：

```
a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6   (21,700,560 B)
```

同一次 run 内其余 ckpt 的哈希互不相同（13：`696e251e…`、39：`d9f0c3d1…`、52：`e8e303f1…`），说明不是拷贝同一份文件，而是真的各自训练得到。结论限定为：**在该机器、该依赖版本、该 seed 下，这条训练链路可逐字节复现**；不推广为"训练是确定性的"（换 GPU/驱动/库版本/并行度都可能改变数值路径）。

### 12.3 权重校验（`scripts/verify_adapter.py` → 新 ckpt-26）

```
n_tensors_on_disk 372 · n_lora_tensors_live 372 · n_matched 372
missing_in_model 0 · extra_in_model 0 · value_mismatched 0 · max_abs_diff 0.0
peft_loader_missing_lora_keys 0 · n_lora_modules 372
lora_modules_outside_language_model 0 · lora_B_all_zero 0 · lora_B_nonzero 186
lora_params disk/live 5,411,328 · weights_verified = true
```

记录文件：`outputs/qlora-v2/v2data-20261124-124959/adapter_check_ckpt26.json`（未入库）。

### 12.4 本轮踩到的两条环境修正（已影响脚本可用性）

1. `scripts/verify_adapter.py` 的 `--base` 默认取 `DEFAULT_BASE` = HF id `SeerRay-Lab/Xiaomi-OCR-0`；本机没有 HF 缓存、HF endpoint 也走不通，直接运行会 `LocalEntryNotFoundError`。**解决**：`--base outputs/modelscope_cache/models/SeerRay-Lab--Xiaomi-OCR-0/snapshots/master`，或设 `OCR_BASE_MODEL` 环境变量（`ocr/engine.py` 支持 `OCR_BASE_MODEL` 覆盖）。
2. 训练启动时会向 ModelScope 复核快照清单（本次 28/29 文件），命中本地缓存，不影响结果；但离线环境下这一步需要能访问 ModelScope 或已有完整缓存。

### 12.5 结论

训练 + 校验链路在本机可用，且与 11-15 的结果逐字节一致。**本轮不产生任何新的效果结论**；下一步按 §9 进入阶段 A（验证集参数对比，不碰测试集），产物写入 `outputs/phaseA/`（Git 忽略）。

> 阶段A 已执行完毕，结果见 **`phase_a_cap_report_v4.md`**（因子一：上限 1024/2048）与 **`phase_a_prompt_report_v4.md`**（因子二：提示词 A、B）。因子一：上限不是瓶颈，2048 不应成为新默认。因子二：臆造模板确由提示词触发（小标题 139→7、重复条目 26%→0.9%），但 A/B 都以欠生成与公式丢失为代价，逐页看冻结指令仍赢 3/5 页——提示词这条线到此为止，下一杠杆是解码侧或训练侧。
