# Checkpoint 交付清单（checkpoint-26）

本文件说明本次交付所用**权重尚未上传**，以及拿到同一份权重所需的全部信息与校验方式。

- 生成时间：2026-11-15
- 训练 run：`outputs/qlora-v2/v2data-20261115-134816`（Git 忽略，见 `ocr_module/.gitignore`）
- 交付 checkpoint：`outputs/qlora-v2/v2data-20261115-134816/checkpoint-26`
- 选中依据：固定验证集 5 页 `eval_loss` 最低（1.18883，step 26 / epoch 2），测试集未参与选择
- 为什么不在 Git 里：仓库此前**没有**任何 Git LFS 规则（无 `.gitattributes`，`git lfs env` 可用但未配置跟踪模式），且 `ocr_module/.gitignore` 明确忽略 `outputs/` 与 `*.safetensors` / `*.pt` / `*.bin`。按"没有合适存储机制时先推代码与报告、权重单列清单"的约定处理，**未擅自新建公开模型仓库，也未把权重塞进普通 Git**。

---

## 1. 已产出的文件与校验值

| 文件 | 字节 | SHA-256 | 是否属于交付必需 |
|---|---|---|---|
| `adapter_model.safetensors` | 21,700,560 | `a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6` | **是**（LoRA 权重本体） |
| `adapter_config.json` | 1,335 | `e5a66259994708c1447aed9dca9f838d650e025887b1eeaac3f0f866dd2e63d8` | **是**（r/alpha/target 正则/base 路径） |
| `additional_config.json` | 67 | `c7799462ebedae6557ffad31566029e2d2f958b7b40e46e972cf901bcaf45733` | 建议（量化/额外配置） |
| `args.json` | 18,880 | `021e7beadedd84fd2975e8bdebb406cdc990d8123ff3afdfec7dc23dc9b5d9ed` | 建议（训练实参存档） |
| `trainer_state.json` | 7,324 | `bd3287ee36860565a3161a3cd76b8aad0693e04b102ff3d29d70540e933be8c6` | 建议（loss/LR 曲线） |
| `README.md` | 5,382 | `ad0f4e4ce77642f6bebd69ba027bbacef807b8f58b85d11f947dc86d52d53c8c` | 否（ms-swift 生成） |
| `optimizer.pt` | 43,602,670 | `3d8301b7de29faad82986378d86c65f5f558de43b3508ace4bd6894933865769` | **否**（仅续训需要，不必交付） |
| `scheduler.pt` | 1,000 | `143d8aad94ab451df98542f599ab5672e673d7dfe1c2dde6738fc967ccb78db5` | 否 |
| `rng_state.pth` | 14,244 | `68080808546ac1ccf387c04a271cc8d341be59d3213c10b06c4e147cc3df1478` | 否 |
| `training_args.bin` | 6,392 | `7c340944398539c685d33e9edaa62e5b96779264f567c06eb1f71b783365faa6` | 否 |

**交付必需部分合计约 21.7 MB**（1 个权重文件 + 1 个 config）。

> `adapter_model.safetensors` 的 SHA-256 与评测运行 sidecar 中的 `adapter_weights_sha256` 一致（`a2be86a2…`），也与 `outputs/protocol/adapter_check_v2data_ckpt26.json` 一致——三方交叉核对通过。

---

## 2. 拿到同一份权重后如何自检

```bash
cd ocr_module
sha256sum outputs/qlora-v2/v2data-20261115-134816/checkpoint-26/adapter_model.safetensors
# 期望 a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6

../.venv/Scripts/python.exe scripts/verify_adapter.py \
  --adapter outputs/qlora-v2/v2data-20261115-134816/checkpoint-26
# 期望 weights_verified=true，372/372，max_abs_diff=0.0
```

基线校验记录：`outputs/protocol/adapter_check_v2data_ckpt26.json`（`weights_verified: true`）。

---

## 3. 基座模型（另行获取，不在本仓库）

| 项 | 值 |
|---|---|
| 模型 id | `SeerRay-Lab/Xiaomi-OCR-0` |
| 架构 | `Qwen3_5ForConditionalGeneration`（model_type `qwen3_5`） |
| 来源 | ModelScope / HuggingFace 同名模型（训练与推理分别用了 ModelScope 与 HF 快照，此前已逐文件 SHA-256 比对一致，含 1.7 GB 权重） |
| 本地快照路径（仅本机有效） | `outputs/modelscope_cache/models/SeerRay-Lab--Xiaomi-OCR-0/snapshots/master` |
| 注意 | LoRA 权重只能挂到**同一基座**上；换基座版本会导致 372 个 LoRA 键失配或数值差异 |

---

## 4. 复现训练（可选）

训练可从数据 + 脚本重建，不依赖本 checkpoint：

```bash
cd ocr_module
powershell -File scripts/train_qlora_v2.ps1 -OutputRoot outputs/qlora-v2
```

- 训练实参存档：`outputs/qlora-v2/v2data-20261115-134816/args.json`
- 数据：`datasets/handwriting_notes/splits/train_swift.jsonl`(13) + `val_swift.jsonl`(5)，哈希见 `datasets/handwriting_notes/data_version_v2.json`
- 预期规模：52 步（4 epoch × 13 样本 / batch1 / grad_accum1），238 s，train_loss 0.7831
- 注意：训练是确定性的**说法不成立**——GPU/驱动/库版本变化可能改变数值路径，重训结果不保证逐位一致；选 checkpoint 请以验证集 `eval_loss` 为准。

---

## 5. 权重托管（远程侧 2026-11-24 决策：**不进 git**）

远程侧已拍板：adapter（21.7 MB）**不进 Git LFS、不进普通 Git**，改用 ModelScope / HuggingFace 托管，本文件记录 SHA-256 与下载路径。仓库仍为 PUBLIC，因此**没有**任何权重被提交或上传。

### 5.1 待托管对象（唯一一份）

| 项 | 值 |
|---|---|
| 文件 | `checkpoint-26/adapter_model.safetensors` |
| 大小 | 21,700,560 B |
| SHA-256 | `a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6` |
| 配套 | `adapter_config.json`（目标模块正则、r8/α32、base_model 路径） |
| 同源证明 | 2026-11-24 重训产出的 ckpt-26 与本文件记录**逐字节相同**，见 `HANDOFF.md` §12.2 |

### 5.2 建议的托管方式

- 仓库名建议：`<owner>/xiaomi-ocr0-handwriting-lora`（公开或私有由 owner 定；若公开，注意 adapter 是在个人课程笔记上微调的）。
- 只上传 `adapter_model.safetensors` + `adapter_config.json`，**不上传训练数据与图片**。
- 上传后回填本节 §5.4 与 `HANDOFF.md` §12/§13 的链接。

### 5.3 上传命令（**本机当前无凭据，未执行**）

```bash
# ModelScope（本机已装 modelscope 1.40.2）
export MODELSCOPE_API_TOKEN=<token>
modelscope upload <owner>/xiaomi-ocr0-handwriting-lora \
  outputs/qlora-v2/v2data-20261124-124959/checkpoint-26/adapter_model.safetensors adapter_model.safetensors

# 或 HuggingFace
export HF_TOKEN=<token>
hf upload <owner>/xiaomi-ocr0-handwriting-lora \
  outputs/qlora-v2/v2data-20261124-124959/checkpoint-26/adapter_model.safetensors checkpoint-26/adapter_model.safetensors
```

校验（下载后必须一致）：

```bash
sha256sum adapter_model.safetensors
# 期望 a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6
python scripts/verify_adapter.py --adapter <下载目录> \
  --base outputs/modelscope_cache/models/SeerRay-Lab--Xiaomi-OCR-0/snapshots/master
# 期望 weights_verified=true，372/372，max_abs_diff=0.0
```

### 5.4 上传回填（待填）

| 项 | 值 |
|---|---|
| 托管地址 | _待填_ |
| 上传时间 / 上传人 | _待填_ |
| 下载后实测 SHA-256 | _待填_ |

### 5.5 状态

**未上传**。本机 `HF_TOKEN` / `MODELSCOPE_API_TOKEN` / `HUGGINGFACE_HUB_TOKEN` 均未设置；上传需要 owner 提供凭据，或由 owner 自行执行 §5.3 的命令。在此之前，拿到权重只能走 §1 的本地路径，或按 §4 重训复现（2026-11-24 已验证可逐字节复现）。


---

# Checkpoint 交付清单（checkpoint-12000，混训 x600）

> 状态：**confirmed — 当前 deliver 默认**（2026-11-27 全量盲测通过）。
> 域内正文 CER 0.8493（基线 0.875）、含公式 0.8478（0.854）、重复 0%；
> 全量 118 页配对盲测 0.3522 vs 0.5548（Δ −0.2026，91 胜 / 26 负 / 1 平），与 40 页抽样结论同向；
> 同一 40 页子集在 118 页数据上复算仍为 −0.1155，逐位可复现。
> 基线 ckpt-26 保留为回滚选项。

- 生成时间：2026-11-26
- 训练 run：`outputs/qlora-mixed/mixed-x600-20261126-121254`（Git 忽略）
- 交付 checkpoint：`checkpoint-12000`（14,000 步中的最佳域内点）
- 选定依据：域内正文 CER 0.8493（基线 ckpt-26 为 0.875）、重复 0%；全量 118 页配对盲测 Δ −0.2026（91/118 胜），40 页抽样 Δ −0.1155
- 权重落点：GitHub 私有仓库 Release
  `Komorebi-GZM/ocr-lora-artifacts` → `ocr-mixed-x600-ckpt12000`（Latest，draft=false）
- 配比：main 7,800（49.1%）/ CASIA 8,000（50.4%）/ GNHK 78（0.5%，本轮未有效测试）

## 校验值（三方一致：本地生成方 / SHA256SUMS.txt / 独立下载回读 2026-11-26）

| 文件 | 字节 | SHA-256 |
|---|---|---|
| `adapter_model.safetensors` | 21,700,560 | `20c55e75da39fc68cee7c88c80d381b6d5e232c30367d589792b1244d80752bc` |
| `adapter_config.json` | 1,335 | `dc47b2032522b0f4f354ed65b42d5ecd55f471b08372006db879ef3326bcf4f2` |

自检：

```bash
gh release download ocr-mixed-x600-ckpt12000 -R Komorebi-GZM/ocr-lora-artifacts
sha256sum adapter_model.safetensors adapter_config.json
# 期望 20c55e75… / dc47b203…
```

## 全量盲测核验（2026-11-27，118 页全量，替代 40 页抽样）

`OmniDocBench` note 子集全部 118 页，两臂同页配对、同协议 `deliver_v4`（rp1.1 / min0 / cap1024）：

| 指标 | ckpt-12000（混训） | ckpt-26（基线） | 差值 |
|---|---|---|---|
| 平均正文 CER | **0.3522** | 0.5548 | **−0.2026（相对 −36.5%）** |
| 中位 ΔCER | — | — | −0.0805 |
| 平均含公式 CER | **0.3493** | 0.5687 | −0.2194 |
| 逐页 胜/负/平 | **91 / 26 / 1** | — | — |
| 触顶页 | **0** | 3 | −3 |
| 判重复页 | **2** | 21 | −19 |
| CER > 1 的页 | **5** | 10 | −5 |
| CER > 0.5 的页 | **29** | 51 | −22 |

分语言：中文 70 页 0.4348 vs 0.6515（−0.2167）；中英混 47 页 0.2289 vs 0.4017（−0.1727）；英文 1 页 0.3636 vs 0.9816（−0.6179）。

**与 40 页抽样的一致性**：把 118 页结果限制到当初那 40 页，Δ = **−0.1155**，与 40 页实测逐位相同（同一协议、确定性解码），结论同向且全量效应更强。

**退化页（26 页，按 issue #19 要求只列 id 不下结论）**：中位退化 +0.032（很小）；显著退化 4 页——
`notes_9e951846094758afac08c620144e3a76_17.jpg`（+1.370，2.800 vs 1.430）、
`notes_f7f010b78016aeebd76e56d9283eb67f_119.jpg`（+1.196，1.221 vs 0.025）、
`notes_f7f010b78016aeebd76e56d9283eb67f_70.jpg`（+0.665，0.804 vs 0.138）、
`page-14cd673f-d86d-45a7-a13e-2b4e1d91c08f.png`（+0.519，1.373 vs 0.854）。
混合臂自身仍有 2 页判重复（`notes_1ba14cb325bc448f7201b20502ecf2b5_35.jpg`、`notes_f7f010b78016aeebd76e56d9283eb67f_49.jpg`），比基线 21 页少很多但非 0。

**运行记录**：混训臂 118 页 45.7 分钟（23.2 s/页，峰值 CUDA 预留 1.03 GiB，整卡占用中位 3087 MiB）；基线臂 61.7 分钟（31.4 s/页，峰值 1.09 GiB，3160 MiB）；两臂墙钟合计 **1 h 48 m 31 s**。

**逐页数据**：`blind118_paired.csv`（仓库根，118 行，含每页 CER、Δ、重复/触顶标志、字符数）。
复现：`python scripts/compare_blind_paired.py --manifest outputs/eval-mixed/blind118_abs.jsonl --a …blind118-mixed/pred.jsonl --b …blind118-base/pred.jsonl --csv …`
