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

### 5.3 实际托管方式：**私有 GitHub 仓库的 Release 资产**

ModelScope 路走不通：本机 `~/.modelscope/credentials/session` 里存的 token 已失效
（`login` 返回 `[E3001] AccessToken错误`），`HF_TOKEN` 也未设置。为了不把权重推进公开的
`wheatfield_braintrust`，改用 **gh 已登录的 `repo` 权限**，建了一个**私有**仓库专门放权重：

| 项 | 值 |
|---|---|
| 仓库 | `Komorebi-GZM/ocr-lora-artifacts`（**private**） |
| Release | `ocr-ckpt26-v1` |
| 页面 | https://github.com/Komorebi-GZM/ocr-lora-artifacts/releases/tag/ocr-ckpt26-v1 |
| 资产 | `adapter_model.safetensors`（21,700,560 B）、`adapter_config.json`、`SHA256SUMS.txt` |
| 上传时间 | 2026-11-25 20:21（本地） |
| 上传后 GitHub 计算的 digest | `sha256:a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6` —— 与本地文件一致 |

取回：

```bash
gh release download ocr-ckpt26-v1 --repo Komorebi-GZM/ocr-lora-artifacts --dir ./ckpt26
cd ckpt26 && sha256sum -c SHA256SUMS.txt
# adapter_model.safetensors: OK
```

没有 gh 时（需要仓库访问权限）：

```bash
curl -L -H "Authorization: token <github-token>" \
  -o adapter_model.safetensors \
  https://github.com/Komorebi-GZM/ocr-lora-artifacts/releases/download/ocr-ckpt26-v1/adapter_model.safetensors
```

校验：

```bash
sha256sum adapter_model.safetensors
# 期望 a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6
python scripts/verify_adapter.py --adapter <下载目录> \
  --base outputs/modelscope_cache/models/SeerRay-Lab--Xiaomi-OCR-0/snapshots/master
# 期望 weights_verified=true，372/372，max_abs_diff=0.0
```

### 5.4 若以后要迁到 ModelScope / HuggingFace

两条命令仍然可用，只需先在对应平台登录（本机当前无有效凭据）：

```bash
export MODELSCOPE_API_TOKEN=<token>
modelscope upload <owner>/xiaomi-ocr0-handwriting-lora adapter_model.safetensors adapter_model.safetensors

export HF_TOKEN=<token>
hf upload <owner>/xiaomi-ocr0-handwriting-lora adapter_model.safetensors checkpoint-26/adapter_model.safetensors
```

迁移后请把地址补进 §5.3 的表，并保留同一份 SHA-256。

### 5.5 状态

**已托管（私有）**：`ocr-ckpt26-v1`，GitHub 端 digest 与本地逐字节一致。仓库为 **private**，因此下载需要 `Komorebi-GZM` 的访问权限；`wheatfield_braintrust` 本身仍是 PUBLIC，但**没有任何权重进入该仓库的 Git 历史**。
