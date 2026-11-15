# 推理诊断报告 v2（基座 vs checkpoint-4）

日期：2026-11-15
阶段：推理协议对齐 + 长生成问题定位（**未做任何训练**，仅小规模诊断推理）
权威配置：`inference_protocol_v2.json`

---

## 0. 结论摘要

| 问题 | 结论 | 证据 |
|---|---|---|
| 旧推理流程与训练是否一致 | **不一致**：旧流程带 `SYSTEM_PROMPT`，从第 **1** 个 token 起就分叉，并多 240 token | `outputs/protocol/probe_IMG_20261104_200903.json` → `A_vs_B_first=1`、`A_vs_B_len_delta=240` |
| 新流程是否与训练逐 token 一致 | **一致**：prompt 1008 token 与 Swift 训练渲染完全相同，assistant 头（含空 think 块）也相同 | 同上 → `prompt_identical=true`、`C_ends_with_A_header=true`、`A_same_header=true` |
| EOS / 停止条件 | `im_end`(248046) 即 EOS；训练监督序列以 `<|im_end|>\n` 结尾；生成到 im_end 停止 | 同上 → `im_end_equals_eos_token_id=true`、`A_second_last_is_im_end=true` |
| adapter 是否真的挂上且未丢权重 | **是**（5 项独立检查全过：372/372 键匹配、max_abs_diff=0、PEFT 无缺失 LoRA 键、372 个 LoRA 模块全在 language_model 内、186 个 lora_B 全非零） | `outputs/protocol/adapter_check.json` → `weights_verified=true` |
| 训练/推理图像预算是否一致 | **已对齐**：双方都用 `max_pixels=1024000` → grid `[1,54,72]`、3888 patch、**972 视觉 token** | `probe` + 各 case JSON |
| 旧的“全分辨率推理”差多少 | grid `[1,192,256]`、49152 patch、**12288 视觉 token**，是训练的 **12.64×** | `d3_ft_train_orig_1024.json` |
| 长生成/退化的根因 | **不是**尺度、**不是** prompt、**不是**上限太小；是 **checkpoint-4 自身退化**（数字递增式模板重复），基座只在 1/5 页退化 | 见第五节 5×2 矩阵 |
| 512 上限是否够 | **不够**：最长非测试参考答案 611 token，基座正常完整输出最长 **633** token（eos、无重复） | `outputs/protocol/supervised_tokens.json`、`d1_*` |
| 上限取多少 | **1024**（相对 611 有 1.6× 余量），全程用非测试页决定，**未参考任何测试集成绩** | 同上 |
| 是否已具备完整对比评测条件 | **是**，但有 1 个前提见第七节（需接受“退化页会被如实计入 CER”） | — |

一句话：**协议已经可以冻结复现；长生成的锅不在推理管线，在 4 步 LoRA 训出来的 adapter。**

---

## 一、消息与生成协议核验

### 1.1 三路渲染对比（CPU，不加载权重）

| 变体 | 说明 | token 数 | 是否与训练一致 |
|---|---|---|---|
| A | ms-swift `qwen3_5` 模板 `mode=train`，即训练真正看到的序列 | 1012 | 基准 |
| B | **旧**仓库推理：system + 列表内容块 | 1252 | ✗ 第 1 token 处即分叉，多 240 token |
| C | **新**推理：无 system，单 user 轮，`"\n"+INSTRUCTION` | 1008 | ✓ 与 A 前缀逐 token 相同 |

A 比 C 多的 4 个 token 正是答案本身（`占位` + `<|im_end|>` + 换行），即 C 的 prompt 恰好停在答案起点之前。

### 1.2 assistant 头与思考模式

模板生成的 assistant 头是：

```
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

- 训练侧与推理侧**完全相同**（`A_same_header=true`）。
- 空 `<think>...</think>` 块意味着**非思考模式**；两侧一致，因此不需要在推理时额外关 thinking。
- 结论：**不要**为了“对齐”而加 system prompt。旧默认带 system 是历史包袱，已改为默认关闭。

### 1.3 EOS 与停止条件

| 项 | 值 |
|---|---|
| 特殊 token id | `image_pad`=248056、`vision_start`=248053、`im_end`=248046、`endoftext`=248044 |
| EOS | `<|im_end|>` = 248046，与 `eos_token` 同 id |
| 训练监督序列尾 | `答案<|im_end|>\n`（`A_second_last_is_im_end=true`、`A_last_is_newline=true`） |
| 推理停止 | 生成到 248046 即停；尾随换行不参与生成 |
| 判定“正常结束” | **只认 `last_token_id == 248046`**；命中上限一律记为 `length`，不当作正常结束 |

### 1.4 adapter 挂载核验（重点：不靠“加载成功”）

原来的校验把磁盘键 `base_model.model.model.language_model...lora_A.weight` 与运行时键
`...lora_A.default.weight` 直接比对，导致 372/372 全“缺失”，**校验本身是坏的**。修正为键名归一化后：

| 检查 | 结果 | 判定 |
|---|---|---|
| 归一化键匹配 | 372 命中 / 0 缺失 / 0 多余 | ✓ |
| 数值一致性 | `max_abs_diff = 0.0` | ✓ |
| PEFT 自身加载器 | 缺失 LoRA 键 0、多余键 0（缺的 474 个是未提供的 4bit 基座权重，符合预期） | ✓ |
| 模块落点 | 372 个 LoRA 模块，**0 个在 language_model 之外**（与 `freeze_vit/freeze_aligner` 一致） | ✓ |
| lora_B 是否被训动 | 186 个 lora_B 全部非零（`lora_B_all_zero=0`） | ✓ |
| 参数量 | 盘上 5,411,328 = 运行时 5,411,328 | ✓ |
| eval / no_grad | `model_training_mode=false`、`any_param_training=false`、generate 包在 `torch.no_grad()` | ✓ |

`trainable_params=0` 是 `is_trainable=False` 的正常结果，**不能**当作“没加载”。

训练日志里的 `PeftModelForCausalLM: 609.5898M Params (5.4113M Trainable [0.8877%])` 与磁盘 5,411,328 完全吻合，是独立的第三方交叉验证。

---

## 二、图像处理对齐

- 唯一旋钮：`processor.image_processor.size["longest_edge"] = max_pixels`，双方共用一个显式配置，默认 `1024000`（= 训练值）。
- 磁盘原图**未被修改**；只控制送入模型的张量尺寸。

| | 训练 / 新推理 | 旧推理（全分辨率） | 比值 |
|---|---|---|---|
| 原图 | 4096×3072 | 4096×3072 | — |
| 实际处理尺寸 | 1152×864 | 3072×4096 | — |
| `image_grid_thw` | `[1,54,72]` | `[1,192,256]` | — |
| **patch 数** | **3888** | **49152** | 12.64× |
| **合并后视觉 token 数** | **972** | **12288** | 12.64× |
| prompt token | 1008 | 12324 | 12.2× |

`patch_size=16`、`merge_size=2`，视觉 token = `patch 数 / 4`。**patch 数与视觉 token 数分开记录，不混称**（旧报告里“3888 token”的说法是错的，3888 是 patch 数）。

> patch 数 → 视觉 token 数的换算已在 `ocr/engine.py` 里显式实现为两个字段 `n_patches` / `n_visual_tokens`。

**尺度只是假设，不是结论**：第三节 d3 用例把“全分辨率”单独跑了一次，结果**没有**出现重复（见 5.2），因此“尺度导致重复”这个假设被证伪。

---

## 三、逐页运行证据

每个用例都落一个 JSON，内含：图像路径+SHA-256、数据版本、基座 id、adapter 路径+权重 sha、prompt 及其 sha、`max_pixels`、原始/实际尺寸、`image_grid_thw`、`max_new_tokens`、生成 token 数、末 token、停止原因、是否触上限、重复检测全部规则与样例、**完整原始生成文本**、markdown、耗时、CUDA 峰值 allocated/reserved、整卡占用前后值。

### 3.1 单因子用例（训练页 IMG_20261104_200903 / 验证页 235932）

| 用例 | 模型 | 尺度 | grid / 视觉 token | 生成 | 停止 | 重复 | 字符 | 秒 | 进程峰值 allocated / 整卡 |
|---|---|---|---|---|---|---|---|---|---|
| d1 | 基座 | 1MP | `[1,54,72]` / 972 | 633 | **eos** | 否 | 901 | 39.9 | 989 MiB / 2935 MiB |
| d2 | +ckpt-4 | 1MP | `[1,54,72]` / 972 | **1024** | **length** | **是**（模板行 23 次、ngram 覆盖 41.8%） | 2281 | 102.5 | 1009 MiB / 2955 MiB |
| d3 | +ckpt-4 | 原尺寸 | `[1,192,256]` / 12288 | 811 | **eos** | 否 | 1305 | 150.9 | **13010 MiB** / 7473 MiB |
| d4 | +ckpt-4 | 1MP | `[1,72,54]` / 972 | 505 | **eos** | 否 | 782 | 48.1 | 1009 MiB / 2955 MiB |

对照物：该训练页参考答案 611 token；同页基座正常输出 633 token。

### 3.2 验证集 5 页 × 双臂（同一进程、同一 cap=1024、同一 max_pixels）

| 臂 | 页面 | 参考 token | 生成 | 停止 | 重复（新规则） | 字符 | 秒 |
|---|---|---|---|---|---|---|---|
| base | 235932 | 529 | 486 | eos | 否 | 754 | 28.6 |
| base | 235946 | 470 | 347 | eos | 否 | 430 | 20.0 |
| base | 235955 | 508 | **1024** | **length** | **是**（unit 163、ngram 覆盖 1.00） | 1791 | 56.8 |
| base | 000005 | 376 | **45** | eos | 否（属于**提前停**） | 69 | 3.1 |
| base | 000012 | 458 | 421 | eos | 否 | 696 | 22.5 |
| ft | 235932 | 529 | 505 | eos | 否 | 782 | 51.2 |
| ft | 235946 | 470 | 621 | eos | **是**（unit 11） | 741 | 59.9 |
| ft | 235955 | 508 | **1024** | **length** | **是**（模板行 7、ngram 0.25） | 1804 | 110.3 |
| ft | 000005 | 376 | **1024** | **length** | **是**（`## 循环结构：RCL` × 105） | 1557 | 111.2 |
| ft | 000012 | 458 | 485 | eos | **是**（ngram 覆盖 0.27） | 676 | 62.4 |

汇总：base = eos 4/5、触上限 1/5、退化 1/5、总 131.0 s；ft = eos 3/5、触上限 2/5、**退化 4/5**、总 395.0 s。

> 表中重复标记是用**修正后**的规则对已存 `raw_output` 重算的结果（`outputs/diagnosis/summary_val_final.json`、`repetition_recheck.json`），原始 JSON 里保存的 `rep_detected` 是修正前规则下的值，两者差异已在 4.3 说明。

### 3.3 内存语义（重要）

- `peak_cuda_allocated` / `peak_cuda_reserved`：**本进程**经由 torch 分配器的峰值。
- `gpu_memory_used_*_mib`：`nvidia-smi memory.used`，**整卡**占用（含浏览器、桌面等其它进程），Windows WDDM 下**不等于**本进程独占显存。
- 硬证据：d3 的 `peak_cuda_allocated = 13010 MiB`，而本机物理显存只有 **8188 MiB**。数字超过物理上限，说明它在 WDDM 下借用了共享内存 —— 因此**任何“GiB 数”都不能直接读成物理显存占用**。

---

## 四、后处理与续跑修正

### 4.1 发现的两个格式错配

| 现象 | 证据 |
|---|---|
| 基座输出整体包在 `<md>...</md>` 里 | `d1`、`val_base_*`，`wrapper=md_tag` |
| 微调输出整体包在 ```markdown 围栏里 | `val_ft_*` |
| **冻结标注是纯 Markdown，无任何包装** | 23 页统计：`<md>` 0 处、围栏 0 处 |
| 标注用 `$...$` / `$$...$$`；**两个模型都输出 `\[...\]` / `\(...\)`** | 标注 `$$` 70 对、`$` 238 个，`\[` **0** 个 |

后果：不修的话，包装会被当成正文计入 CER，而 `extract_formulas` 一个公式都取不到（公式指标直接归零）。

### 4.2 修正（两侧完全相同的确定性归一化）

`ocr/postprocess.py::normalize_output`：

1. 剥离**一个**整体外层包装：```fence``` 或 `<md>...</md>`；内部代码块保留；**未闭合的包装不剥离**（截断时围栏没闭合，剥了反而破坏信息）。
2. 统一公式定界符：`\[..\]` → `$$..$$`，`\(..\)` → `$..$`（只换定界符，不动内容）。
3. 幂等，`raw_output` 永远单独保存原始生成，不覆盖。

`scripts/evaluate.py` 对**预测与参考同时**调用同一函数，保证“两侧同样的后处理”。
单测：`tests/test_postprocess_wrapper.py`（18 项，含真实失败样例）；全量 33 项通过。

### 4.3 重复检测规则（公开、有样例）

| 规则 | 定义 | 来源 |
|---|---|---|
| `line_run` | 同一非空行连续 ≥5 次 | 基础规则 |
| `unit_run` | 去掉表格行后，2–40 字符单元连续重复 ≥5 次 | 235955 的 `课程代码 | ` × 163 |
| `template_line_run` | 连续 ≥5 行**仅数字不同**（数字替换为 `#` 后相同），表格行不参与 | d2 的 `W_{12}..W_{33}` 同式递增 |
| `ngram` | 10-gram 出现 ≥20 次且覆盖率 ≥0.25 | d2（91 次 / 41.8%）、235955（208 次 / 100%） |

两处精度修正（有真实样例支撑，不是为了调分）：

- `unit_run` 新增“排除表格行”：235946 的正常宽表分隔行 `| ---` 重复 12 次被误判 → 修正后不再误报。
- `template_line_run` 同理排除表格行，避免合法等结构表格误报。

修正前后对同一批已存输出的判定变化（`repetition_recheck.json`）：只有 235946 由“是”变“否”（去误报），其余全部一致。

**规则边界说清楚**：命中上限 ≠ 失败（d2 触上限且退化，但 `base/000005` 的 45 token 是“提前停”，规则正确地没标它）；退化必须有样例才认。

### 4.4 续跑

`scripts/infer_test_baseline.py`：

- 落盘即写 UTF-8、`newline="\n"`，**每页写完就 flush**；
- 重跑同命令自动跳过已完成页；
- 旁边写 `.protocol.json` 指纹（基座、adapter 目录 + 权重 sha + `weights_verified`、`max_pixels`、`max_new_tokens`、`use_system_prompt`、instruction sha、manifest sha、数据版本标注 sha、后处理版本）；
- **指纹不一致直接拒绝续写**，不同配置的预测永远不可能被拼到一个文件里；
- adapter 未通过权重校验时**拒绝开始**（不是警告，是退出）。

---

## 五、诊断

原则：只用 train/val 页、一次只改一个因子、GPU 串行、同进程对照。

### 5.1 假设裁决

| 假设 | 裁决 | 依据 |
|---|---|---|
| 提示词/模板不一致导致跑飞 | **否** | 1.1/1.2：prompt 已逐 token 相同，改完仍退化 |
| adapter 没挂对 / 权重丢失 | **否** | 1.4：5 项检查全过 |
| 图像尺度（12.6× 放大）导致重复 | **否（已证伪）** | d3 用原尺寸推理：**无重复、正常 eos**，只是慢到 150.9 s；同时显存被压到 7473/8188 |
| 生成上限太小导致“看起来长” | **否** | 触上限的两页参考答案只有 508 / 376 token；若真需要更长，参考不会这么短 |
| **checkpoint-4 自身退化** | **是（主因）** | 同页同协议，ft 退化 4/5，base 退化 1/5；ft 触上限 2/5 vs base 1/5；ft 速度约 2× 慢 |

### 5.2 逐个因子

- **尺度因子**（d2 vs d3，同页同模型同 cap）：1MP → 退化 1024 token；原尺寸 → 正常 811 token eos。尺度变大**没有**诱发重复，反而“变好”了一点（视觉信息更多），代价是耗时 1.47×、显存压力极大。→ 训练对齐的 1MP 是正确选择，它既是训练分布，也更省。
- **模型因子**（d1 vs d2，完全同协议）：基座 633 token 正常收尾，微调 1024 token 退化。差异只能来自 adapter。
- **页面因子**：235955 两臂都退化（base ngram 覆盖 100%）→ 该页存在与模型无关的内容/格式病理（密集无结构表格）。其余 3 页只有 ft 退化 → adapter 诱发。
- **退化形态**：d2 是“公式只变下标”（`W_{12}`…`W_{33}`）；000005 是“标题只重复同一行”（`## 循环结构：RCL` × 105）；235955 是“表格单元填满整页”。三种形态都被规则抓到，且都带样例。

### 5.3 顺带发现的另一类失败

`base/000005`：45 token、69 字符、**正常 eos**，但参考答案 376 token —— 这是**提前停止**（输出严重不全），不是重复。它提醒：只看“是否触上限”会漏掉这一类，必须同时看“生成长度 / 参考长度”的比值。本次未针对它做修复（超出本阶段范围），仅记录。

---

## 六、训练记录的解释修正

### 6.1 `warmup_steps=0` 不等于“没有 warmup” —— 但实际确实几乎没有

- `args.json` 写着 `warmup_steps=0`、`warmup_ratio=0.03`。
- 本环境 transformers 5.14.1 仍保留 `warmup_ratio` 字段（已弃用），其 `__post_init__` 执行 `warmup_steps = warmup_ratio`。**实测**（直接实例化 ms-swift 的 `TrainingArguments`）：

  | 传入 | 运行时 `warmup_steps` | `get_warmup_steps(4)` |
  |---|---|---|
  | `warmup_ratio=0.03, warmup_steps=0` | **0.03** | **1** |
  | `warmup_steps=0` | 0.0 | 0 |

  即 `args.json` 里的 `warmup_steps: 0` 是**落盘值失真**，运行时是 0.03，按比例算出 **1 步** warmup。
- 但真实 LR 日志（`checkpoint-4/trainer_state.json`）显示第 1 步就用了满值：

  | step | learning_rate | 占峰值 |
  |---|---|---|
  | 1 | 1.0e-4 | 100% |
  | 2 | 7.5e-5 | 75% |
  | 3 | 2.5e-5 | 25% |
  | 4 | 0.0 | 0% |

- 结论：**总步数只有 4 步时，3% 的 warmup 退化成 1 步、且第 1 步就吃满 LR，等效于没有 warmup**。这个结论不是从 `warmup_steps=0` 推的，是从 LR 日志读出来的。

### 6.2 调度曲线与步数记账存在一处未解释的偏差（如实记录）

实测 LR 比例序列 1.00 / 0.75 / 0.25 / 0.00 恰好是 **cosine 在 3 个区间上取点**（`0.5(1+cos(πk/3))`）的形状，而 `trainer_state.global_step=4`。也就是说调度器内部的总步数与训练器记录的 optimizer step 相差 1。

- 影响：LR 起点满值、终点归零这两个关键事实不变，**不改变本报告任何结论**。
- 状态：**未查清**，列为遗留问题（不影响数据/评测，只影响对训练曲线的精细解读）。

### 6.3 8.85 GiB 不能读成物理显存

- 日志字段是 ms-swift 的 `memory(GiB)`。
- 本机物理显存 **8188 MiB = 8.0 GiB < 8.85 GiB**。数字超过物理上限，说明该指标不是物理占用（WDDM 会借共享内存）。
- 加上 3.3 的 d3 证据（进程峰值 13010 MiB），“GiB 数 = 独占显存”这个读法被彻底否掉。
- 正确读法：**它是进程侧分配器统计的峰值，只能用于同机同配置横向比较**。

### 6.4 这次微调实际的规模（比表面看起来小很多）

| 项 | 值 |
|---|---|
| 训练页数 | 13（train split） |
| `per_device_train_batch_size` / `grad_accum` | 1 / 16 |
| epoch | 4 |
| **optimizer step 总数** | **4** |
| 可训练参数 | 5.41 M（占 0.89%） |
| `freeze_vit` / `freeze_aligner` | True / True |
| 各 epoch 验证 loss | 1.3224 → 1.2778 → 1.2533 → **1.2456**（best = step 4 = checkpoint-4） |

也就是说 checkpoint-4 是**只做了 4 次参数更新**的最优检查点。它足以改变输出风格（换用围栏、加标题层级），但不足以稳定收敛输出长度 —— 这与观察到的“ft 更容易跑飞”完全吻合，也解释了为什么输出仍沿用基座的 `\[...\]` 公式约定而没学会标注的 `$$`。

`max_length=8192`、`truncation_strategy=delete`：最长样本 1615 token，**没有任何样本被删除**（否则 optimizer step 会变成 3）。

---

## 七、结论：是否具备完整对比评测条件

**具备。** 前提只有一条：接受“退化页产生的超长/重复输出会如实进入 CER”，不做人工剔除——因为剔除就变成挑样本，破坏可比性。

已经就绪的部分：

1. 数据冻结（v2），test.jsonl 成员与 v1 逐字节一致；
2. 基座与 checkpoint-4 走**同一个** `OCREngine`、同一 prompt、同一 `max_pixels=1024000`、同一 `max_new_tokens=1024`、同一后处理；
3. adapter 权重正确性有 5 项证据；
4. 每页证据可回溯（图像 sha、prompt sha、grid、token 数、停止原因、重复样例、耗时、内存）；
5. 评测脚本对两侧施加同一归一化，`□` 页面按既有规则整体排除并计入 excluded；
6. 续跑与配置隔离有硬约束（指纹不符即拒绝）。

建议的完整评测命令（**尚未执行，等确认**）：

```bash
# 基座
python scripts/infer_test_baseline.py \
    --manifest datasets/handwriting_notes/splits/test.jsonl \
    --output outputs/baseline/test_pred_v2.jsonl \
    --max-pixels 1024000 --max-new-tokens 1024

# checkpoint-4
python scripts/infer_test_baseline.py \
    --manifest datasets/handwriting_notes/splits/test.jsonl \
    --output outputs/finetuned/test_pred_v2.jsonl \
    --adapter outputs/qlora-v1/v2-20261115-114258/checkpoint-4 \
    --max-pixels 1024000 --max-new-tokens 1024

# 两边同一评测
python scripts/evaluate.py --pred outputs/baseline/test_pred_v2.jsonl \
    --ref datasets/handwriting_notes/splits/test.jsonl --out outputs/eval/baseline_v2
python scripts/evaluate.py --pred outputs/finetuned/test_pred_v2.jsonl \
    --ref datasets/handwriting_notes/splits/test.jsonl --out outputs/eval/finetuned_v2
```

预算：测试 5 页 × 2 臂，按诊断实测（ft 平均 79 s/页、基座 26 s/页）约 **10 分钟** GPU，可串行一次跑完。

**注意**：一旦开跑就不要中途换 `max_pixels` 或上限；换任何一项都必须换新的 `--output` 路径（脚本会用指纹强制这一点）。

---

## 八、遗留问题（不阻塞评测）

1. 调度器总步数与 `global_step` 差 1（6.2），未查清，不影响结论。
2. `base/000005` 的**提前停止**（45 token）没有专门处理；如果要提升整体 CER，这是与“退化”并列的第二类失败模式。
3. 标注用 `$`/`$$`、模型输出 `\[`/`\(`：目前靠后处理统一。若后续要继续微调，应显式在监督数据与提示词里把定界符约定讲清，或直接接受并在评测侧统一（现状）。
4. 4 步 LoRA 显然欠训练。若要做“微调是否有效”的结论，需要一次正常的训练（例如按 `len(train)` 正确计算步数、至少几百步）；本阶段不做。

---

## 附：本阶段改动的文件

| 文件 | 改动 |
|---|---|
| `ocr/engine.py` | 重写 v2 协议；`attach_adapter` 改为 5 项可验证检查；`normalize_output` 接入；重复规则加 `template_line_run`、`unit_run` 排除表格行 |
| `ocr/prompts.py` | 默认无 system；单 user 轮 + `"\n"+INSTRUCTION` |
| `ocr/postprocess.py` | 新增 `strip_md_tag`、`normalize_math_delimiters`、`normalize_output`（幂等） |
| `scripts/evaluate.py` | 两侧同一归一化；错误页 dump 用归一化文本 |
| `scripts/infer_test_baseline.py` | 指纹加 `postprocess_version` / `weights_verified`；未通过权重校验即拒跑 |
| `scripts/probe_protocol.py` | 新增 assistant 头 / EOS / thinking 检查 |
| `scripts/verify_adapter.py` | 新增，单跑 adapter 校验 |
| `scripts/diagnose_sample.py` | 新增，非测试集整集采样（base/ft 同进程对照） |
| `scripts/diagnose_inference.py` | 适配新 postprocess 字段 |
| `tests/test_postprocess_wrapper.py` | 新增 18 项（包装/定界符/重复规则，含真实样例） |
| `tests/test_messages.py` | 改为 v2 prompt 契约 |
| `inference_protocol_v2.json` | 新增，冻结配置 |
