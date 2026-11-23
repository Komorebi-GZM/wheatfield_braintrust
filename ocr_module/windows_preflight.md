# Windows 训练前检查

- 当前环境：Windows，RTX 4060 Laptop 8188 MiB，PyTorch 2.6.0+cu124、transformers 5.14.1、bitsandbytes 0.50.2、ms-swift 4.6.0；CUDA 可用，Swift SFT 流水线可导入。
- Xiaomi-OCR-0 的 `architectures` 为 `Qwen3_5ForConditionalGeneration`。当前 ms-swift 将该架构匹配到 `qwen3_5`、`ovis_ocr2`、`qwen3_5_emb`、`wemm_embedding`，仅指定模板不足以消除歧义；训练脚本现显式指定 `--model_type qwen3_5 --template qwen3_5`。
- 已安装 `qwen-vl-utils 0.0.14`。使用本地缓存和显式模型类型，ms-swift 的 `get_model_processor(..., load_model=False)` 成功返回 `Qwen3VLProcessor`。这没有加载训练模型或执行训练。
- Swift 仍提示缺少 `decord`，其用途涉及视频；本任务是静态图像，目前只证明处理器可加载，不证明后续训练不会要求它。
- 已通过 `scripts/probe_swift_dataset.py` 在 `qwen3_5` 训练模板、`max_length=8192`、`max_pixels=1024000` 下编码 13 张训练页和 5 张验证页，均生成视觉张量与非空监督标签；总 token 数 1144～1615，未触发长度限制。原图不修改；图像网格为 `[1,72,54]` 或 `[1,54,72]`，按 patch size 16 对应 864×1152 或 1152×864 的处理器输入。
- 已在最长训练样本 `IMG_20261104_200903.jpg`（1615 token）上用 Swift 模板和 4-bit NF4、双重量化、rank 8 LoRA、冻结视觉塔及梯度检查点，做一次不包含优化器步骤的前向/反向探测：损失约 1.4604，372 个可训练张量获得梯度，PyTorch 峰值 allocated 6,579,958,272 字节、reserved 7,017,070,592 字节（WDDM 统计不能简单等同独占物理显存）。未更新或保存权重；优化器状态、验证、完整 Swift Trainer 流程及其他后台显存占用未包含在上述结果中，不能据此断言完整训练稳定。
- 又运行了包含 AdamW 状态分配的同样探测，学习率与权重衰减均为 0，372 个状态条目成功创建；峰值 allocated 6,579,958,272 字节、reserved 7,017,070,592 字节。未保存权重，这不覆盖完整 Trainer、验证和系统显存竞争。
- 探测初次调用 data collator 时因未绑定模型触发内部临时模型识别歧义；Swift SFT 源码会在 `template.use_model` 时绑定已有模型，探测按相同方式绑定后通过。Windows 多进程数据加载路径仍待实际验证。
- 新增 PowerShell 入口 `scripts/train_qlora.ps1`；`-Inspect` 已验证参数和输入文件，采用 0 个数据加载子进程。由于训练集只有 13 页，保存和验证改为每 epoch，日志每步输出；尚未运行完整 Swift Trainer 验证或训练。
- `feat/ocr-xiaomi-qlora` 已合入 `pr-1` + `fix/ocr-legacy-path-portability`（`fce9112`）。在本机按真实清单逐条解析：train 13、val 5、test 5、train_swift 13、val_swift 5，共 23 页全部命中本地图片（`resolve_image_path`，历史 `E:\...` 绝对路径也能落到 `datasets/handwriting_notes/images/`），0 缺失。
- **行尾与冻结哈希（重要）**：本机 `core.autocrlf=true`。仓库里 `annotation_template.jsonl` 的 blob 是 LF，冻结哈希 `34f262…` 按 LF 计算；检出成 CRLF 后，`data_version.py` 直接按字节算出 `62e520…`。`splits/*.jsonl` 则反之——提交时就带 CRLF，冻结值本就按 CRLF 计算。因此在本机重跑 `data_version.py` 会改掉清单里的哈希而**没有任何内容变化**：不要用重新生成的 `data_version_v2.json` / `.sha256` 覆盖仓库版本。判断数据是否被动过要看内容级 checks（`image_count_23`、`splits_disjoint`、`splits_cover_all_images`、`split_images_exist`、`markdown_equals_annotation`、`swift_images_in_annotation`、`membership_unchanged_vs_v1`，本机 7 项全为 true），不要看哈希是否相等。若要哈希跨平台可复现，需要单独决定统一行尾并重算一次清单——那会改动 splits 字节，属数据变更，应先取得 owner 明确同意。
