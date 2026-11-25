# OmniDocBench note 子集（评测专用，不入训练）

来源 opendatalab/OmniDocBench（1651 页基准，仅取 data_source=note 的 118 页手写笔记：
70 简体中文 + 47 中英混排 + 1 英文）。人工标注，公式为 LaTeX 文本，与主域同构。

**用途红线：仅作第三方评测（陌生字迹泛化盲测），禁止入训练集**（公开基准有泄漏风险）。

复现：`download_rest.py`（HF tree API 分页拉全 1651 images）→ `convert_omnidocbench.py`。
raw/ 与 images/ 均不入 git。
