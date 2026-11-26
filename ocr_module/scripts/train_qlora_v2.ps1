param(
    [string]$BaseModel = 'SeerRay-Lab/Xiaomi-OCR-0',
    [string]$OutputRoot = 'outputs/qlora-v2',
    [int]$Epochs = 4,
    [int]$GradAccum = 1,
    [int]$MaxSteps = -1,
    [string]$TrainManifest = 'datasets/handwriting_notes/splits/train_swift.jsonl',
    [string]$ValManifest = 'datasets/handwriting_notes/splits/val_swift.jsonl',
    [string]$RunTag = '',
    [int]$SaveSteps = 0,
    [switch]$Inspect
)

# 修正版 QLoRA（v2 数据）。与 v1 跑的唯一差异：
#   gradient_accumulation_steps 16 -> 1
# v1 用 13 个训练样本 + accum 16 时，每个 epoch 只有 1 个 optimizer step，
# 4 个 epoch 总共 4 步，warmup 退化成 1 步、LR 到第 4 步就归零，等于没训。
# 其余超参与 v1 完全一致，避免夹带调参。

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $env:MODELSCOPE_CACHE) { $env:MODELSCOPE_CACHE = Join-Path $projectRoot 'outputs/modelscope_cache' }
$swift = Join-Path (Split-Path -Parent $projectRoot) '.venv/Scripts/swift.exe'
$train = $TrainManifest
$validation = $ValManifest
foreach ($path in @($swift, $train, $validation)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing required file: $path" }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$prefix = if ($RunTag) { $RunTag } else { 'v2data' }
$runDir = Join-Path $OutputRoot "$prefix-$stamp"
$saveStrategy = if ($SaveSteps -gt 0) { "steps" } else { "epoch" }
$saveStepArg = if ($SaveSteps -gt 0) { "--save_steps $SaveSteps" } else { "" }
if (-not $Inspect) {
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    # Windows 上 ms-swift 的 update_last_checkpoint_symlink 会：
    #   os.symlink -> WinError 1314（无创建符号链接权限）-> 回滚时 os.remove 一个目录
    #   -> PermissionError [WinError 5]（Windows 不允许 os.remove 删目录）-> 训练在第 1 个
    #   save 时崩溃。预先放一个同名空目录，让它走 "already exists and is not a symlink" 的
    #   早退分支，从而完全不依赖 last-checkpoint 符号链接（本项目的 checkpoint 一律用真实
    #   的 checkpoint-N 目录名引用）。
    New-Item -ItemType Directory -Force -Path (Join-Path $runDir 'last-checkpoint') | Out-Null
    $cmdLines = @(
        "swift sft --model $BaseModel --model_type qwen3_5 --template qwen3_5",
        "--tuner_type lora --quant_method bnb --quant_bits 4",
        "--bnb_4bit_quant_type nf4 --bnb_4bit_use_double_quant true --bnb_4bit_compute_dtype bfloat16",
        "--torch_dtype bfloat16 --dataset $train --val_dataset $validation --split_dataset_ratio 0",
        "--max_length 8192 --truncation_strategy delete --max_pixels 1024000",
        "--per_device_train_batch_size 1 --gradient_accumulation_steps $GradAccum --num_train_epochs $Epochs",
        "--learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.03",
        "--lora_rank 8 --lora_alpha 32 --lora_dropout 0.05 --target_modules all-linear",
        "--gradient_checkpointing true --freeze_vit true --freeze_aligner true",
        "--eval_strategy epoch --save_strategy $saveStrategy $saveStepArg --save_total_limit 4 --logging_steps 1",
        "--dataloader_num_workers 0 --seed 42 --data_seed 42 --max_steps $MaxSteps --add_version false",
        "--output_dir $runDir"
    )
    Set-Content -LiteralPath (Join-Path $runDir 'command.txt') -Value $cmdLines -Encoding utf8
    Write-Output "RUN_DIR=$runDir"
}

$arguments = @(
    'sft', '--model', $BaseModel, '--model_type', 'qwen3_5', '--template', 'qwen3_5',
    '--tuner_type', 'lora', '--quant_method', 'bnb', '--quant_bits', '4',
    '--bnb_4bit_quant_type', 'nf4', '--bnb_4bit_use_double_quant', 'true',
    '--bnb_4bit_compute_dtype', 'bfloat16',
    '--torch_dtype', 'bfloat16',
    '--dataset', $train, '--val_dataset', $validation, '--split_dataset_ratio', '0',
    '--max_length', '8192', '--truncation_strategy', 'delete', '--max_pixels', '1024000',
    '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', "$GradAccum",
    '--num_train_epochs', "$Epochs",
    '--learning_rate', '1e-4', '--lr_scheduler_type', 'cosine', '--warmup_ratio', '0.03',
    '--lora_rank', '8', '--lora_alpha', '32', '--lora_dropout', '0.05',
    '--target_modules', 'all-linear',
    '--gradient_checkpointing', 'true', '--freeze_vit', 'true', '--freeze_aligner', 'true',
    '--eval_strategy', 'epoch', '--save_strategy', $saveStrategy, '--save_total_limit', '4',
    '--logging_steps', '1', '--dataloader_num_workers', '0',
    '--seed', '42', '--data_seed', '42', '--max_steps', "$MaxSteps", '--add_version', 'false',
    '--output_dir', $runDir
)
if ($SaveSteps -gt 0) {
    $arguments += @('--save_steps', "$SaveSteps")
}
if ($Inspect) {
    [pscustomobject]@{ Executable = $swift; Arguments = $arguments; Train = $train; Validation = $validation; RunDir = $runDir } |
        ConvertTo-Json -Depth 4
    exit 0
}
& $swift @arguments
if ($LASTEXITCODE -ne 0) { throw "ms-swift exited with code $LASTEXITCODE" }
