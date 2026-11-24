param(
    [string]$BaseModel = 'SeerRay-Lab/Xiaomi-OCR-0',
    [string]$OutputDir = 'outputs/qlora-v1',
    [switch]$Inspect
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $env:MODELSCOPE_CACHE) { $env:MODELSCOPE_CACHE = Join-Path $projectRoot 'outputs/modelscope_cache' }
$swift = Join-Path (Split-Path -Parent $projectRoot) '.venv/Scripts/swift.exe'
$train = 'datasets/handwriting_notes/splits/train_swift.jsonl'
$validation = 'datasets/handwriting_notes/splits/val_swift.jsonl'
foreach ($path in @($swift, $train, $validation)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing required file: $path" }
}
$arguments = @(
    'sft', '--model', $BaseModel, '--model_type', 'qwen3_5', '--template', 'qwen3_5',
    '--tuner_type', 'lora', '--quant_method', 'bnb', '--quant_bits', '4',
    '--torch_dtype', 'bfloat16', '--dataset', $train, '--val_dataset', $validation,
    '--max_length', '8192', '--max_pixels', '1024000',
    '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '16',
    '--learning_rate', '1e-4', '--num_train_epochs', '4',
    '--lora_rank', '8', '--lora_alpha', '32', '--lora_dropout', '0.05',
    '--target_modules', 'all-linear', '--gradient_checkpointing', 'true', '--freeze_vit', 'true',
    '--logging_steps', '1', '--save_strategy', 'epoch', '--eval_strategy', 'epoch', '--save_total_limit', '2',
    '--output_dir', $OutputDir, '--warmup_ratio', '0.03', '--lr_scheduler_type', 'cosine',
    '--dataloader_num_workers', '0'
)
if ($Inspect) {
    [pscustomobject]@{ Executable = $swift; Arguments = $arguments; Train = $train; Validation = $validation } |
        ConvertTo-Json -Depth 4
    exit 0
}
& $swift @arguments
if ($LASTEXITCODE -ne 0) { throw "ms-swift exited with code $LASTEXITCODE" }
