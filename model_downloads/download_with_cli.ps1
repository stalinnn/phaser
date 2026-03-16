# 用 huggingface-cli 下载，若 Python 脚本卡在 0% 可改用本脚本
# 需先：pip install -U huggingface_hub
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "600"
$base = Join-Path $PSScriptRoot "models"
New-Item -ItemType Directory -Force -Path $base | Out-Null

$models = @(
    @{ repo = "tiiuae/Falcon3-Mamba-7B-Instruct"; dir = "Falcon3-Mamba-7B-Instruct" },
    @{ repo = "Qwen/Qwen2-7B"; dir = "Qwen2-7B" },
    @{ repo = "meta-llama/Meta-Llama-3.1-8B"; dir = "Meta-Llama-3.1-8B" }
)

foreach ($m in $models) {
    $localDir = Join-Path $base $m.dir
    Write-Host "[下载] $($m.repo) -> $localDir"
    huggingface-cli download $m.repo --local-dir $localDir
    if ($LASTEXITCODE -ne 0) { Write-Host "[失败] $($m.repo)" }
    else { Write-Host "[完成] $($m.dir)" }
}
Write-Host "全部完成。"
