# Evaluate Base / Candidate / CGSR on one dataset with the paper-facing protocol.
# Usage: powershell -File scripts\eval.ps1 <dataset> <data_root> [extra tools\eval_cgsr.py args]
#   dataset: voc | context59 | ade20k | coco_stuff27 | cityscapes
# Expects base predictor weights in weights\<ckpt>.pth, text embeddings in
# weights\text\<embedding>.pth (docs\WEIGHTS.md) and the base-code checkout in
# third_party\base_predictor (scripts\setup_upstream.ps1); override with $env:BASE_CODE_ROOT.
param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)
$ErrorActionPreference = "Stop"
$BaseCodeRoot = if ($env:BASE_CODE_ROOT) { $env:BASE_CODE_ROOT } else { "third_party\base_predictor" }

$Assets = @{
    "voc"          = @("voc.pth", "voc_ViT16_clip_text.pth")
    "context59"    = @("context.pth", "context_ViT16_clip_text_alpha.pth")
    "ade20k"       = @("ade.pth", "ade_ViT16_clip_text.pth")
    "coco_stuff27" = @("coco.pth", "stuff27_ViT16_clip_text.pth")
    "cityscapes"   = @("city.pth", "city_ViT16_clip_text.pth")
}
if (-not $Assets.ContainsKey($Dataset)) { throw "unknown dataset: $Dataset" }
$Ckpt, $Text = $Assets[$Dataset]

python tools\eval_cgsr.py `
    --dataset $Dataset `
    --base-code-root $BaseCodeRoot `
    --data-root $DataRoot `
    --checkpoint "weights\$Ckpt" `
    --text-embedding "weights\text\$Text" `
    --output-dir "outputs\$Dataset" `
    @Extra
exit $LASTEXITCODE
