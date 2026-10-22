# M1 ablation (Windows): same as run_m1_ablation.sh
# Usage: .\scripts\run_m1_ablation.ps1 scan24 scan105 scan122
# Requires: baseline at $BASELINE\<scan>; conda env with torch; py -3 or $env:PYTHON

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Scans = @("scan24", "scan105", "scan122")
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$UggsRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $UggsRoot

$envFile = Join-Path $UggsRoot "configs\dtu_path.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*export\s+UGGS_DTU_ROOT=(.+)$') {
            $v = $matches[1].Trim().Trim('"').Trim("'")
            $v = $v -replace '^/mnt/c/', 'C:\' -replace '/', '\'
            $env:UGGS_DTU_ROOT = $v
        }
        elseif ($_ -match '^\s*UGGS_DTU_ROOT=(.+)$') {
            $env:UGGS_DTU_ROOT = $matches[1].Trim().Trim('"')
        }
    }
}
if (-not $env:UGGS_DTU_ROOT) {
    Write-Error "Set UGGS_DTU_ROOT or add configs/dtu_path.env"
}

$PY = if ($env:PYTHON) { $env:PYTHON } else { "py" }
$PY_ARGS = @()
if ($PY -eq "py") { $PY_ARGS = @("-3") }

$BASE = if ($env:BASELINE_ROOT) { $env:BASELINE_ROOT } else { ".\output_compare\ggs_baseline" }
$EVAL = if ($env:DTU_EVAL_ROOT) { $env:DTU_EVAL_ROOT } else { ".\dtu_eval\Offical_DTU_Dataset" }
$MET = if ($env:METRIC_ROOT) { $env:METRIC_ROOT } else { ".\output_compare\metrics" }
$REG = if ($env:REG_FROM_ITER) { $env:REG_FROM_ITER } else { "7000" }
$SKIP = $env:ABLATION_SKIP_TRAIN -eq "1"

$LO1 = if ($env:LAMBDA_OCC_A1) { $env:LAMBDA_OCC_A1 } else { "0.02" }
$LO3 = if ($env:LAMBDA_OCC_A3) { $env:LAMBDA_OCC_A3 } else { "0.02" }
$LO4 = if ($env:LAMBDA_OCC_A4) { $env:LAMBDA_OCC_A4 } else { "0.02" }
$LDO2 = if ($env:LAMBDA_DEPTH_OCC_A2) { $env:LAMBDA_DEPTH_OCC_A2 } else { "0.06" }
$LDO3 = if ($env:LAMBDA_DEPTH_OCC_A3) { $env:LAMBDA_DEPTH_OCC_A3 } else { "0.06" }
$LDR4 = if ($env:LAMBDA_DEPTH_OCC_RANK_A4) { $env:LAMBDA_DEPTH_OCC_RANK_A4 } else { "0.04" }

New-Item -ItemType Directory -Force -Path $MET | Out-Null

function Invoke-Train {
    param([string]$Sub, [string]$Lo, [string]$Ldo, [string]$Ldr, [string]$Scan)
    $out = Join-Path $UggsRoot "output_compare\$Sub\$Scan"
    Write-Host "----- train $Scan -> $out -----"
    & $PY @PY_ARGS train.py -s (Join-Path $env:UGGS_DTU_ROOT $Scan) -m $out -r 2 --use_decoupled_appearance 3 `
        --lambda_occ $Lo --lambda_depth_occ $Ldo --lambda_depth_occ_rank $Ldr `
        --regularization_from_iter $REG `
        --occ_entropy_weight 1.0 --occ_sparsity_weight 0.05 --depth_occ_min_alpha 0.05 `
        --occ_rank_margin 0.02 --occ_rank_alpha_soft 0.08
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $PY @PY_ARGS mesh_extract.py -m $out
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Compare {
    param([string]$Sub, [string]$Scan)
    $opt = Join-Path $UggsRoot "output_compare\$Sub\$Scan"
    $json = Join-Path $MET "${Scan}_baseline_vs_${Sub}.json"
    Write-Host "----- compare $Scan -> $json -----"
    & $PY @PY_ARGS scripts\compare_models.py `
        --baseline (Join-Path $BASE $Scan) `
        --optimized $opt `
        --dtu_eval_root $EVAL `
        --out $json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

foreach ($s in $Scans) {
    if (-not (Test-Path (Join-Path $BASE $s))) {
        Write-Error "Missing baseline: $(Join-Path $BASE $s)"
    }
}

if (-not $SKIP) {
    foreach ($s in $Scans) { Invoke-Train "m1_ablate_01_occ_only" $LO1 "0" "0" $s }
    foreach ($s in $Scans) { Invoke-Train "m1_ablate_02_depth_occ_only" "0" $LDO2 "0" $s }
    foreach ($s in $Scans) { Invoke-Train "m1_ablate_03_full" $LO3 $LDO3 "0" $s }
    foreach ($s in $Scans) { Invoke-Train "m1_ablate_04_full_rank" $LO4 $LDO3 $LDR4 $s }
}

foreach ($s in $Scans) {
    Invoke-Compare "m1_ablate_01_occ_only" $s
    Invoke-Compare "m1_ablate_02_depth_occ_only" $s
    Invoke-Compare "m1_ablate_03_full" $s
    Invoke-Compare "m1_ablate_04_full_rank" $s
}

Write-Host "=== Summary ==="
& $PY @PY_ARGS (Join-Path $ScriptDir "summarize_m1_ablation.py") --metrics_dir $MET
Write-Host "Done."
