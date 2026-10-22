# M1: unified occupancy-depth training (Windows).
# Usage:
#   .\scripts\run_m1.ps1 -ScanPath <path_to_dtu_scan> [-OutDir <dir>]
# Env-style overrides (optional):
#   $env:LAMBDA_OCC='0.02'; $env:LAMBDA_DEPTH_OCC='0.06'; .\scripts\run_m1.ps1 ...

param(
    [Parameter(Mandatory = $true)]
    [string] $ScanPath,
    [string] $OutDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$UggsRoot = Split-Path -Parent $ScriptDir
Set-Location $UggsRoot

$envFile = Join-Path $UggsRoot "configs\dtu_path.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim().Trim('"')
            [Environment]::SetEnvironmentVariable($k, $v, "Process")
        }
    }
}

$scanName = Split-Path -Leaf $ScanPath
if (-not $OutDir) {
    $OutDir = Join-Path $UggsRoot "output_compare\m1_unified\$scanName"
}

$lambdaOcc = if ($env:LAMBDA_OCC) { $env:LAMBDA_OCC } else { "0.02" }
$lambdaDepthOcc = if ($env:LAMBDA_DEPTH_OCC) { $env:LAMBDA_DEPTH_OCC } else { "0.06" }
$regFrom = if ($env:REG_FROM_ITER) { $env:REG_FROM_ITER } else { "7000" }
$occEnt = if ($env:OCC_ENTROPY_WEIGHT) { $env:OCC_ENTROPY_WEIGHT } else { "1.0" }
$occSparse = if ($env:OCC_SPARSITY_WEIGHT) { $env:OCC_SPARSITY_WEIGHT } else { "0.05" }
$depthMinA = if ($env:DEPTH_OCC_MIN_ALPHA) { $env:DEPTH_OCC_MIN_ALPHA } else { "0.05" }

Write-Host "[M1] out=$OutDir lambda_occ=$lambdaOcc lambda_depth_occ=$lambdaDepthOcc reg_from=$regFrom"

py -3 train.py `
    -s $ScanPath `
    -m $OutDir `
    -r 2 `
    --use_decoupled_appearance 3 `
    --lambda_occ $lambdaOcc `
    --lambda_depth_occ $lambdaDepthOcc `
    --regularization_from_iter $regFrom `
    --occ_entropy_weight $occEnt `
    --occ_sparsity_weight $occSparse `
    --depth_occ_min_alpha $depthMinA

py -3 mesh_extract.py -m $OutDir
