param([switch]$Execute)
$ErrorActionPreference = 'Stop'
if (-not $Execute) { throw 'Review cleanup_plan.json first, then supply -Execute. This removes intermediate optimizer checkpoints permanently.' }
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath.TrimEnd('\')
$planPath = Join-Path $PSScriptRoot 'cleanup_plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
$journal = Join-Path $PSScriptRoot 'cleanup_deletions.jsonl'
$summaryPath = Join-Path $PSScriptRoot 'cleanup_result.json'
if ((Test-Path -LiteralPath $journal) -or (Test-Path -LiteralPath $summaryPath)) { throw 'Cleanup already has a receipt; do not overwrite it.' }
if ($plan.workspace.TrimEnd('\') -ne $taskRoot -or $plan.candidate_count -ne 60 -or $plan.candidates.Count -ne 60) { throw 'Unexpected workspace or plan count.' }
$paths = @{}
foreach ($row in $plan.candidates) {
    if ($row.path -notmatch '^unified_geometry_gs/output_compare/ggs_tau_(0\.3|0\.5|1\.2)/scan105/chkpnt(1[0-9]|2[0-9])000\.pth$') { throw "Out of scope: $($row.path)" }
    $full = (Resolve-Path -LiteralPath (Join-Path $taskRoot $row.path)).ProviderPath
    if (-not $full.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or $paths.ContainsKey($full)) { throw 'Escaped or duplicate target.' }
    $probe = Get-Item -LiteralPath $full
    if ($probe.PSIsContainer -or $probe.Length -ne $row.bytes) { throw "Target changed: $full" }
    while ($probe.FullName -ne $taskRoot) {
        if ($probe.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse target/ancestor: $($probe.FullName)" }
        $probe = Get-Item -LiteralPath ([IO.Path]::GetDirectoryName($probe.FullName))
    }
    $paths[$full] = $row
}
foreach ($row in $plan.preserved_essential) {
    $full = Join-Path $taskRoot $row.path
    if ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw "Protected file changed: $full" }
}
$deleted = 0; $removedBytes = [long]0; $failure = $null; $protectedVerified = $false
try {
    foreach ($full in ($paths.Keys | Sort-Object)) {
        $row = $paths[$full]
        if ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw "Candidate changed: $full" }
        Remove-Item -LiteralPath $full -ErrorAction Stop
        if (Test-Path -LiteralPath $full) { throw "Target still exists: $full" }
        $deleted++; $removedBytes += [long]$row.bytes
        @{path=$row.path;bytes=$row.bytes;sha256=$row.sha256;deleted_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress | Add-Content -LiteralPath $journal -Encoding utf8
    }
    foreach ($row in $plan.preserved_essential) {
        if ((Get-FileHash -LiteralPath (Join-Path $taskRoot $row.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw "Protected file changed after cleanup: $($row.path)" }
    }
    $protectedVerified = $true
} catch { $failure = $_.Exception.Message }
@{completed_utc=[DateTime]::UtcNow.ToString('o');plan_sha256=(Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant();deleted_files=$deleted;removed_logical_bytes=$removedBytes;removed_GiB=$removedBytes/1GB;protected_files=$plan.preserved_essential.Count;protected_hashes_verified_before_and_after=$protectedVerified;failure=$failure;note='Logical file sizes, not a measurement of physical disk free-space delta. No backup of intermediate optimizer states was made.'} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath -Encoding utf8
Get-Content -LiteralPath $summaryPath
if ($failure) { throw $failure }
