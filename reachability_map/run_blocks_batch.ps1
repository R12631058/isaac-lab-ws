# run_blocks_batch.ps1
# Runs Block 0 (re-run) + Blocks 3~9 sequentially

$ErrorActionPreference = "Stop"
$ROOT = "C:\Users\RMML\IsaacLab"
$SCRIPT = "scripts\isaaclab_ws\reachability_map\parallel_base_optimizer_linear.py"

$blocks = @(
    @{ id=0; cx=0.0; cy=0.0; cz=0.0 },
    @{ id=3; cx=0.3; cy=0.0; cz=0.0 },
    @{ id=4; cx=0.4; cy=0.0; cz=0.0 },
    @{ id=5; cx=0.5; cy=0.0; cz=0.0 },
    @{ id=6; cx=0.6; cy=0.0; cz=0.0 },
    @{ id=7; cx=0.7; cy=0.0; cz=0.0 },
    @{ id=8; cx=0.8; cy=0.0; cz=0.0 },
    @{ id=9; cx=0.9; cy=0.0; cz=0.0 }
)

$total = $blocks.Count
$i = 1

foreach ($b in $blocks) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  [$i/$total] Block $($b.id)  center=($($b.cx), $($b.cy), $($b.cz))"
    Write-Host "============================================================"

    $start = Get-Date
    $cmd = "conda activate env_isaaclab && isaaclab.bat -p $SCRIPT --block_id $($b.id) --block_center $($b.cx) $($b.cy) $($b.cz) --max_waypoints 70"

    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Block $($b.id) failed with exit code $LASTEXITCODE. Stopping."
        exit 1
    }

    $elapsed = (Get-Date) - $start
    Write-Host "[OK] Block $($b.id) done in $([math]::Round($elapsed.TotalMinutes, 1)) min"
    $i++
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  All $total blocks completed!"
Write-Host "  Next: run build_cmap_database.py to rebuild the database."
Write-Host "============================================================"
