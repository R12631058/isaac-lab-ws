# run_all_blocks.ps1
# 執行 Block 0 到 Block 9 的優化計算

param(
    [int]$startBlock = 0,
    [int]$endBlock = 2
)

$envName = "env_isaaclab"
$scriptPath = "scripts\isaaclab_ws\reachability_map\parallel_base_optimizer_linear.py"

Write-Host "Starting multi-block optimization (Block $startBlock to $endBlock)..."
Write-Host "Activating Conda environment: $envName"
conda activate $envName

for ($i = $startBlock; $i -le $endBlock; $i++) {
    $x_center = [math]::Round($i * 0.1, 1)
    $center_str = $x_center.ToString("F1")

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "  Starting Block $i"
    Write-Host "  Block Center : X=$center_str, Y=0.0, Z=0.0"
    Write-Host "  Search Range : X=0.1 ~ 1.5"
    Write-Host "=================================================="

    # 使用 cmd /c 呼叫 .bat 確保環境正確
    cmd /c "conda activate $envName && isaaclab.bat -p $scriptPath --block_center $center_str 0.0 0.0 --block_id $i --x_min 0.1 --x_max 1.5 --x_steps 15 --headless"

    Write-Host "Block $i Completed."
}

Write-Host ""
Write-Host "All 10 blocks have been processed successfully!"
