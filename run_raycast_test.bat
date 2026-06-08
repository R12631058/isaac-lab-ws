@echo off
REM 在 env_isaaclab 环境中运行 raycast 测试
REM 这个脚本确保使用正确的环境来运行 test_fixed_and_moving_sphere.py

echo Activating env_isaaclab environment...
call conda activate env_isaaclab

if errorlevel 1 (
    echo ❌ Failed to activate env_isaaclab
    exit /b 1
)

echo ✅ Environment activated
echo.
echo Running raycast test...
echo ======================================================================

python scripts/isaaclab_ws/experiments/test_fixed_and_moving_sphere.py

if errorlevel 1 (
    echo.
    echo ❌ Test failed
    exit /b 1
)

echo.
echo ✅ Test completed successfully
pause
