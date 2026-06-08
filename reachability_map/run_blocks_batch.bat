@echo off
REM run_blocks_batch.bat
REM 依序執行 Block 0 (重跑) + Block 3~9
REM 使用方式：在 conda env_isaaclab 環境下執行

SET ROOT=C:\Users\RMML\IsaacLab
SET SCRIPT=scripts\isaaclab_ws\reachability_map\parallel_base_optimizer_linear.py
SET ISAACLAB=%ROOT%\isaaclab.bat

echo ============================================================
echo  Block Batch Runner: Block 0, 3, 4, 5, 6, 7, 8, 9
echo ============================================================

REM ──── Block 0 (re-run with fixed pruning) ────
echo.
echo [1/8] Running Block 0  center=(0.0, 0.0, 0.0) ...
call %ISAACLAB% -p %SCRIPT% --block_id 0 --block_center 0.0 0.0 0.0 --max_waypoints 70
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Block 0 failed. Stopping. && exit /b 1)
echo [OK] Block 0 done.

REM ──── Block 3 ────
echo.
echo [2/8] Running Block 3  center=(0.3, 0.0, 0.0) ...
call %ISAACLAB% -p %SCRIPT% --block_id 3 --block_center 0.3 0.0 0.0 --max_waypoints 70
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Block 3 failed. Stopping. && exit /b 1)
echo [OK] Block 3 done.

REM ──── Block 4 ────
echo.
echo [3/8] Running Block 4  center=(0.4, 0.0, 0.0) ...
call %ISAACLAB% -p %SCRIPT% --block_id 4 --block_center 0.4 0.0 0.0 --max_waypoints 70
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Block 4 failed. Stopping. && exit /b 1)
echo [OK] Block 4 done.

REM ──── Block 5 ────
echo.
echo [4/8] Running Block 5  center=(0.5, 0.0, 0.0) ...
call %ISAACLAB% -p %SCRIPT% --block_id 5 --block_center 0.5 0.0 0.0 --max_waypoints 70
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Block 5 failed. Stopping. && exit /b 1)
echo [OK] Block 5 done.

REM ──── Block 6 ────
echo.
echo [5/8] Running Block 6  center=(0.6, 0.0, 0.0) ...
call %ISAACLAB% -p %SCRIPT% --block_id 6 --block_center 0.6 0.0 0.0 --max_waypoints 70
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Block 6 failed. Stopping. && exit /b 1)
echo [OK] Block 6 done.

REM ──── Block 7 ────
echo.
echo [6/8] Running Block 7  center=(0.7, 0.0, 0.0) ...
call %ISAACLAB% -p %SCRIPT% --block_id 7 --block_center 0.7 0.0 0.0 --max_waypoints 70
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Block 7 failed. Stopping. && exit /b 1)
echo [OK] Block 7 done.

REM ──── Block 8 ────
echo.
echo [7/8] Running Block 8  center=(0.8, 0.0, 0.0) ...
call %ISAACLAB% -p %SCRIPT% --block_id 8 --block_center 0.8 0.0 0.0 --max_waypoints 70
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Block 8 failed. Stopping. && exit /b 1)
echo [OK] Block 8 done.

REM ──── Block 9 ────
echo.
echo [8/8] Running Block 9  center=(0.9, 0.0, 0.0) ...
call %ISAACLAB% -p %SCRIPT% --block_id 9 --block_center 0.9 0.0 0.0 --max_waypoints 70
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Block 9 failed. Stopping. && exit /b 1)
echo [OK] Block 9 done.

echo.
echo ============================================================
echo  All 8 blocks completed! Now run build_cmap_database.py
echo ============================================================
