@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Custom Isaac Lab launcher that uses conda env_isaaclab environment
rem This ensures the correct Python version is used

echo [INFO] Using custom Isaac Lab launcher with conda env_isaaclab

rem Set the conda environment path explicitly
set "CONDA_PREFIX=C:\Users\RMML\anaconda3\envs\env_isaaclab"
set "python_exe=%CONDA_PREFIX%\python.exe"

rem Verify the Python executable exists
if not exist "%python_exe%" (
    echo [ERROR] Python executable not found at: %python_exe%
    echo [ERROR] Please ensure env_isaaclab conda environment is installed
    pause
    exit /b 1
)

echo [INFO] Using python from: %python_exe%

rem Set the Isaac Lab path
set "ISAACLAB_PATH=%~dp0"

rem Call the original isaaclab.bat with the conda environment set
call "%ISAACLAB_PATH%\isaaclab.bat" %*
