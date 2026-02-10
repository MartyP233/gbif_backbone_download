@echo off
echo ====================================
echo GBIF Taxonomy Downloader Setup
echo ====================================
echo.

REM Check if uv is installed
echo [1/3] Checking for uv...
uv --version >nul 2>&1
if errorlevel 1 (
    echo uv is not installed. Installing uv...
    echo.
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: Failed to install uv
        echo Please install manually from: https://docs.astral.sh/uv/getting-started/installation/
        echo.
        pause
        exit /b 1
    )
    echo.
    echo uv installed successfully!
    echo Please restart this script or open a new command prompt.
    echo.
    pause
    exit /b 0
) else (
    uv --version
)
echo.

REM Install dependencies
echo [2/3] Installing dependencies...
uv sync
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

REM Run the script
echo [3/3] Running GBIF taxonomy downloader...
echo.
echo ====================================
uv run python main.py
echo ====================================
echo.

if errorlevel 1 (
    echo ERROR: Script encountered an error
    pause
    exit /b 1
)

echo.
echo SUCCESS! The GBIF taxonomy has been downloaded and converted.
echo Output file: data\raw_gbif__backbone.parquet
echo.
pause
