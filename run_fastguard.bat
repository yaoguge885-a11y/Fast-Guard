@echo off
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    echo Using virtual environment...
    "venv\Scripts\python.exe" main.py
) else (
    echo Virtual environment not found. Using system python...
    where python >nul 2>nul
    if %errorlevel%==0 (
        python main.py
    ) else (
        py -3 main.py
    )
)

pause
