@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Could not find .venv\Scripts\activate.bat
    echo Create the virtual environment before using this shortcut.
    pause
    exit /b 1
)

title TKD Card Inventory - Python Environment
call ".venv\Scripts\activate.bat"

echo.
echo Virtual environment activated.
echo Project folder: %CD%
echo Type exit when you are finished.
echo.

cmd /k
