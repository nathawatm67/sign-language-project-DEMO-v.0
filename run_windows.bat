@echo off
chcp 65001 >nul
echo.
echo =====================================================
echo   Smart Thai Sign Language - Windows Installer
echo =====================================================
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] ไม่พบ Python!
    echo     https://python.org/downloads
    echo     ** ติ๊ก "Add Python to PATH" **
    pause & exit /b 1
)
echo [OK] Python ready
echo.
echo [1/2] Installing libraries...
pip install "opencv-python" "mediapipe==0.10.13" "numpy" --quiet
if errorlevel 1 (echo [X] Install failed & pause & exit /b 1)
echo [OK] Libraries ready
echo.
echo =====================================================
echo   HELLO      = Open palm, 5 fingers
echo   THANK YOU  = Thumbs-up, raise to chin
echo   SORRY      = Tight fist, thumb inside
echo   Q=quit  H=landmark  S=screenshot  D=debug
echo =====================================================
echo.
echo [2/2] Starting program...
python thai_sign_3words.py
echo.
pause
