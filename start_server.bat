@echo off
echo Starting Flask server in the background...
start /B python app.py
timeout /t 3 /nobreak >nul
echo.
echo ========================================
echo Server should be running!
echo ========================================
echo.
echo Access your website at:
echo   http://localhost:5000
echo   http://127.0.0.1:5000
echo.
echo To stop the server, close this window or find and kill the python process.
echo.
pause

