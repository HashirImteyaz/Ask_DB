@echo off
echo Testing Redis Connection...
cd /d "%~dp0redis"
redis-cli.exe ping
if %ERRORLEVEL% EQU 0 (
    echo Redis is running successfully!
) else (
    echo Redis is not running. Please start Redis first using start_redis.bat
)
pause
