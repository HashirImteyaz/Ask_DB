@echo off
echo Starting Redis Server...
cd /d "%~dp0redis"
redis-server.exe --port 6379
pause
