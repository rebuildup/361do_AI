@echo off
chcp 65001 > nul
echo AI Agent System を停止中...
docker-compose down
echo.
echo システムが停止されました。
pause
