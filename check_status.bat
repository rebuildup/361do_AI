@echo off
chcp 65001 > nul
echo AI Agent System の状態確認...
echo.
docker-compose ps
echo.
echo APIヘルスチェック:
curl -s http://localhost:8000/health
echo.
echo WebUIアクセス確認:
curl -s -I http://localhost:3000 | findstr "HTTP"
echo.
pause
