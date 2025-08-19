@echo off
chcp 65001 > nul
echo AI Agent System を起動中...
docker-compose up -d
echo.
echo システムが起動しました！
echo.
echo アクセス方法:
echo - Open WebUI: http://localhost:3000
echo - Agent API: http://localhost:8000
echo - Nginx統合: http://localhost
echo.
echo システム状態を確認中...
docker-compose ps
echo.
echo 起動完了！ブラウザで http://localhost:3000 にアクセスしてください。
pause
