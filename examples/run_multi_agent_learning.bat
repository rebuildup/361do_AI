@echo off
echo ========================================
echo マルチエージェント学習システム
echo ========================================
echo.
echo 4つのエージェントが8時間にわたって
echo 相互学習を行います
echo.
echo 停止方法: Ctrl+C
echo ========================================
echo.

REM Python環境の確認
python --version >nul 2>&1
if errorlevel 1 (
    echo エラー: Pythonが見つかりません
    echo Pythonをインストールしてください
    pause
    exit /b 1
)

REM 仮想環境の確認とアクティベート
if exist .venv\Scripts\activate.bat (
    echo 仮想環境をアクティベート中...
    call .venv\Scripts\activate.bat
)

REM システム実行
echo システムを開始しています...
python multi_agent_learning_system.py

echo.
echo システムが終了しました
pause