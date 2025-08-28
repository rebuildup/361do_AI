@echo off
echo ========================================
echo マルチエージェント学習システム
echo 準備・最適化・実行 統合ランチャー
echo ========================================
echo.
echo このスクリプトは以下を順次実行します:
echo 1. システム事前チェック
echo 2. システム最適化
echo 3. 8時間マルチエージェント学習実行
echo.
echo 実行時間: 約8時間
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

REM ルートディレクトリに移動
cd /d "%~dp0\.."

REM 統合ランチャー実行
echo 統合ランチャーを開始しています...
python launch_multi_agent_learning.py

echo.
echo 実行が完了しました
pause