@echo off
echo ========================================
echo マルチエージェント学習システム テスト
echo ========================================
echo.
echo システムの動作確認を行います
echo.
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

REM テスト実行
echo テストを開始しています...
python test_multi_agent_system.py

echo.
echo テストが完了しました
pause