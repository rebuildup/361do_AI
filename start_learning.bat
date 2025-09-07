@echo off
echo ========================================
echo 361do_AI 4時間継続学習システム
echo ========================================
echo.

REM 必要なディレクトリを作成
if not exist "logs" mkdir logs
if not exist "data" mkdir data

echo 学習システムを開始します...
echo 注意: このプロセスは4時間継続して実行されます
echo Ctrl+C で早期終了可能です
echo.

python start_continuous_learning.py

echo.
echo 学習システムが終了しました
pause
