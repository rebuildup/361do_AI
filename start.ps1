# 自己学習AIエージェント 361do_AI 起動スクリプト (PowerShell)

Write-Host "🚀 自己学習AIエージェント 361do_AI を起動します..." -ForegroundColor Green

# 環境チェック
Write-Host "📋 環境チェック中..." -ForegroundColor Yellow

# Python環境チェック
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Pythonがインストールされていません" -ForegroundColor Red
    exit 1
}

# Node.js環境チェック
try {
    $nodeVersion = node --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Node.js not found"
    }
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.jsがインストールされていません" -ForegroundColor Red
    exit 1
}

# Yarn環境チェック
try {
    $yarnVersion = yarn --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Yarn not found"
    }
    Write-Host "✅ Yarn: $yarnVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Yarnがインストールされていません" -ForegroundColor Red
    exit 1
}

Write-Host "✅ 環境チェック完了" -ForegroundColor Green

# 依存関係インストール
Write-Host "📦 依存関係をインストール中..." -ForegroundColor Yellow

# Python依存関係
if (-not (Test-Path ".venv")) {
    Write-Host "🐍 Python仮想環境を作成中..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "🐍 Python依存関係をインストール中..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"
pip install -r requirements.txt

# フロントエンド依存関係
Write-Host "⚛️ フロントエンド依存関係をインストール中..." -ForegroundColor Yellow
Set-Location frontend
yarn install
Set-Location ..

Write-Host "✅ 依存関係インストール完了" -ForegroundColor Green

# フロントエンドビルド
Write-Host "🔨 フロントエンドをビルド中..." -ForegroundColor Yellow
Set-Location frontend
yarn build
Set-Location ..

Write-Host "✅ フロントエンドビルド完了" -ForegroundColor Green

# アプリケーション起動
Write-Host "🌐 アプリケーションを起動中..." -ForegroundColor Green
Write-Host "URL: http://localhost:80" -ForegroundColor Cyan
Write-Host "React UI + FastAPI バックエンドで起動します" -ForegroundColor Cyan

python main.py --ui react --host 0.0.0.0 --port 80
