#!/bin/bash
# 自己学習AIエージェント 361do_AI 起動スクリプト

set -e

echo "🚀 自己学習AIエージェント 361do_AI を起動します..."

# 環境チェック
echo "📋 環境チェック中..."

# Python環境チェック
if ! command -v python &> /dev/null; then
    echo "❌ Pythonがインストールされていません"
    exit 1
fi

# Node.js環境チェック
if ! command -v node &> /dev/null; then
    echo "❌ Node.jsがインストールされていません"
    exit 1
fi

# Yarn環境チェック
if ! command -v yarn &> /dev/null; then
    echo "❌ Yarnがインストールされていません"
    exit 1
fi

echo "✅ 環境チェック完了"

# 依存関係インストール
echo "📦 依存関係をインストール中..."

# Python依存関係
if [ ! -d ".venv" ]; then
    echo "🐍 Python仮想環境を作成中..."
    python -m venv .venv
fi

echo "🐍 Python依存関係をインストール中..."
source .venv/bin/activate 2>/dev/null || .venv\Scripts\activate 2>/dev/null
pip install -r requirements.txt

# フロントエンド依存関係
echo "⚛️ フロントエンド依存関係をインストール中..."
cd frontend
yarn install
cd ..

echo "✅ 依存関係インストール完了"

# フロントエンドビルド
echo "🔨 フロントエンドをビルド中..."
cd frontend
yarn build
cd ..

echo "✅ フロントエンドビルド完了"

# アプリケーション起動
echo "🌐 アプリケーションを起動中..."
echo "URL: http://localhost:80"
echo "React UI + FastAPI バックエンドで起動します"

python main.py --ui react --host 0.0.0.0 --port 80
