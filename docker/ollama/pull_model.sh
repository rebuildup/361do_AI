#!/bin/bash

# OLLAMA サービス開始
ollama serve &
OLLAMA_PID=$!

# サービスが開始されるまで待機
sleep 10

# qwen2:7b-instruct モデルをプル
echo "Downloading qwen2:7b-instruct model..."
ollama pull qwen2:7b-instruct

# モデルダウンロード完了確認
if ollama list | grep -q "qwen2:7b-instruct"; then
    echo "qwen2:7b-instruct model successfully downloaded!"
else
    echo "Failed to download qwen2:7b-instruct model"
    exit 1
fi

# バックグラウンドプロセスを前面に移動
wait $OLLAMA_PID
