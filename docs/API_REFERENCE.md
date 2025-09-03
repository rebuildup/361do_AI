# API リファレンス

## 概要

Advanced Self-Learning AI Agent は、RESTful API、Python クライアント、WebSocket 接続を通じて利用できます。このドキュメントでは、すべての利用可能な API エンドポイントとその使用方法について詳しく説明します。

## 🌐 REST API

### ベース URL

```
http://localhost:8000
```

### 認証

現在、開発環境では認証は不要です。本番環境では API キーまたは JWT トークンが必要になります。

```bash
# 本番環境での認証例
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     http://localhost:8000/api/v1/chat
```

## 📡 チャット API

### POST /chat

基本的なチャット機能を提供します。

#### リクエスト

```json
{
  "message": "こんにちは、機械学習について教えてください",
  "use_memory": true,
  "use_cot": true,
  "model": "deepseek-r1:7b",
  "session_id": "optional_session_id",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

#### レスポンス

```json
{
  "response": "機械学習は、コンピュータが明示的にプログラムされることなく...",
  "reasoning_steps": [
    "ステップ1: 機械学習の定義を考える",
    "ステップ2: 主要なアルゴリズムを整理する",
    "ステップ3: 実用例を挙げる"
  ],
  "session_id": "session_123",
  "model_used": "deepseek-r1:7b",
  "response_time": 1.23,
  "memory_used": true,
  "metadata": {
    "vram_usage": "4.2GB",
    "gpu_utilization": "85%",
    "inference_time": 1.15
  }
}
```

#### cURL 例

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Pythonでソートアルゴリズムを実装してください",
       "use_memory": true,
       "use_cot": true
     }'
```

#### Python 例

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "機械学習の基本概念を説明してください",
        "use_memory": True,
        "use_cot": True,
        "temperature": 0.7
    }
)

result = response.json()
print(f"応答: {result['response']}")
print(f"推論ステップ: {result['reasoning_steps']}")
```

### POST /chat/stream

ストリーミングチャット機能を提供します。

#### リクエスト

```json
{
  "message": "長い説明が必要な複雑な質問",
  "use_memory": true,
  "use_cot": true,
  "stream": true
}
```

#### レスポンス（Server-Sent Events）

```
data: {"type": "start", "session_id": "session_123"}

data: {"type": "reasoning_step", "step": 1, "content": "まず基本概念を整理します"}

data: {"type": "content", "content": "機械学習は"}

data: {"type": "content", "content": "データから"}

data: {"type": "end", "metadata": {"response_time": 2.34}}
```

#### Python 例（ストリーミング）

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/chat/stream",
    json={
        "message": "詳細な説明をお願いします",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        if data['type'] == 'content':
            print(data['content'], end='', flush=True)
```

## 🧠 推論 API

### POST /reasoning/cot

Chain-of-Thought 推論を実行します。

#### リクエスト

```json
{
  "query": "2x + 5 = 15 を解いてください",
  "steps": 5,
  "model": "deepseek-r1:7b"
}
```

#### レスポンス

```json
{
  "query": "2x + 5 = 15 を解いてください",
  "reasoning_steps": [
    {
      "step": 1,
      "thought": "方程式 2x + 5 = 15 を解く必要がある",
      "action": "両辺から5を引く"
    },
    {
      "step": 2,
      "thought": "2x = 15 - 5 = 10",
      "action": "両辺を2で割る"
    },
    {
      "step": 3,
      "thought": "x = 10 / 2 = 5",
      "action": "解を確認する"
    }
  ],
  "final_answer": "x = 5",
  "confidence": 0.95
}
```

### POST /reasoning/batch

複数の推論を並列実行します。

#### リクエスト

```json
{
  "queries": ["質問1: ...", "質問2: ...", "質問3: ..."],
  "use_cot": true,
  "parallel": true
}
```

#### レスポンス

```json
{
  "results": [
    {
      "query": "質問1: ...",
      "response": "回答1...",
      "reasoning_steps": ["..."]
    },
    {
      "query": "質問2: ...",
      "response": "回答2...",
      "reasoning_steps": ["..."]
    }
  ],
  "total_time": 3.45,
  "parallel_efficiency": 0.87
}
```

## 💾 記憶 API

### GET /memory/sessions

セッション一覧を取得します。

#### レスポンス

```json
{
  "sessions": [
    {
      "session_id": "session_123",
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T11:45:00Z",
      "message_count": 15,
      "summary": "機械学習についての議論"
    }
  ],
  "total_sessions": 1
}
```

### GET /memory/sessions/{session_id}

特定のセッションの履歴を取得します。

#### レスポンス

```json
{
  "session_id": "session_123",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "こんにちは",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "こんにちは！何かお手伝いできることはありますか？",
      "timestamp": "2024-01-15T10:30:05Z"
    }
  ],
  "metadata": {
    "total_messages": 2,
    "session_duration": "01:15:00"
  }
}
```

### POST /memory/search

記憶を検索します。

#### リクエスト

```json
{
  "query": "機械学習",
  "limit": 10,
  "importance_threshold": 0.5,
  "memory_types": ["conversation", "knowledge", "preference"]
}
```

#### レスポンス

```json
{
  "memories": [
    {
      "id": "mem_123",
      "content": "ユーザーは機械学習に興味がある",
      "importance": 0.8,
      "memory_type": "preference",
      "created_at": "2024-01-15T10:30:00Z",
      "relevance_score": 0.92
    }
  ],
  "total_found": 1
}
```

### POST /memory/store

新しい記憶を保存します。

#### リクエスト

```json
{
  "content": "ユーザーはPythonプログラミングが得意",
  "importance": 0.7,
  "memory_type": "user_skill",
  "tags": ["programming", "python", "skill"]
}
```

#### レスポンス

```json
{
  "memory_id": "mem_456",
  "status": "stored",
  "importance": 0.7
}
```

## 🧬 学習 API

### GET /learning/status

学習システムの状態を取得します。

#### レスポンス

```json
{
  "evolutionary_learning": {
    "enabled": true,
    "current_generation": 5,
    "population_size": 10,
    "best_fitness": 0.87,
    "training_progress": 0.65
  },
  "adapters": {
    "total_adapters": 15,
    "active_adapters": 3,
    "best_adapter": {
      "id": "adapter_123",
      "fitness": 0.87,
      "config": {
        "r": 16,
        "lora_alpha": 32
      }
    }
  }
}
```

### POST /learning/evolve

進化的学習を開始します。

#### リクエスト

```json
{
  "training_data": [
    { "input": "質問1", "output": "回答1" },
    { "input": "質問2", "output": "回答2" }
  ],
  "generations": 5,
  "population_size": 10,
  "mutation_rate": 0.1
}
```

#### レスポンス

```json
{
  "evolution_id": "evo_123",
  "status": "started",
  "estimated_duration": "00:30:00"
}
```

### GET /learning/evolution/{evolution_id}

進化的学習の進捗を取得します。

#### レスポンス

```json
{
  "evolution_id": "evo_123",
  "status": "running",
  "current_generation": 3,
  "total_generations": 5,
  "progress": 0.6,
  "best_fitness": 0.82,
  "estimated_remaining": "00:12:00"
}
```

## 📊 監視 API

### GET /monitoring/status

システム状態を取得します。

#### レスポンス

```json
{
  "system": {
    "status": "healthy",
    "uptime": "02:15:30",
    "version": "1.0.0"
  },
  "gpu": {
    "utilization": 75,
    "memory_used": 4.2,
    "memory_total": 6.0,
    "temperature": 68,
    "power_usage": 150
  },
  "cpu": {
    "utilization": 45,
    "cores": 16,
    "frequency": 3.2
  },
  "memory": {
    "used": 18.5,
    "total": 32.0,
    "swap_used": 2.1
  },
  "performance": {
    "avg_response_time": 1.23,
    "requests_per_minute": 45,
    "error_rate": 0.02
  }
}
```

### GET /monitoring/metrics

詳細なメトリクスを取得します。

#### レスポンス

```json
{
  "timestamp": "2024-01-15T12:00:00Z",
  "metrics": {
    "inference_latency_ms": [1200, 1150, 1300, 1100],
    "gpu_utilization_percent": [75, 78, 72, 80],
    "vram_usage_gb": [4.2, 4.3, 4.1, 4.4],
    "cpu_utilization_percent": [45, 48, 42, 50],
    "memory_usage_gb": [18.5, 18.7, 18.3, 18.9]
  },
  "aggregated": {
    "avg_inference_latency": 1187.5,
    "max_gpu_utilization": 80,
    "avg_vram_usage": 4.25
  }
}
```

### GET /monitoring/alerts

アクティブなアラートを取得します。

#### レスポンス

```json
{
  "alerts": [
    {
      "id": "alert_123",
      "type": "warning",
      "message": "GPU温度が閾値を超えています",
      "metric": "gpu_temperature",
      "value": 82,
      "threshold": 80,
      "timestamp": "2024-01-15T12:00:00Z",
      "resolved": false
    }
  ],
  "total_alerts": 1,
  "critical_count": 0,
  "warning_count": 1
}
```

## 🔧 設定 API

### GET /config

現在の設定を取得します。

#### レスポンス

```json
{
  "gpu": {
    "max_vram_gb": 5.0,
    "quantization_levels": [8, 4, 3],
    "temperature_threshold": 80
  },
  "models": {
    "primary": "deepseek-r1:7b",
    "fallback": "qwen2.5:7b-instruct-q4_k_m"
  },
  "learning": {
    "adapter_pool_size": 10,
    "mutation_rate": 0.1
  }
}
```

### PUT /config

設定を更新します。

#### リクエスト

```json
{
  "gpu": {
    "max_vram_gb": 4.5,
    "temperature_threshold": 75
  }
}
```

#### レスポンス

```json
{
  "status": "updated",
  "changes": [
    "gpu.max_vram_gb: 5.0 -> 4.5",
    "gpu.temperature_threshold: 80 -> 75"
  ],
  "restart_required": false
}
```

## 🐍 Python クライアント

### インストール

```python
# プロジェクト内のクライアントを使用
from src.advanced_agent.client import AdvancedAgentClient
```

### 基本的な使用方法

```python
import asyncio
from src.advanced_agent.client import AdvancedAgentClient

async def main():
    # クライアントの初期化
    client = AdvancedAgentClient("http://localhost:8000")

    # 基本的なチャット
    response = await client.chat(
        message="機械学習について教えてください",
        use_memory=True,
        use_cot=True
    )

    print(f"応答: {response.content}")
    print(f"推論ステップ: {response.reasoning_steps}")

    # ストリーミングチャット
    async for chunk in client.chat_stream(
        message="詳細な説明をお願いします"
    ):
        print(chunk.content, end='', flush=True)

    # 記憶の検索
    memories = await client.search_memories(
        query="プログラミング",
        limit=5
    )

    for memory in memories:
        print(f"記憶: {memory.content}")

    # システム状態の確認
    status = await client.get_system_status()
    print(f"GPU使用率: {status.gpu.utilization}%")
    print(f"VRAM使用量: {status.gpu.memory_used}GB")

# 実行
asyncio.run(main())
```

### 高度な使用方法

```python
# バッチ推論
results = await client.batch_reasoning([
    "質問1",
    "質問2",
    "質問3"
])

# 進化的学習の開始
evolution_id = await client.start_evolution(
    training_data=training_data,
    generations=5
)

# 学習進捗の監視
while True:
    progress = await client.get_evolution_progress(evolution_id)
    print(f"進捗: {progress.progress * 100:.1f}%")

    if progress.status == "completed":
        break

    await asyncio.sleep(10)

# 設定の更新
await client.update_config({
    "gpu": {"max_vram_gb": 4.5}
})
```

## 🔌 WebSocket API

### 接続

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onopen = function (event) {
  console.log("WebSocket接続が確立されました");
};

ws.onmessage = function (event) {
  const data = JSON.parse(event.data);
  console.log("受信:", data);
};
```

### メッセージ形式

#### チャットメッセージ

```json
{
  "type": "chat",
  "data": {
    "message": "こんにちは",
    "use_memory": true,
    "use_cot": true
  }
}
```

#### システム監視の購読

```json
{
  "type": "subscribe",
  "data": {
    "events": ["system_status", "alerts", "performance"]
  }
}
```

#### レスポンス

```json
{
  "type": "chat_response",
  "data": {
    "response": "こんにちは！何かお手伝いできることはありますか？",
    "session_id": "session_123"
  }
}
```

## 📝 エラーハンドリング

### エラーレスポンス形式

```json
{
  "error": {
    "code": "VRAM_INSUFFICIENT",
    "message": "VRAM不足のため推論を実行できません",
    "details": {
      "required_vram": "6.0GB",
      "available_vram": "4.2GB",
      "suggestion": "軽量モデルを使用するか、量子化レベルを上げてください"
    }
  },
  "timestamp": "2024-01-15T12:00:00Z",
  "request_id": "req_123"
}
```

### 一般的なエラーコード

| コード              | 説明                   | 対処法                                 |
| ------------------- | ---------------------- | -------------------------------------- |
| `VRAM_INSUFFICIENT` | VRAM 不足              | 軽量モデルの使用、量子化レベルの調整   |
| `MODEL_NOT_FOUND`   | モデルが見つからない   | Ollama でモデルをダウンロード          |
| `INFERENCE_TIMEOUT` | 推論タイムアウト       | タイムアウト値の調整、軽量モデルの使用 |
| `MEMORY_FULL`       | 記憶システムの容量不足 | 古い記憶の削除、容量制限の調整         |
| `CONFIG_INVALID`    | 設定が無効             | 設定ファイルの確認と修正               |

### Python でのエラーハンドリング

```python
from src.advanced_agent.client import AdvancedAgentClient, AgentError

async def handle_errors():
    client = AdvancedAgentClient("http://localhost:8000")

    try:
        response = await client.chat("複雑な質問")
    except AgentError as e:
        if e.code == "VRAM_INSUFFICIENT":
            print("VRAM不足です。軽量モデルを試します...")
            response = await client.chat(
                "複雑な質問",
                model="qwen2:1.5b-instruct-q4_k_m"
            )
        else:
            print(f"エラー: {e.message}")
            raise
```

## 🔒 セキュリティ

### API キー認証（本番環境）

```python
# 環境変数でAPIキーを設定
import os
os.environ['AGENT_API_KEY'] = 'your-secret-api-key'

# クライアントでAPIキーを使用
client = AdvancedAgentClient(
    "http://localhost:8000",
    api_key="your-secret-api-key"
)
```

### レート制限

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "レート制限を超えました",
    "details": {
      "limit": "60 requests per minute",
      "reset_time": "2024-01-15T12:01:00Z"
    }
  }
}
```

## 📊 パフォーマンス最適化

### バッチリクエスト

```python
# 複数の推論を効率的に実行
responses = await client.batch_chat([
    "質問1",
    "質問2",
    "質問3"
], parallel=True)
```

### キャッシュの活用

```python
# 記憶システムを活用した効率的な推論
response = await client.chat(
    "以前話した機械学習について詳しく教えて",
    use_memory=True,  # 関連する過去の会話を自動的に参照
    cache_ttl=3600    # 1時間キャッシュ
)
```

### 接続プールの使用

```python
# 複数の同時接続を効率的に管理
client = AdvancedAgentClient(
    "http://localhost:8000",
    max_connections=10,
    timeout=30
)
```

---

**📡 この API リファレンスを参考に、Advanced Self-Learning AI Agent を効果的に活用してください！**
