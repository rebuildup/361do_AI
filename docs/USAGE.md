# 使用方法ガイド

## 概要

Advanced Self-Learning AI Agent は、RTX 4050 6GB VRAM 環境で動作する高性能自己学習 AI エージェントシステムです。このガイドでは、システムの基本的な使用方法から高度な機能まで、実践的な使用例を交えて説明します。

## 🚀 基本的な使用方法

### 1. システムの起動

#### 1.1 環境の確認

```bash
# システム要件の確認
python -m src.advanced_agent.core.environment

# GPU設定の確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Ollamaの確認
ollama list
```

#### 1.2 Web UI での使用（推奨）

```bash
# Streamlit Web UI の起動
streamlit run src/advanced_agent/interfaces/streamlit_app.py

# ブラウザで http://localhost:8501 にアクセス
```

#### 1.3 API サーバーでの使用

```bash
# FastAPI サーバーの起動
python -m src.advanced_agent.interfaces.fastapi_gateway

# API ドキュメント: http://localhost:8000/docs
```

#### 1.4 CLI での使用

```bash
# 基本的な推論実行
python -m src.advanced_agent.reasoning.demo

# 記憶システムのテスト
python -m src.advanced_agent.memory.demo

# 監視システムの確認
python -m src.advanced_agent.monitoring.demo
```

### 2. 基本的な対話

#### 2.1 Web UI での対話

1. ブラウザで http://localhost:8501 にアクセス
2. サイドバーで設定を調整
3. メッセージ入力欄に質問を入力
4. 「送信」ボタンをクリック
5. リアルタイムで応答を確認

#### 2.2 API での対話

```python
import requests

# 基本的なチャット
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "こんにちは、自己学習について教えてください",
        "use_memory": True,
        "use_cot": True
    }
)

print(response.json()["response"])
```

#### 2.3 Python クライアントでの対話

```python
import asyncio
from src.advanced_agent.reasoning.basic_engine import BasicReasoningEngine

async def chat_example():
    engine = BasicReasoningEngine()

    # 基本的な推論
    response = await engine.reason(
        "複雑な数学問題を段階的に解いてください: 2x + 5 = 15"
    )

    print(f"応答: {response.content}")
    print(f"推論ステップ: {response.reasoning_steps}")

# 実行
asyncio.run(chat_example())
```

## 🧠 推論機能の使用

### 1. Chain-of-Thought 推論

```python
from src.advanced_agent.reasoning.chain_of_thought import ChainOfThoughtEngine

async def cot_example():
    engine = ChainOfThoughtEngine()

    # 段階的推論の実行
    result = await engine.reason_step_by_step(
        "なぜ地球は丸いのですか？物理学的な根拠を含めて説明してください"
    )

    # 推論過程の表示
    for i, step in enumerate(result.reasoning_steps, 1):
        print(f"ステップ {i}: {step}")

    print(f"\n最終回答: {result.final_answer}")

asyncio.run(cot_example())
```

### 2. モデルの動的切り替え

```python
from src.advanced_agent.inference.ollama_client import OllamaClient

async def model_switching_example():
    client = OllamaClient()

    # 高性能モデルでの推論
    response1 = await client.generate(
        "複雑な哲学的問題について考察してください",
        model="deepseek-r1:7b"
    )

    # 軽量モデルでの推論
    response2 = await client.generate(
        "簡単な質問に答えてください",
        model="qwen2:1.5b-instruct-q4_k_m"
    )

    print(f"高性能モデル: {response1}")
    print(f"軽量モデル: {response2}")

asyncio.run(model_switching_example())
```

## 💾 記憶システムの使用

### 1. 永続的記憶の活用

```python
from src.advanced_agent.memory.persistent_memory import PersistentMemoryManager

async def memory_example():
    memory = PersistentMemoryManager()

    # 重要な情報の保存
    await memory.store_memory(
        content="ユーザーはPythonプログラミングに興味がある",
        importance=0.8,
        memory_type="user_preference"
    )

    # 関連記憶の検索
    related_memories = await memory.search_memories(
        query="プログラミング",
        limit=5
    )

    for memory in related_memories:
        print(f"記憶: {memory.content} (重要度: {memory.importance})")

asyncio.run(memory_example())
```

### 2. セッション管理

```python
from src.advanced_agent.memory.session_manager import SessionManager

async def session_example():
    session_mgr = SessionManager()

    # 新しいセッションの開始
    session_id = await session_mgr.create_session("user_123")

    # セッション内での対話
    await session_mgr.add_message(
        session_id,
        "user",
        "機械学習について教えてください"
    )

    await session_mgr.add_message(
        session_id,
        "assistant",
        "機械学習は..."
    )

    # セッション履歴の取得
    history = await session_mgr.get_session_history(session_id)

    for msg in history:
        print(f"{msg.role}: {msg.content}")

asyncio.run(session_example())
```

## 🧬 学習機能の使用

### 1. 進化的学習の実行

```python
from src.advanced_agent.evolution.evolutionary_system import EvolutionaryLearningSystem

async def evolution_example():
    evolution_system = EvolutionaryLearningSystem()

    # 学習データの準備
    training_data = [
        {"input": "質問1", "output": "回答1"},
        {"input": "質問2", "output": "回答2"},
        # ... 更多数据
    ]

    # 進化的学習の実行
    best_adapter = await evolution_system.evolve_adapters(
        training_data=training_data,
        generations=5,
        population_size=10
    )

    print(f"最適なアダプター: {best_adapter.config}")
    print(f"性能スコア: {best_adapter.fitness_score}")

asyncio.run(evolution_example())
```

### 2. LoRA アダプターの管理

```python
from src.advanced_agent.adaptation.peft_manager import PEFTManager

async def adapter_example():
    peft_manager = PEFTManager()

    # 新しいアダプターの作成
    adapter_config = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1
    }

    adapter_id = await peft_manager.create_adapter(
        "custom_adapter",
        adapter_config
    )

    # アダプターの訓練
    await peft_manager.train_adapter(
        adapter_id,
        training_data,
        epochs=3
    )

    # アダプターの評価
    metrics = await peft_manager.evaluate_adapter(
        adapter_id,
        test_data
    )

    print(f"アダプター性能: {metrics}")

asyncio.run(adapter_example())
```

## 📊 監視機能の使用

### 1. リアルタイム監視

```python
from src.advanced_agent.monitoring.system_monitor import SystemMonitor

async def monitoring_example():
    monitor = SystemMonitor()

    # 監視の開始
    await monitor.start_monitoring()

    # 現在のシステム状態の取得
    status = await monitor.get_system_status()

    print(f"GPU使用率: {status.gpu_utilization}%")
    print(f"VRAM使用量: {status.vram_used_gb:.1f}GB / {status.vram_total_gb:.1f}GB")
    print(f"CPU使用率: {status.cpu_utilization}%")
    print(f"RAM使用量: {status.ram_used_gb:.1f}GB / {status.ram_total_gb:.1f}GB")

asyncio.run(monitoring_example())
```

### 2. 性能最適化

```python
from src.advanced_agent.optimization.auto_optimizer import AutoOptimizer

async def optimization_example():
    optimizer = AutoOptimizer()

    # 自動最適化の実行
    optimization_result = await optimizer.optimize_system()

    print(f"最適化結果: {optimization_result.status}")
    print(f"性能向上: {optimization_result.performance_improvement}%")
    print(f"VRAM節約: {optimization_result.vram_saved_gb:.1f}GB")

asyncio.run(optimization_example())
```

## 🔧 設定のカスタマイズ

### 1. GPU 設定の調整

```yaml
# config/system.yaml
gpu:
  max_vram_gb: 4.5 # VRAM使用量を4.5GBに制限
  quantization_levels: [4, 3] # より積極的な量子化
  temperature_threshold: 75 # 温度閾値を下げる
```

### 2. モデル設定の変更

```yaml
# config/system.yaml
models:
  primary: "qwen2.5:7b-instruct-q4_k_m" # より軽量なモデルを使用
  fallback: "qwen2:1.5b-instruct-q4_k_m"
  emergency: "qwen2:1.5b-instruct-q4_k_m"
```

### 3. 記憶システムの調整

```yaml
# config/system.yaml
persistent_memory:
  max_short_term_items: 500 # 短期記憶を削減
  max_long_term_items: 5000 # 長期記憶を削減
  importance_threshold: 0.8 # より厳しい重要度閾値
```

## 🎯 実用的な使用例

### 1. プログラミング支援

```python
# Web UIまたはAPIで以下のような質問を送信
query = """
Pythonで機械学習のデータ前処理を行うコードを書いてください。
以下の要件を満たしてください：
1. CSVファイルの読み込み
2. 欠損値の処理
3. カテゴリカル変数のエンコーディング
4. 数値の正規化
"""

# システムが段階的に推論し、完全なコードを生成
```

### 2. 学習支援

```python
query = """
量子力学の基本概念について、初心者にもわかりやすく説明してください。
特に以下の点を含めてください：
1. 波動関数とは何か
2. 不確定性原理
3. 量子もつれ
4. 実生活への応用例
"""

# システムが記憶を活用して、ユーザーのレベルに合わせた説明を生成
```

### 3. 問題解決支援

```python
query = """
Webアプリケーションの性能が低下しています。
以下の症状から原因を特定し、解決策を提案してください：
- ページの読み込みが遅い
- データベースクエリが多い
- メモリ使用量が高い
- CPU使用率が常に80%以上
"""

# システムが段階的に分析し、具体的な解決策を提案
```

## 🔍 トラブルシューティング

### 1. VRAM 不足の対処

```bash
# 軽量設定での起動
AGENT_MAX_VRAM_GB=3.5 python -m src.advanced_agent.interfaces.streamlit_app

# または設定ファイルを編集
# config/system.yaml の gpu.max_vram_gb を 3.5 に変更
```

### 2. 推論速度の改善

```yaml
# config/system.yaml
cpu:
  max_threads: 20 # CPUスレッド数を増加
  offload_threshold: 0.6 # CPUオフロードを早める

gpu:
  quantization_levels: [4] # 4bit量子化のみ使用
```

### 3. 記憶システムの最適化

```bash
# データベースの最適化
python -c "
from src.advanced_agent.memory.persistent_memory import PersistentMemoryManager
import asyncio
asyncio.run(PersistentMemoryManager().optimize_database())
"
```

## 📈 性能監視とメトリクス

### 1. Web UI での監視

- サイドバーの「システム状態」セクションでリアルタイム監視
- GPU 使用率、VRAM 使用量、温度を確認
- 推論速度と精度のメトリクスを表示

### 2. CLI での監視

```bash
# システム状態の確認
python -m src.advanced_agent.monitoring.demo

# 詳細な性能分析
python -m src.advanced_agent.monitoring.performance_analyzer
```

### 3. ログの確認

```bash
# リアルタイムログの確認
tail -f logs/advanced_agent.log

# エラーログの検索
grep "ERROR" logs/advanced_agent.log
```

## 🚀 高度な使用方法

### 1. カスタムツールの追加

```python
from src.advanced_agent.inference.tools import ToolRegistry

# カスタムツールの定義
def custom_calculator(expression: str) -> str:
    """数式を計算するツール"""
    try:
        result = eval(expression)
        return f"計算結果: {result}"
    except Exception as e:
        return f"エラー: {e}"

# ツールの登録
registry = ToolRegistry()
registry.register_tool("calculator", custom_calculator)
```

### 2. マルチモーダル機能の使用

```python
from src.advanced_agent.multimodal.document_ai import DocumentAI

async def multimodal_example():
    doc_ai = DocumentAI()

    # 画像の分析
    result = await doc_ai.analyze_image("path/to/image.jpg")
    print(f"画像分析結果: {result}")

    # ドキュメントの処理
    doc_result = await doc_ai.process_document("path/to/document.pdf")
    print(f"ドキュメント分析: {doc_result}")

asyncio.run(multimodal_example())
```

### 3. バッチ処理

```python
from src.advanced_agent.reasoning.basic_engine import BasicReasoningEngine

async def batch_processing():
    engine = BasicReasoningEngine()

    questions = [
        "質問1: ...",
        "質問2: ...",
        "質問3: ..."
    ]

    # 並列処理で効率的に実行
    tasks = [engine.reason(q) for q in questions]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(f"質問{i+1}の回答: {result.content}")

asyncio.run(batch_processing())
```

## 📚 次のステップ

1. **[API リファレンス](API_REFERENCE.md)** - 詳細な API 仕様
2. **[アーキテクチャ](ARCHITECTURE.md)** - システム設計の理解
3. **[カスタマイズガイド](CUSTOMIZATION.md)** - 高度なカスタマイズ方法
4. **[デプロイメントガイド](DEPLOYMENT.md)** - 本番環境での運用

---

**🎯 このガイドを参考に、Advanced Self-Learning AI Agent を最大限活用してください！**
