# トラブルシューティングガイド

## 概要

Advanced Self-Learning AI Agent の使用中に発生する可能性のある問題と、その解決方法について説明します。問題の種類別に整理し、段階的な解決手順を提供します。

## 🚨 緊急時の対処法

### システムが応答しない場合

```bash
# 1. プロセスの確認
ps aux | grep python | grep advanced_agent

# 2. 強制終了
pkill -f "advanced_agent"

# 3. GPU プロセスの確認・終了
nvidia-smi
sudo kill -9 <GPU_PROCESS_ID>

# 4. 安全モードでの再起動
AGENT_MAX_VRAM_GB=3.0 python -m src.advanced_agent.interfaces.streamlit_app
```

### VRAM 不足による緊急停止

```bash
# 緊急時の軽量モード起動
export AGENT_OLLAMA_PRIMARY_MODEL="qwen2:1.5b-instruct-q4_k_m"
export AGENT_MAX_VRAM_GB=2.5
export AGENT_QUANTIZATION_LEVELS="3"

python -m src.advanced_agent.interfaces.streamlit_app
```

## 💾 VRAM・メモリ関連の問題

### 問題 1: "CUDA out of memory" エラー

#### 症状

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 6.00 GiB total capacity; 4.50 GiB already allocated)
```

#### 原因

- VRAM 使用量が 6GB 制限を超過
- 複数のモデルが同時にロードされている
- 量子化設定が不適切

#### 解決手順

**ステップ 1: 即座の対処**

```bash
# GPU メモリをクリア
python -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
"
```

**ステップ 2: 設定調整**

```yaml
# config/system.yaml
gpu:
  max_vram_gb: 4.0 # 5.0から4.0に削減
  quantization_levels: [4, 3] # より積極的な量子化
  memory_reserve_gb: 1.0 # 予約メモリを増加
```

**ステップ 3: モデル変更**

```yaml
# config/system.yaml
models:
  primary: "qwen2.5:7b-instruct-q4_k_m" # より軽量なモデル
  fallback: "qwen2:1.5b-instruct-q4_k_m"
```

**ステップ 4: 動的最適化の有効化**

```python
# 自動最適化の実行
from src.advanced_agent.optimization.auto_optimizer import AutoOptimizer

async def fix_vram_issue():
    optimizer = AutoOptimizer()
    result = await optimizer.optimize_vram_usage()
    print(f"最適化結果: {result.status}")
    print(f"VRAM節約: {result.vram_saved_gb:.1f}GB")

import asyncio
asyncio.run(fix_vram_issue())
```

### 問題 2: メモリリークによる性能低下

#### 症状

- 時間経過とともに応答速度が低下
- VRAM 使用量が徐々に増加
- システムが不安定になる

#### 診断方法

```python
# メモリ使用量の監視
from src.advanced_agent.monitoring.system_monitor import SystemMonitor

async def diagnose_memory_leak():
    monitor = SystemMonitor()

    for i in range(10):
        status = await monitor.get_system_status()
        print(f"時刻 {i}: VRAM使用量 {status.gpu.memory_used:.2f}GB")
        await asyncio.sleep(60)  # 1分間隔で監視

asyncio.run(diagnose_memory_leak())
```

#### 解決方法

```python
# 定期的なメモリクリーンアップ
class MemoryManager:
    def __init__(self):
        self.cleanup_interval = 300  # 5分間隔

    async def start_cleanup_scheduler(self):
        while True:
            await self.cleanup_memory()
            await asyncio.sleep(self.cleanup_interval)

    async def cleanup_memory(self):
        # GPU メモリクリア
        torch.cuda.empty_cache()

        # Python ガベージコレクション
        import gc
        gc.collect()

        # 未使用モデルのアンロード
        await self._unload_unused_models()
```

### 問題 3: スワップメモリの過度な使用

#### 症状

- システム全体が重くなる
- ディスクアクセスが頻繁に発生
- 応答時間が著しく遅い

#### 解決方法

```yaml
# config/system.yaml
memory:
  system_ram_gb: 16 # 使用可能RAM量を正確に設定
  swap_limit_gb: 4 # スワップ使用量を制限
  cache_size_mb: 256 # キャッシュサイズを削減

cpu:
  offload_threshold: 0.9 # CPUオフロードを遅らせる
```

## 🤖 Ollama 関連の問題

### 問題 1: Ollama サービスに接続できない

#### 症状

```
ConnectionError: Cannot connect to Ollama service at http://localhost:11434
```

#### 解決手順

**ステップ 1: サービス状態確認**

```bash
# Ollama サービスの状態確認
ollama list
ollama ps

# サービスが動作していない場合
ollama serve
```

**ステップ 2: ポート確認**

```bash
# ポート11434の使用状況確認
netstat -tulpn | grep 11434
lsof -i :11434
```

**ステップ 3: 設定確認**

```bash
# 環境変数の確認
echo $OLLAMA_HOST
echo $OLLAMA_ORIGINS

# 設定ファイルの確認
cat config/.env | grep OLLAMA
```

**ステップ 4: 手動起動**

```bash
# Ollama を手動で起動
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 別ターミナルでモデル確認
ollama list
```

### 問題 2: モデルのダウンロードに失敗

#### 症状

```
Error: failed to pull model 'deepseek-r1:7b': network error
```

#### 解決方法

**ステップ 1: ネットワーク確認**

```bash
# インターネット接続確認
ping -c 4 ollama.ai
curl -I https://ollama.ai

# プロキシ設定確認
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

**ステップ 2: 手動ダウンロード**

```bash
# 段階的ダウンロード
ollama pull qwen2:1.5b-instruct-q4_k_m  # 軽量モデルから
ollama pull qwen2.5:7b-instruct-q4_k_m  # 中程度モデル
ollama pull deepseek-r1:7b               # 高性能モデル
```

**ステップ 3: 代替モデルの使用**

```yaml
# config/system.yaml - 利用可能なモデルのみ使用
models:
  primary: "qwen2.5:7b-instruct-q4_k_m"
  fallback: "qwen2:1.5b-instruct-q4_k_m"
  emergency: "qwen2:1.5b-instruct-q4_k_m"
```

### 問題 3: モデルの推論が遅い

#### 症状

- 応答時間が 10 秒以上
- GPU 使用率が低い
- CPU が高負荷

#### 診断・解決方法

**ステップ 1: 性能測定**

```python
# 推論性能の測定
from src.advanced_agent.monitoring.performance_analyzer import PerformanceAnalyzer

async def measure_inference_performance():
    analyzer = PerformanceAnalyzer()

    # ベンチマークテスト
    result = await analyzer.benchmark_inference(
        queries=["簡単な質問", "複雑な質問", "長い質問"],
        iterations=5
    )

    print(f"平均応答時間: {result.avg_response_time:.2f}秒")
    print(f"GPU使用率: {result.avg_gpu_utilization:.1f}%")
    print(f"スループット: {result.throughput:.1f} queries/sec")

asyncio.run(measure_inference_performance())
```

**ステップ 2: 最適化適用**

```python
# 自動最適化の実行
from src.advanced_agent.optimization.prometheus_optimizer import PrometheusOptimizer

async def optimize_inference_speed():
    optimizer = PrometheusOptimizer()

    # 現在の性能測定
    baseline = await optimizer.measure_baseline_performance()

    # 最適化実行
    optimizations = await optimizer.optimize_inference_pipeline()

    # 最適化後の性能測定
    optimized = await optimizer.measure_optimized_performance()

    improvement = ((optimized.response_time - baseline.response_time) / baseline.response_time) * 100
    print(f"応答時間改善: {improvement:.1f}%")

asyncio.run(optimize_inference_speed())
```

## 🧠 推論・学習関連の問題

### 問題 1: Chain-of-Thought 推論が機能しない

#### 症状

- 推論ステップが表示されない
- 論理的でない回答
- エラーメッセージが表示される

#### 解決方法

**ステップ 1: 設定確認**

```python
# CoT設定の確認
from src.advanced_agent.core.config import get_config

config = get_config()
print(f"CoT有効: {config.reasoning.enable_cot}")
print(f"推論ステップ数: {config.reasoning.max_steps}")
print(f"使用モデル: {config.models.primary}")
```

**ステップ 2: 手動テスト**

```python
# CoTエンジンの直接テスト
from src.advanced_agent.reasoning.chain_of_thought import ChainOfThoughtEngine

async def test_cot_engine():
    engine = ChainOfThoughtEngine()

    try:
        result = await engine.reason_step_by_step(
            "2 + 2 = ? を段階的に計算してください"
        )

        print("推論ステップ:")
        for i, step in enumerate(result.reasoning_steps, 1):
            print(f"  {i}. {step}")

        print(f"最終回答: {result.final_answer}")

    except Exception as e:
        print(f"CoTエラー: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_cot_engine())
```

**ステップ 3: プロンプト調整**

```python
# プロンプトテンプレートの確認・修正
from src.advanced_agent.reasoning.prompt_manager import PromptManager

prompt_manager = PromptManager()

# カスタムCoTプロンプトの設定
custom_cot_prompt = """
あなたは段階的に考える AI アシスタントです。
以下の質問に対して、思考過程を明確に示しながら回答してください。

質問: {query}

思考過程:
1. まず問題を理解します
2. 解決に必要な情報を整理します
3. 段階的に解決策を考えます
4. 最終的な回答を導きます

回答:
"""

prompt_manager.update_template("cot_reasoning", custom_cot_prompt)
```

### 問題 2: 進化的学習が収束しない

#### 症状

- 世代を重ねても性能が向上しない
- フィットネススコアが不安定
- 学習が途中で停止する

#### 診断方法

```python
# 学習進捗の詳細監視
from src.advanced_agent.evolution.evolutionary_system import EvolutionaryLearningSystem

async def diagnose_evolution_issues():
    evolution_system = EvolutionaryLearningSystem()

    # 現在の学習状態を確認
    status = await evolution_system.get_learning_status()

    print(f"現在の世代: {status.current_generation}")
    print(f"個体群サイズ: {status.population_size}")
    print(f"最高フィットネス: {status.best_fitness}")
    print(f"平均フィットネス: {status.avg_fitness}")
    print(f"フィットネス分散: {status.fitness_variance}")

    # 個体群の多様性確認
    diversity = await evolution_system.calculate_population_diversity()
    print(f"個体群多様性: {diversity}")

    if diversity < 0.1:
        print("⚠️ 個体群の多様性が低すぎます。変異率を上げることを推奨します。")

asyncio.run(diagnose_evolution_issues())
```

#### 解決方法

```yaml
# config/system.yaml - 学習パラメータの調整
learning:
  adapter_pool_size: 15 # 個体群サイズを増加
  generation_size: 8 # 世代サイズを増加
  mutation_rate: 0.15 # 変異率を上げる
  crossover_rate: 0.8 # 交配率を上げる
  max_training_epochs: 5 # 訓練エポック数を削減
  learning_rate: 0.0005 # 学習率を上げる
  diversity_threshold: 0.1 # 多様性閾値を設定
```

### 問題 3: LoRA アダプターの訓練に失敗

#### 症状

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

#### 解決方法

**ステップ 1: デバイス設定の確認**

```python
# デバイス設定の診断
import torch

print(f"CUDA利用可能: {torch.cuda.is_available()}")
print(f"CUDA デバイス数: {torch.cuda.device_count()}")
print(f"現在のデバイス: {torch.cuda.current_device()}")
print(f"デバイス名: {torch.cuda.get_device_name(0)}")

# メモリ情報
if torch.cuda.is_available():
    print(f"総VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"使用VRAM: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")
```

**ステップ 2: 訓練設定の修正**

```python
# 正しいデバイス設定での訓練
from src.advanced_agent.adaptation.qlora_trainer import QLoRATrainer

async def fix_training_device_issues():
    trainer = QLoRATrainer()

    # デバイス設定を明示的に指定
    trainer.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer.force_device_consistency = True

    # 訓練実行
    result = await trainer.train_adapter(
        adapter_config={
            "r": 8,  # より小さなランク
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],  # 対象モジュールを限定
            "lora_dropout": 0.1
        },
        training_data=training_data
    )

    return result
```

## 📊 監視・性能関連の問題

### 問題 1: Prometheus メトリクスが収集されない

#### 症状

- Grafana ダッシュボードにデータが表示されない
- メトリクスエンドポイントにアクセスできない
- 監視アラートが機能しない

#### 解決手順

**ステップ 1: Prometheus サービス確認**

```bash
# Prometheus プロセス確認
ps aux | grep prometheus

# ポート確認
netstat -tulpn | grep 9090

# 設定ファイル確認
cat config/prometheus.yml
```

**ステップ 2: メトリクス収集の手動テスト**

```python
# メトリクス収集の直接テスト
from src.advanced_agent.monitoring.prometheus_collector import PrometheusCollector

async def test_metrics_collection():
    collector = PrometheusCollector()

    try:
        # メトリクス収集開始
        await collector.start_collection()

        # 5秒間収集
        await asyncio.sleep(5)

        # 収集されたメトリクスを確認
        metrics = await collector.get_current_metrics()

        print("収集されたメトリクス:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")

    except Exception as e:
        print(f"メトリクス収集エラー: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_metrics_collection())
```

**ステップ 3: 設定修正**

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "advanced-agent"
    static_configs:
      - targets: ["localhost:8000"]
    scrape_interval: 5s
    metrics_path: "/metrics"
```

### 問題 2: GPU 監視が機能しない

#### 症状

- GPU 使用率が 0%と表示される
- VRAM 使用量が取得できない
- GPU 温度が表示されない

#### 解決方法

**ステップ 1: NVIDIA ドライバー確認**

```bash
# NVIDIA ドライバー確認
nvidia-smi
nvcc --version

# NVIDIA ML ライブラリ確認
python -c "
try:
    import pynvml
    pynvml.nvmlInit()
    print('NVIDIA ML ライブラリ正常')

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    print(f'GPU: {name}')
except Exception as e:
    print(f'NVIDIA ML エラー: {e}')
"
```

**ステップ 2: 権限確認**

```bash
# GPU デバイスファイルの権限確認
ls -la /dev/nvidia*

# 現在のユーザーがdockerグループに属しているか確認
groups $USER

# 必要に応じてグループに追加
sudo usermod -a -G docker $USER
```

**ステップ 3: 監視システムの再起動**

```python
# GPU監視の手動初期化
from src.advanced_agent.monitoring.system_monitor import SystemMonitor

async def reinitialize_gpu_monitoring():
    monitor = SystemMonitor()

    try:
        # GPU監視の初期化
        await monitor.initialize_gpu_monitoring()

        # テスト実行
        gpu_stats = await monitor.collect_gpu_metrics()

        print("GPU監視正常:")
        print(f"  使用率: {gpu_stats.utilization}%")
        print(f"  VRAM: {gpu_stats.memory_used:.1f}GB / {gpu_stats.memory_total:.1f}GB")
        print(f"  温度: {gpu_stats.temperature}°C")

    except Exception as e:
        print(f"GPU監視エラー: {e}")

        # フォールバック: CPU監視のみ
        await monitor.fallback_to_cpu_only_monitoring()

asyncio.run(reinitialize_gpu_monitoring())
```

## 🌐 Web UI・API 関連の問題

### 問題 1: Streamlit アプリが起動しない

#### 症状

```
ModuleNotFoundError: No module named 'streamlit'
```

#### 解決方法

**ステップ 1: 依存関係の確認・再インストール**

```bash
# 仮想環境の確認
which python
pip list | grep streamlit

# 依存関係の再インストール
pip install -r requirements.txt

# Streamlit の個別インストール
pip install streamlit>=1.28.0
```

**ステップ 2: パス設定の確認**

```python
# Python パスの確認
import sys
print("Python パス:")
for path in sys.path:
    print(f"  {path}")

# プロジェクトルートの追加
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"プロジェクトルート追加: {project_root}")
```

**ステップ 3: 手動起動**

```bash
# 直接起動
cd /path/to/advanced-self-learning-agent
python -m streamlit run src/advanced_agent/interfaces/streamlit_app.py

# ポート指定起動
streamlit run src/advanced_agent/interfaces/streamlit_app.py --server.port 8502
```

### 問題 2: FastAPI サーバーが応答しない

#### 症状

- API エンドポイントにアクセスできない
- 502 Bad Gateway エラー
- タイムアウトエラー

#### 解決方法

**ステップ 1: サーバー状態確認**

```bash
# プロセス確認
ps aux | grep uvicorn
ps aux | grep fastapi

# ポート確認
netstat -tulpn | grep 8000
lsof -i :8000
```

**ステップ 2: 手動起動・テスト**

```bash
# 開発モードでの起動
uvicorn src.advanced_agent.interfaces.fastapi_gateway:app --reload --host 0.0.0.0 --port 8000

# ヘルスチェック
curl http://localhost:8000/health

# API ドキュメント確認
curl http://localhost:8000/docs
```

**ステップ 3: ログ確認・デバッグ**

```python
# FastAPI アプリの直接テスト
from src.advanced_agent.interfaces.fastapi_gateway import FastAPIGateway

async def test_fastapi_app():
    gateway = FastAPIGateway()

    # アプリケーションの初期化確認
    print(f"アプリ設定: {gateway.app.title}")
    print(f"ルート数: {len(gateway.app.routes)}")

    # 各ルートの確認
    for route in gateway.app.routes:
        print(f"  {route.methods} {route.path}")

asyncio.run(test_fastapi_app())
```

### 問題 3: WebSocket 接続が切断される

#### 症状

- リアルタイム更新が停止する
- 接続エラーメッセージが表示される
- ページの再読み込みが必要

#### 解決方法

**ステップ 1: 接続設定の調整**

```javascript
// WebSocket 接続の改善
const ws = new WebSocket("ws://localhost:8000/ws");

// 再接続機能の追加
let reconnectInterval = 1000;
const maxReconnectInterval = 30000;

function connect() {
  ws = new WebSocket("ws://localhost:8000/ws");

  ws.onopen = function (event) {
    console.log("WebSocket接続成功");
    reconnectInterval = 1000; // 再接続間隔をリセット
  };

  ws.onclose = function (event) {
    console.log("WebSocket接続切断、再接続中...");
    setTimeout(connect, reconnectInterval);
    reconnectInterval = Math.min(reconnectInterval * 2, maxReconnectInterval);
  };

  ws.onerror = function (error) {
    console.error("WebSocketエラー:", error);
  };
}
```

**ステップ 2: サーバー側の設定調整**

```python
# WebSocket 接続の安定化
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.heartbeat_interval = 30  # 30秒間隔でハートビート

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

        # ハートビート開始
        asyncio.create_task(self.heartbeat(websocket))

    async def heartbeat(self, websocket: WebSocket):
        try:
            while websocket in self.active_connections:
                await websocket.send_json({"type": "heartbeat"})
                await asyncio.sleep(self.heartbeat_interval)
        except WebSocketDisconnect:
            self.disconnect(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
```

## 🔧 設定・環境関連の問題

### 問題 1: 設定ファイルが読み込まれない

#### 症状

- デフォルト設定が使用される
- 環境変数が反映されない
- 設定変更が無効

#### 解決方法

**ステップ 1: 設定ファイルの存在確認**

```bash
# 設定ファイルの確認
ls -la config/
cat config/system.yaml
cat config/.env
```

**ステップ 2: 設定読み込みのテスト**

```python
# 設定読み込みの直接テスト
from src.advanced_agent.core.config import get_config, load_config

try:
    config = get_config()
    print("設定読み込み成功:")
    print(f"  GPU最大VRAM: {config.gpu.max_vram_gb}GB")
    print(f"  プライマリモデル: {config.models.primary}")
    print(f"  監視間隔: {config.monitoring.gpu_monitoring.interval_seconds}秒")

except Exception as e:
    print(f"設定読み込みエラー: {e}")

    # デフォルト設定の生成
    from src.advanced_agent.core.config import AdvancedAgentConfig
    default_config = AdvancedAgentConfig()
    print("デフォルト設定を使用します")
```

**ステップ 3: 環境変数の確認**

```bash
# 環境変数の確認
env | grep AGENT_
echo $AGENT_MAX_VRAM_GB
echo $AGENT_OLLAMA_PRIMARY_MODEL
```

### 問題 2: 権限エラー

#### 症状

```
PermissionError: [Errno 13] Permission denied: '/path/to/logs/advanced_agent.log'
```

#### 解決方法

**ステップ 1: ディレクトリ権限の確認・修正**

```bash
# ログディレクトリの権限確認
ls -la logs/
ls -la data/

# 権限修正
chmod 755 logs/
chmod 755 data/
chmod 644 logs/*.log
chmod 644 data/*.db
```

**ステップ 2: 所有者の確認・変更**

```bash
# 所有者確認
ls -la logs/ data/

# 所有者変更（必要に応じて）
sudo chown -R $USER:$USER logs/
sudo chown -R $USER:$USER data/
```

**ステップ 3: 代替パスの使用**

```python
# 権限問題の回避
import tempfile
import os

# 一時ディレクトリの使用
temp_dir = tempfile.mkdtemp(prefix="advanced_agent_")
os.environ['AGENT_LOG_DIR'] = temp_dir
os.environ['AGENT_DATA_DIR'] = temp_dir

print(f"一時ディレクトリを使用: {temp_dir}")
```

## 📋 診断ツール

### 総合診断スクリプト

```python
#!/usr/bin/env python3
"""
Advanced Self-Learning AI Agent 総合診断ツール
"""

import asyncio
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def comprehensive_diagnosis():
    """総合診断の実行"""

    print("🔍 Advanced Self-Learning AI Agent 総合診断")
    print("=" * 60)

    # 1. システム要件確認
    print("\n1. システム要件確認")
    print("-" * 30)

    try:
        from src.advanced_agent.core.environment import validate_environment_startup
        report = validate_environment_startup()
        print(f"✅ システム要件: {report.overall_status}")
    except Exception as e:
        print(f"❌ システム要件確認エラー: {e}")

    # 2. GPU確認
    print("\n2. GPU確認")
    print("-" * 30)

    try:
        import torch
        print(f"CUDA利用可能: {'✅' if torch.cuda.is_available() else '❌'}")
        if torch.cuda.is_available():
            print(f"GPU名: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    except Exception as e:
        print(f"❌ GPU確認エラー: {e}")

    # 3. Ollama確認
    print("\n3. Ollama確認")
    print("-" * 30)

    try:
        from src.advanced_agent.inference.ollama_client import OllamaClient
        client = OllamaClient()
        models = await client.list_models()
        print(f"✅ Ollama接続成功")
        print(f"利用可能モデル: {len(models)}個")
        for model in models[:3]:  # 最初の3個を表示
            print(f"  - {model}")
    except Exception as e:
        print(f"❌ Ollama確認エラー: {e}")

    # 4. 設定確認
    print("\n4. 設定確認")
    print("-" * 30)

    try:
        from src.advanced_agent.core.config import get_config
        config = get_config()
        print(f"✅ 設定読み込み成功")
        print(f"最大VRAM: {config.gpu.max_vram_gb}GB")
        print(f"プライマリモデル: {config.models.primary}")
    except Exception as e:
        print(f"❌ 設定確認エラー: {e}")

    # 5. 記憶システム確認
    print("\n5. 記憶システム確認")
    print("-" * 30)

    try:
        from src.advanced_agent.memory.persistent_memory import PersistentMemoryManager
        memory = PersistentMemoryManager()
        await memory.initialize()
        print(f"✅ 記憶システム正常")
    except Exception as e:
        print(f"❌ 記憶システムエラー: {e}")

    # 6. 推論エンジン確認
    print("\n6. 推論エンジン確認")
    print("-" * 30)

    try:
        from src.advanced_agent.reasoning.basic_engine import BasicReasoningEngine
        engine = BasicReasoningEngine()
        result = await engine.reason("テスト質問: 1+1は？")
        print(f"✅ 推論エンジン正常")
        print(f"テスト応答: {result.content[:50]}...")
    except Exception as e:
        print(f"❌ 推論エンジンエラー: {e}")

    print("\n" + "=" * 60)
    print("診断完了")

if __name__ == "__main__":
    asyncio.run(comprehensive_diagnosis())
```

### 性能ベンチマークツール

```python
#!/usr/bin/env python3
"""
性能ベンチマークツール
"""

import asyncio
import time
from typing import List, Dict

async def performance_benchmark():
    """性能ベンチマークの実行"""

    print("🚀 性能ベンチマーク開始")
    print("=" * 50)

    # テスト質問
    test_queries = [
        "簡単な質問: 今日の天気は？",
        "中程度の質問: Pythonでソートアルゴリズムを実装してください",
        "複雑な質問: 機械学習の基本概念と応用について詳しく説明してください"
    ]

    from src.advanced_agent.reasoning.basic_engine import BasicReasoningEngine
    engine = BasicReasoningEngine()

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\nテスト {i}: {query[:30]}...")

        start_time = time.time()

        try:
            result = await engine.reason(query)
            end_time = time.time()

            response_time = end_time - start_time
            results.append({
                "query": query,
                "response_time": response_time,
                "success": True,
                "response_length": len(result.content)
            })

            print(f"  応答時間: {response_time:.2f}秒")
            print(f"  応答長: {len(result.content)}文字")

        except Exception as e:
            end_time = time.time()
            results.append({
                "query": query,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e)
            })
            print(f"  エラー: {e}")

    # 結果サマリー
    print("\n" + "=" * 50)
    print("ベンチマーク結果サマリー")
    print("=" * 50)

    successful_results = [r for r in results if r["success"]]

    if successful_results:
        avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
        print(f"平均応答時間: {avg_response_time:.2f}秒")
        print(f"成功率: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")

        if avg_response_time <= 2.0:
            print("✅ 性能目標達成（2秒以内）")
        else:
            print("⚠️ 性能目標未達成（2秒超過）")
    else:
        print("❌ すべてのテストが失敗しました")

if __name__ == "__main__":
    asyncio.run(performance_benchmark())
```

## 📞 サポート・コミュニティ

### 問題報告の方法

1. **GitHub Issues**: バグ報告・機能要望
2. **Discussions**: 質問・議論
3. **Wiki**: コミュニティドキュメント

### 問題報告時に含める情報

```bash
# システム情報の収集
echo "=== システム情報 ==="
uname -a
python --version
pip list | grep -E "(torch|transformers|langchain|ollama)"

echo "=== GPU情報 ==="
nvidia-smi

echo "=== 設定情報 ==="
cat config/system.yaml | head -20

echo "=== ログ情報 ==="
tail -50 logs/advanced_agent.log
```

---

**🔧 このトラブルシューティングガイドを参考に、問題を効率的に解決してください！**
