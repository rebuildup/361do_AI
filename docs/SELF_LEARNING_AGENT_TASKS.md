# 自己学習 AI エージェント実装タスク

## プロジェクト概要

完全な自然言語を与えるだけでエージェントが自主的に自己学習やエージェントを動かし、自身の機能を使う自己学習 AI エージェントを構築する。

## システム制約

- **GPU VRAM**: 6GB (RTX 4050 Laptop)
- **システム RAM**: 32GB
- **CPU**: Intel i7-13700H (16 コア)
- **既存環境**: Ollama + Qwen2:7b-instruct (4.4GB)

## 実装要件

### 基本要件

- セッションをクリアせず、ずっと会話が続く方式
- エージェント自身が自分のカスタムプロンプトを書き換える
- エージェント自身が自分のチューニングデータを操作する
- どこからエージェントを起動しても同じセッションで会話が続く
- Deepseek と同じような推論機能
- エージェント機能（ネット検索、コマンド実行、ファイル書き換え、MCP の使用）
- SAKANA AI の AI 交配進化システム
- ユーザーと関わることを報酬とした報酬系システム

### 技術要件

- RTX 4050 6GB VRAM 最適化
- 永続的記憶システム
- リアルタイム監視
- エラー処理とフォールバック

### 開発方針

- **構成はできるだけシンプルに**
- **サンプルコードがネット上にあるものはできるだけそのまま使う**
- **ルートのフォルダには本当に最低限のファイルしか置かない**
- **不要になったファイルはすぐに削除する**
- **playwright mcp を使った実際のテストを重視する**

## 詳細タスク一覧（1 タスク 5 ファイル以下）

### Phase 1: 基盤システム構築

#### タスク 1: 永続セッション管理システム ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: セッション状態の完全永続化と復元機能

**実装内容**:

- `src/advanced_agent/core/session_manager.py` - セッション管理コア
- `src/advanced_agent/memory/session_storage.py` - セッション永続化
- `tests/unit/test_session_manager.py` - セッションテスト

#### タスク 2: データベーススキーマ実装 ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: エージェント状態、プロンプト、チューニングデータの永続化

**実装内容**:

- [x] `src/advanced_agent/database/models.py` - SQLAlchemy モデル定義
- [x] `src/advanced_agent/database/migrations.py` - マイグレーション
- [x] `src/advanced_agent/database/connection.py` - データベース接続管理
- [x] `tests/unit/test_database.py` - データベーステスト

#### タスク 3: 設定管理システム ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: システム設定の動的管理とバリデーション

**実装内容**:

- [x] `src/advanced_agent/config/settings.py` - Pydantic 設定
- [x] `src/advanced_agent/config/loader.py` - 設定読み込み
- [x] `config/agent_config.yaml` - 設定ファイル

### Phase 2: 推論エンジン構築

#### タスク 4: 基本推論エンジン ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: Ollama 統合と Chain-of-Thought 推論

**実装内容**:

- [x] `src/advanced_agent/reasoning/ollama_client.py` - Ollama クライアント
- [x] `src/advanced_agent/reasoning/cot_engine.py` - Chain-of-Thought 実装
- [x] `src/advanced_agent/reasoning/prompt_templates.py` - プロンプトテンプレート
- [x] `tests/unit/test_reasoning.py` - 推論テスト

#### タスク 5: 推論品質評価システム ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: 推論結果の品質測定と改善

**実装内容**:

- [x] `src/advanced_agent/reasoning/quality_evaluator.py` - 品質評価
- [x] `src/advanced_agent/reasoning/metrics.py` - 評価指標
- [x] `tests/unit/test_quality_evaluation.py` - 品質評価テスト

#### タスク 6: 推論結果構造化 ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: 推論ステップの構造化と可視化

**実装内容**:

- [x] `src/advanced_agent/reasoning/result_parser.py` - 結果解析
- [x] `src/advanced_agent/reasoning/visualizer.py` - 可視化
- [x] `tests/unit/test_result_parsing.py` - 解析テスト

### Phase 3: 記憶システム構築

#### タスク 7: ベクトル記憶システム ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: ChromaDB 統合とセマンティック検索

**実装内容**:

- [x] `src/advanced_agent/memory/vector_store.py` - ChromaDB 統合
- [x] `src/advanced_agent/memory/embedding_manager.py` - 埋め込み管理
- [x] `src/advanced_agent/memory/semantic_search.py` - セマンティック検索
- [x] `tests/unit/test_vector_memory.py` - ベクトル記憶テスト

#### タスク 8: 記憶重要度評価 ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: 記憶の重要度判定と自動整理

**実装内容**:

- [x] `src/advanced_agent/memory/importance_evaluator.py` - 重要度評価
- [x] `src/advanced_agent/memory/memory_cleaner.py` - 記憶整理
- [x] `tests/unit/test_memory_importance.py` - 重要度テスト

#### タスク 9: 記憶統合システム ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: 短期・長期記憶の統合管理

**実装内容**:

- [x] `src/advanced_agent/memory/memory_integrator.py` - 記憶統合
- [x] `src/advanced_agent/memory/context_builder.py` - コンテキスト構築
- [x] `tests/unit/test_memory_integration.py` - 統合テスト

### Phase 4: ツールシステム構築

#### タスク 10: 基本ツール実装 ✅

**ステータス**: 完了  
**編集ファイル数**: 5 ファイル  
**説明**: ネット検索、コマンド実行、ファイル操作ツール

**実装内容**:

- [x] `src/advanced_agent/tools/web_search.py` - ネット検索ツール
- [x] `src/advanced_agent/tools/command_executor.py` - コマンド実行ツール
- [x] `src/advanced_agent/tools/file_manager.py` - ファイル操作ツール
- [x] `src/advanced_agent/tools/tool_registry.py` - ツール登録
- [x] `tests/unit/test_tools.py` - ツールテスト

#### タスク 11: MCP 統合システム ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: Model Context Protocol 統合

**実装内容**:

- [x] `src/advanced_agent/tools/mcp_client.py` - MCP クライアント
- [x] `src/advanced_agent/tools/mcp_tools.py` - MCP ツール
- [x] `src/advanced_agent/tools/tool_chain.py` - ツールチェーン
- [x] `tests/unit/test_mcp_integration.py` - MCP テスト

#### タスク 12: 動的ツール生成 ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: LLM による動的ツール作成

**実装内容**:

- [x] `src/advanced_agent/tools/dynamic_tool_generator.py` - 動的生成
- [x] `src/advanced_agent/tools/tool_validator.py` - ツール検証
- [x] `tests/unit/test_dynamic_tools.py` - 動的ツールテスト

### Phase 5: 学習システム構築

#### タスク 13: プロンプト管理システム ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: プロンプトテンプレートの動的管理

**実装内容**:

- [x] `src/advanced_agent/learning/prompt_manager.py` - プロンプト管理
- [x] `src/advanced_agent/learning/prompt_evaluator.py` - プロンプト評価
- [x] `src/advanced_agent/learning/prompt_optimizer.py` - プロンプト最適化
- [x] `tests/unit/test_prompt_management.py` - プロンプトテスト

#### タスク 14: チューニングデータ管理 ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: チューニングデータの生成・管理・品質評価

**実装内容**:

- [x] `src/advanced_agent/learning/data_generator.py` - データ生成
- [x] `src/advanced_agent/learning/data_quality.py` - 品質評価
- [x] `src/advanced_agent/learning/data_manager.py` - データ管理
- [x] `tests/unit/test_tuning_data.py` - データテスト

#### タスク 15: 進化システム実装 ✅

**ステータス**: 完了  
**編集ファイル数**: 5 ファイル  
**説明**: SAKANA AI スタイルの進化アルゴリズム

**実装内容**:

- [x] `src/advanced_agent/evolution/genetic_algorithm.py` - 遺伝的アルゴリズム
- [x] `src/advanced_agent/evolution/fitness_evaluator.py` - 適応度評価
- [x] `src/advanced_agent/evolution/crossover.py` - 交配アルゴリズム
- [x] `src/advanced_agent/evolution/mutation.py` - 変異アルゴリズム
- [x] `tests/unit/test_evolution.py` - 進化テスト

### Phase 6: 報酬システム構築

#### タスク 16: 報酬計算システム ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: ユーザー関与度と回答品質の報酬計算

**実装内容**:

- [x] `src/advanced_agent/reward/reward_calculator.py` - 報酬計算
- [x] `src/advanced_agent/reward/engagement_analyzer.py` - 関与度分析
- [x] `tests/unit/test_reward_calculation.py` - 報酬テスト

#### タスク 17: 強化学習統合 ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: 強化学習アルゴリズムの統合

**実装内容**:

- [x] `src/advanced_agent/reward/rl_agent.py` - RL エージェント
- [x] `src/advanced_agent/reward/policy_network.py` - ポリシーネットワーク
- [x] `src/advanced_agent/reward/value_function.py` - 価値関数
- [x] `tests/unit/test_reinforcement_learning.py` - RL テスト

### Phase 7: 監視システム構築

#### タスク 18: GPU 監視システム ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: RTX 4050 のリアルタイム監視

**実装内容**:

- [x] `src/advanced_agent/monitoring/system_monitor.py` - GPU 監視統合
- [x] `src/advanced_agent/monitoring/performance_analyzer.py` - リソース最適化
- [x] `tests/unit/test_system_monitor.py` - GPU 監視テスト

#### タスク 19: パフォーマンス監視 ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: システム全体のパフォーマンス監視

**実装内容**:

- [x] `src/advanced_agent/monitoring/performance_analyzer.py` - パフォーマンス監視
- [x] `src/advanced_agent/monitoring/prometheus_collector.py` - メトリクス収集
- [x] `src/advanced_agent/monitoring/grafana_dashboard.py` - アラートシステム
- [x] `tests/unit/test_performance_monitoring.py` - パフォーマンステスト

### Phase 8: UI/API 構築

#### タスク 20: Streamlit UI 統合 ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: 既存 Streamlit UI との統合

**実装内容**:

- [x] `src/advanced_agent/interfaces/streamlit_ui.py` - Streamlit 統合
- [x] `src/advanced_agent/interfaces/chat_interface.py` - チャットインターフェース
- [x] `src/advanced_agent/interfaces/monitoring_dashboard.py` - 監視ダッシュボード
- [x] `tests/unit/test_streamlit_ui.py` - UI テスト

#### タスク 21: FastAPI 統合 ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: REST API と WebSocket 統合

**実装内容**:

- [x] `src/advanced_agent/interfaces/fastapi_gateway.py` - FastAPI アプリ
- [x] `src/advanced_agent/interfaces/api_models.py` - WebSocket 処理
- [x] `src/advanced_agent/interfaces/auth.py` - API エンドポイント
- [x] `tests/unit/test_fastapi_gateway.py` - API テスト

### Phase 9: テストシステム構築

#### タスク 22: Playwright MCP テスト ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: ブラウザ自動化テスト

**実装内容**:

- [x] `tests/e2e/playwright_test_suite.py` - Playwright テストスイート
- [x] `tests/e2e/browser_automation.py` - ブラウザ自動化
- [x] `tests/e2e/ui_validation.py` - UI 検証

#### タスク 23: 統合テストスイート ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: 全システムの統合テスト

**実装内容**:

- [x] `tests/integration/system_test.py` - システム統合テスト
- [x] `tests/integration/performance_test.py` - パフォーマンステスト
- [x] `tests/integration/load_test.py` - 負荷テスト
- [x] `tests/integration/stability_test.py` - 安定性テスト

### Phase 10: デプロイメント最適化

#### タスク 24: メモリ最適化 ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: VRAM 使用量の最適化

**実装内容**:

- [x] `src/advanced_agent/optimization/memory_manager.py` - メモリ管理
- [x] `src/advanced_agent/optimization/quantization.py` - 量子化最適化
- [x] `tests/performance/test_memory_optimization.py` - メモリテスト

#### タスク 25: 推論速度最適化 ✅

**ステータス**: 完了  
**編集ファイル数**: 3 ファイル  
**説明**: 推論処理の高速化

**実装内容**:

- [x] `src/advanced_agent/optimization/inference_optimizer.py` - 推論最適化
- [x] `src/advanced_agent/optimization/batch_processor.py` - バッチ処理
- [x] `tests/performance/test_inference_speed.py` - 速度テスト

#### タスク 26: デプロイメント設定 ✅

**ステータス**: 完了  
**編集ファイル数**: 4 ファイル  
**説明**: 本番環境デプロイメント

**実装内容**:

- [x] `Dockerfile` - コンテナ設定
- [x] `docker-compose.yml` - サービス構成
- [x] `scripts/deploy.sh` - デプロイスクリプト
- [x] `requirements.txt` - 依存関係管理
- [ ] `config/production.yaml` - 本番設定

## 技術スタック

### コア技術

- **Python 3.9+**: メイン開発言語
- **SQLite**: 永続化データベース
- **ChromaDB**: ベクトルデータベース
- **LangChain**: LLM 統合フレームワーク
- **Ollama**: ローカル LLM 実行

### 機械学習・AI

- **HuggingFace Transformers**: 埋め込みモデル
- **Sentence-Transformers**: セマンティック検索
- **PyTorch**: 深層学習フレームワーク
- **PEFT**: Parameter-Efficient Fine-Tuning
- **QLoRA**: 4bit 量子化 + LoRA

### 監視・ログ

- **Prometheus**: メトリクス収集
- **Grafana**: 可視化
- **Loguru**: ログ管理
- **pynvml**: GPU 監視

### UI・API

- **Streamlit**: Web UI
- **FastAPI**: REST API
- **WebSocket**: リアルタイム通信
- **Playwright**: ブラウザ自動化

### テスト

- **pytest**: テストフレームワーク
- **Playwright**: E2E テスト
- **locust**: 負荷テスト
- **memory_profiler**: メモリプロファイリング

## ファイル構成（シンプル化）

### ルートディレクトリ（最低限）

```
├── main.py                 # メインエントリーポイント
├── requirements.txt        # 依存関係
├── config.yaml            # 設定ファイル
├── README.md              # ドキュメント
└── .gitignore             # Git除外設定
```

### ソースディレクトリ

```
src/advanced_agent/
├── core/                  # コアシステム
├── database/              # データベース
├── config/                # 設定管理
├── reasoning/             # 推論エンジン
├── memory/                # 記憶システム
├── tools/                 # ツール群
├── learning/              # 学習システム
├── evolution/             # 進化システム
├── reward/                # 報酬システム
├── monitoring/            # 監視システム
├── optimization/          # 最適化
├── ui/                    # ユーザーインターフェース
└── api/                   # API
```

### テストディレクトリ

```
tests/
├── unit/                  # 単体テスト
├── integration/           # 統合テスト
├── e2e/                   # E2Eテスト
└── performance/           # パフォーマンステスト
```

### データディレクトリ

```
data/
├── sessions/              # セッションデータ
├── models/                # モデルファイル
├── logs/                  # ログファイル
└── backups/               # バックアップ
```

## パフォーマンス要件

### レスポンス時間

- 通常の質問: < 2 秒
- 複雑な推論: < 10 秒
- ツール実行: < 30 秒
- 進化プロセス: < 5 分

### メモリ使用量

- ベースメモリ: < 2GB
- 最大メモリ: < 5GB (RTX 4050 制約)
- 永続化データ: < 1GB
- 一時データ: < 500MB

### スループット

- 同時セッション: 10 セッション
- 1 日あたりのインタラクション: 1000 回
- 進化サイクル: 100 インタラクションごと
- ツール実行: 10 回/分

## 進捗管理

### Phase 1: 基盤システム構築

- [x] タスク 1: 永続セッション管理システム
- [ ] タスク 2: データベーススキーマ実装
- [ ] タスク 3: 設定管理システム

### Phase 2: 推論エンジン構築

- [x] タスク 4: 基本推論エンジン
- [x] タスク 5: 推論品質評価システム
- [x] タスク 6: 推論結果構造化

### Phase 3: 記憶システム構築

- [x] タスク 7: ベクトル記憶システム
- [x] タスク 8: 記憶重要度評価
- [x] タスク 9: 記憶統合システム

### Phase 4: ツールシステム構築

- [x] タスク 10: 基本ツール実装
- [x] タスク 11: MCP 統合システム
- [x] タスク 12: 動的ツール生成

### Phase 5: 学習システム構築

- [x] タスク 13: プロンプト管理システム
- [x] タスク 14: チューニングデータ管理
- [x] タスク 15: 進化システム実装

### Phase 6: 報酬システム構築

- [x] タスク 16: 報酬計算システム
- [x] タスク 17: 強化学習統合

### Phase 7: 監視システム構築

- [x] タスク 18: GPU 監視システム
- [x] タスク 19: パフォーマンス監視

### Phase 8: UI/API 構築

- [x] タスク 20: Streamlit UI 統合
- [x] タスク 21: FastAPI 統合

### Phase 9: テストシステム構築

- [x] タスク 22: Playwright MCP テスト
- [x] タスク 23: 統合テストスイート

### Phase 10: デプロイメント最適化

- [x] タスク 24: メモリ最適化
- [x] タスク 25: 推論速度最適化
- [x] タスク 26: デプロイメント設定

## 注意事項

### 技術的制約

- RTX 4050 6GB VRAM 制約
- ローカル環境での実行
- オープンソースライセンス準拠

### 開発方針

- 段階的実装（Phase 別）
- テスト駆動開発
- ドキュメント整備
- コードレビュー
- 不要ファイルの即座削除

### 品質保証

- 単体テスト（pytest）
- 統合テスト（pytest）
- E2E テスト（Playwright）
- パフォーマンステスト（locust）
- セキュリティテスト

### 参考実装の活用

- 既存のオープンソースライブラリを最大限活用
- サンプルコードのそのまま使用を推奨
- 車輪の再発明を避ける
- コミュニティのベストプラクティスに従う
