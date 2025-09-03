# 実装計画

## 概要

RTX 4050 6GB VRAM 環境で動作する高性能自己学習 AI エージェントの実装を、**オープンソースライブラリを最大限活用**して段階的かつテスト駆動で進めます。LangChain、AutoGen、HuggingFace、ChromaDB などの成熟したフレームワークを統合し、オリジナルコードを最小限に抑えて開発効率と安定性を向上させます。

## 実装タスク

- [x] 0. プロジェクト構造の整理とクリーンアップ

  - 既存の複雑なエージェント実装を削除し、新しいプロジェクト構造を構築
  - 不要なファイルとフォルダを削除し、必要な設定ファイルのみを保持
  - _要件: 全体的なシステム整理_

- [x] 0.1 既存ファイルの整理と削除

  - src/agent、src/codex_agent、systems フォルダ内の既存エージェント実装を削除
  - examples、tools、scripts フォルダ内の不要なファイルを削除
  - logs、results フォルダ内の古いログファイルを削除
  - _要件: プロジェクト構造の簡素化_

- [x] 0.2 新しいプロジェクト構造の作成

  - src/advanced_agent フォルダを作成し、新しいエージェントの基盤構造を構築
  - config、tests、docs フォルダを整理し、新システム用の設定を準備
  - requirements.txt を更新し、必要なライブラリのみを含める新しい requirements.txt を作成
  - _要件: 新システム用の基盤準備_

- [x] 0.3 設定ファイルとドキュメントの整理

  - 既存の設定ファイル（config/_.env, config/_.json）を新システム用に更新
  - 古いドキュメント（docs/\*.md）を削除し、新システム用のドキュメント構造を準備
  - README.md を更新し、新しいプロジェクトの概要を記載
  - _要件: プロジェクト情報の更新_

- [x] 1. オープンソース基盤システムの統合

  - PSUtil + NVIDIA-ML による GPU/CPU 監視システムを統合
  - HuggingFace Accelerate + BitsAndBytes による自動量子化を実装
  - LangChain + Ollama クライアント統合を構築
  - _要件: 1.1, 1.4, 4.1, 4.2_

- [x] 1.1 PSUtil + NVIDIA-ML システム監視の統合

  - PSUtil による CPU/メモリ監視機能を統合
  - NVIDIA-ML-Py による GPU 統計取得機能を統合
  - Prometheus Client によるメトリクス収集システムを構築
  - 既存ライブラリの設定ファイルベース統合（オリジナルコード最小化）
  - _要件: 1.4, 4.1, 4.2_

- [x] 1.2 Pydantic + Loguru 設定・ログシステムの統合

  - Pydantic Settings による YAML/環境変数統合設定管理を実装
  - Loguru による構造化ログシステムを統合
  - 設定ファイルによるシステム起動時環境検証を実装（コード最小化）
  - _要件: 7.4, 4.3_

- [x] 1.3 LangChain + Ollama 統合クライアントの実装

  - LangChain Community Ollama 統合を使用してクライアント接続を構築
  - Ollama Python SDK による DeepSeek-R1 モデル通信を実装
  - LangChain のフォールバック機能を使用した代替モデル切り替えを実装
  - _要件: 1.1, 7.2_

- [x] 2. LangChain ReAct Agent + Ollama 推論システムの統合

  - LangChain ReAct Agent による Chain-of-Thought 推論を統合
  - HuggingFace BitsAndBytes による自動量子化を統合
  - LangChain Tools による推論結果構造化を実装
  - _要件: 1.1, 1.2, 1.3_

- [x] 2.1 LangChain + Ollama 基本推論エンジンの統合

  - LangChain Community Ollama LLM クラスを使用した基本推論機能を構築
  - LangChain PromptTemplate による プロンプト管理を統合
  - LangChain Callbacks による推論時間・メモリ測定を実装
  - _要件: 1.1, 1.3, 5.1_

- [x] 2.2 LangChain ReAct Agent Chain-of-Thought の統合

  - LangChain create_react_agent による段階的思考プロセスを統合
  - LangChain AgentExecutor による中間ステップ構造化を実装
  - LangChain の既存評価機能による推論品質評価を統合
  - _要件: 1.2, 1.3, 1.5_

- [x] 2.3 HuggingFace BitsAndBytes 動的量子化の統合

  - BitsAndBytesConfig による自動量子化設定を統合
  - HuggingFace Accelerate による段階的量子化を実装
  - Transformers AutoModel による量子化品質監視を統合
  - _要件: 1.4, 1.5, 4.2_

- [x] 3. HuggingFace PEFT LoRA システムの統合

  - HuggingFace PEFT による LoRA アダプタ管理を統合
  - PEFT LoraConfig による Parameter-Efficient Fine-Tuning を実装
  - HuggingFace Hub による アダプタ共有・管理を統合
  - _要件: 2.3, 7.1, 7.3_

- [x] 3.1 HuggingFace PEFT アダプタプールの統合

  - PEFT get_peft_model による複数アダプタ管理を統合
  - HuggingFace Hub による アダプタ性能スコア追跡を実装
  - PEFT の既存スワッピング機能を使用したメモリ効率化を統合
  - _要件: 2.3, 5.3, 7.1_

- [x] 3.2 PEFT + BitsAndBytes QLoRA システムの統合

  - BitsAndBytesConfig + LoraConfig による QLoRA 統合を実装
  - HuggingFace Trainer による 4GB 制限学習パイプラインを構築
  - Transformers TrainerCallback による学習監視を統合
  - _要件: 2.3, 4.1, 4.2_

- [x] 3.3 HuggingFace Evaluate アダプタ評価の統合

  - HuggingFace Evaluate による タスク別性能指標計算を統合
  - Datasets ライブラリによる A/B テスト比較システムを構築
  - PEFT の既存機能による性能劣化検出・ロールバックを統合
  - _要件: 2.5, 4.5, 7.1_

- [x] 4. AutoGen マルチエージェント進化学習システムの統合

  - AutoGen AssistantAgent による モデル交配・自然選択を統合
  - AutoGen GroupChat による 世代管理・性能追跡を実装
  - HuggingFace Datasets による 合成データ生成を統合
  - _要件: 2.1, 2.2, 2.5_

- [x] 4.1 AutoGen 進化的エージェント群の統合

  - AutoGen AssistantAgent による 進化プロセス管理を統合
  - AutoGen GroupChatManager による 選択・交配・変異オペレータを実装
  - AutoGen の既存会話履歴機能による 世代管理・系譜追跡を統合
  - _要件: 2.1, 2.2, 2.5_

- [x] 4.2 PEFT + AutoGen アダプタ交配システムの統合

  - PEFT の既存マージ機能による LoRA アダプタ重み交配を統合
  - AutoGen エージェント による 交配確率・変異率動的調整を実装
  - HuggingFace Evaluate による 交配結果品質評価を統合
  - _要件: 2.1, 2.2, 2.5_

- [x] 4.3 HuggingFace Datasets 合成データ生成の統合

  - HuggingFace Datasets による 自動データ生成パイプラインを統合
  - Transformers Pipeline による 生成データ品質フィルタリングを実装
  - Datasets の既存機能による ドメイン特化データ管理を統合
  - _要件: 2.4, 2.5_

- [x] 5. LangChain + ChromaDB 永続的記憶システムの統合

  - LangChain Memory による セッション永続化・自動記憶管理を統合
  - ChromaDB + SQLAlchemy による 重要度ベース記憶選択を実装
  - Sentence-Transformers による 長期記憶化システムを統合
  - _要件: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 5.1 LangChain + ChromaDB 永続的記憶管理の統合

  - LangChain ConversationSummaryBufferMemory による 会話履歴永続化を統合
  - ChromaDB による 効率的ベクトル記憶保存・検索システムを統合
  - LangChain の既存セッション機能による コンテキスト継続を実装
  - _要件: 6.1, 6.5_

- [x] 5.2 LangChain Memory + HuggingFace 記憶システムの統合

  - LangChain の既存 Memory クラスによる 短期・長期記憶管理を統合
  - HuggingFace Transformers Pipeline による 重要度評価・自動分類を実装
  - ChromaDB の既存機能による 記憶容量制限・自動整理を統合
  - _要件: 6.2, 6.4_

- [x] 5.3 ChromaDB + Sentence-Transformers 記憶検索の統合

  - Sentence-Transformers による 重要度判定・類似度計算を統合
  - ChromaDB similarity_search による 類似記憶検索を統合
  - LangChain の既存パターン認識機能による 経験学習を実装
  - _要件: 6.2, 6.3_

- [x] 5.4 SQLAlchemy + LangChain セッション管理の統合

  - SQLAlchemy ORM による セッション復元・ユーザー管理を統合
  - LangChain の既存セッション機能による 完全復元を実装
  - SQLAlchemy の既存整合性機能による 自動修復を統合
  - _要件: 6.1, 6.5_

- [x] 6. HuggingFace Transformers マルチモーダル処理の統合

  - HuggingFace Pipeline による テキスト・コード・画像統合処理を統合
  - HuggingFace Accelerate による GPU/CPU ハイブリッド処理を実装
  - Transformers AutoProcessor による マルチモーダル統合を統合
  - _要件: 3.1, 3.2, 3.4_

- [x] 6.1 HuggingFace Accelerate ハイブリッド処理の統合

  - HuggingFace Accelerate による GPU/CPU 負荷分散を統合
  - Accelerate の既存並列処理機能による パイプライン・同期を実装
  - Accelerate の既存分割機能による 自動タスク分割を統合
  - _要件: 3.4, 4.1, 4.4_

- [x] 6.2 HuggingFace Code Generation Pipeline の統合

  - Transformers CodeGen Pipeline による コード生成機能を統合
  - HuggingFace の既存構文チェック機能による エラー修正を実装
  - Transformers の既存評価機能による コード品質評価を統合
  - _要件: 3.2, 3.5_

- [x] 6.3 HuggingFace Document AI Pipeline の統合

  - Transformers Document AI による 構造化情報抽出を統合
  - HuggingFace の既存統合機能による マルチモーダル結果統合を実装
  - Transformers の既存信頼度機能による スコア付き出力を統合
  - _要件: 3.3, 3.5_

- [x] 7. Prometheus + Grafana 監視・最適化システムの統合

  - Prometheus Client による システム性能継続監視を統合
  - Grafana による 異常検出・自動復旧ダッシュボードを構築
  - PSUtil + NVIDIA-ML による リアルタイム最適化を統合
  - _要件: 4.1, 4.2, 4.3, 4.4_

- [x] 7.1 Prometheus + PSUtil + NVIDIA-ML 監視の統合

  - Prometheus Client による GPU/CPU/メモリ メトリクス収集を統合
  - PSUtil + NVIDIA-ML による リアルタイム統計取得を統合
  - Prometheus の既存アラート機能による ボトルネック検出を実装
  - _要件: 4.1, 4.2, 4.5_

- [x] 7.2 HuggingFace + Prometheus 自動最適化の統合

  - HuggingFace の既存最適化機能による パラメータ自動調整を統合
  - Prometheus メトリクスによる 動的リソース配分を実装
  - Prometheus の既存履歴機能による 最適化学習を統合
  - _要件: 4.2, 4.4, 4.5_

- [x] 7.3 Prometheus + Grafana 異常検出・復旧の統合

  - Prometheus の既存パターン検出による エラー分類を統合
  - Grafana アラート による 段階的復旧戦略実行を実装
  - Prometheus の既存フェイルセーフ機能による 復旧失敗対応を統合
  - _要件: 4.3, 4.4_

- [x] 8. FastAPI + Streamlit + Typer インターフェースの統合

  - FastAPI による 高応答性 REST API を統合
  - Streamlit による リアルタイム Web UI を構築
  - Typer による CLI インターフェースを統合
  - _要件: 5.1, 5.2, 5.4, 5.5_

- [x] 8.1 FastAPI REST API ゲートウェイの統合

  - FastAPI の既存高性能機能による API サーバーを統合
  - FastAPI の既存 OpenAI 互換機能による エンドポイント・認証を実装
  - FastAPI の既存制限機能による レート制限・監視を統合
  - _要件: 5.1, 7.2_

- [x] 8.2 Streamlit リアルタイム Web UI の統合

  - Streamlit の既存応答性機能による フロントエンドを統合
  - Streamlit の既存可視化機能による 進捗・VRAM 表示を実装
  - Streamlit の既存セッション機能による 履歴管理・継続を統合
  - _要件: 5.1, 5.2, 5.5_

- [x] 8.3 Pydantic + Streamlit 設定管理の統合

  - Pydantic Settings による 動的設定変更・反映を統合
  - Streamlit の既存選択機能による モデル選択・切り替えを実装
  - Streamlit の既存ファイル機能による バックアップ・復元 UI を統合
  - _要件: 5.4, 7.4, 7.5_

- [x] 9. Pytest + HuggingFace Evaluate 統合テスト・最適化

  - Pytest による エンドツーエンドテスト・性能最適化を統合
  - HuggingFace Evaluate による 本番環境安定性確保を実装
  - 既存ドキュメントツールによる ドキュメント整備を統合
  - _要件: 全要件の統合検証_

- [x] 9.1 Pytest + LangChain 統合テストスイートの統合

  - Pytest の既存機能による エンドツーエンド推論テスト・全連携検証を統合
  - Pytest + PSUtil による メモリ圧迫安定性テストを実装
  - Pytest の既存長時間テスト機能による 性能劣化検出を統合
  - Pytest + SQLAlchemy による セッション永続化・記憶整合性テストを統合
  - _要件: 全要件の統合検証_

- [x] 9.2 HuggingFace Evaluate + Prometheus 性能ベンチマークの統合

  - HuggingFace Evaluate による 推論速度・メモリ効率・学習効果評価を統合
  - ChromaDB + Evaluate による 記憶検索性能・精度評価を実装
  - HuggingFace の既存比較機能による 他システム性能比較を統合
  - Grafana + Evaluate による 最適化効果可視化・レポート生成を統合
  - _要件: 4.5, 5.1, 5.2, 6.3_

- [x] 9.3 既存ツール本番環境準備・ドキュメント統合

  - Docker + Docker Compose による デプロイメント自動化を統合
  - MkDocs による ユーザーマニュナル・開発者ドキュメント作成を統合
  - 既存ドキュメントツールによる トラブルシューティング・FAQ 整備を統合
  - SQLAlchemy + ChromaDB の既存機能による バックアップ・復元手順文書化を統合
  - _要件: 7.4, 7.5_

- [x] 10. 実際の AI 推論機能の確立

  - Mock 応答を実際の Ollama 接続に置き換え (qwen2:7b-instruct モデル使用)
  - Chain-of-Thought 推論の実装と検証 (ReAct Agent スタイル)
  - エラーハンドリングとフォールバック機能の強化 (複数モデル対応)
  - 接続テストとデバッグ用ツールの実装
  - _要件: 1.1, 1.2, 1.3, 5.1_

- [x] 10.1 Mock 応答を実際の Ollama 接続に置き換え

  - streamlit_ui.py の \_call_chat_api メソッドで実際の Ollama API 統合を確認
  - 利用可能なモデル (qwen2:7b-instruct) での接続テストを実装
  - Ollama 接続エラー時の詳細なエラーハンドリングと復旧手順を追加
  - 簡単なテスト用 Streamlit アプリ (streamlit_simple_test.py) を作成
  - _要件: 1.1, 5.1_

- [x] 10.2 Chain-of-Thought 推論の実装

  - LangChain ReAct Agent による段階的思考プロセスを統合
  - 推論ステップの可視化と構造化出力を実装
  - 推論品質評価と信頼度スコア計算を追加
  - _要件: 1.2, 1.3, 1.5_

- [x] 10.3 エラーハンドリングとフォールバック機能の強化

  - Ollama 接続失敗時の自動フォールバック機能を実装 (複数モデル対応)
  - GPU メモリ不足時の軽量モデル切り替え処理を追加
  - ユーザーフレンドリーなエラーメッセージと詳細な復旧手順を実装
  - 接続テストスクリプト (test_ollama_connection.py) による診断機能を追加
  - _要件: 1.4, 4.2, 4.3_
