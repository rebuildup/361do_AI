# 自己学習 AI エージェント 使用ガイド

## 概要

このガイドでは、AI エージェントシステムの自己学習機能について詳しく説明します。エージェントは会話データから自動的に学習し、プロンプトを最適化し、知識ベースを構築することで、継続的に改善されます。

## 主な機能

### 1. 自動学習システム

- **学習データ分析**: 会話から有用な学習データを自動抽出
- **プロンプト最適化**: LLM を使用したプロンプトの自動改善
- **知識抽出**: 高品質な会話から知識を自動抽出
- **パフォーマンス分析**: 学習効果の自動評価と適応

### 2. 手動学習管理

- **カスタム学習データ追加**: 手動で学習データを追加
- **知識ベース管理**: 知識アイテムの追加・編集・削除
- **プロンプト最適化**: 特定のプロンプトの手動最適化
- **学習データエクスポート/インポート**: 学習データの移行

### 3. 監視・分析

- **学習状態監視**: 学習システムの進行状況確認
- **パフォーマンスレポート**: 学習効果の詳細分析
- **統計情報**: 学習データと知識ベースの統計

## 使用方法

### 基本的な学習システムの開始

```python
# 学習システムの状態確認
result = await agent_learning_status()

# 学習システムを開始
result = await agent_start_learning()

# 手動で学習サイクルを実行
result = await agent_trigger_learning_cycle()
```

### 学習データの管理

#### カスタム学習データの追加

```python
result = await agent_add_learning_data(
    content="ユーザーがよく尋ねる質問とその回答",
    category="faq",
    tags=["よくある質問", "サポート"],
    metadata={"difficulty": "easy", "priority": "high"}
)
```

#### 学習データの取得

```python
# 全学習データを取得
result = await agent_get_learning_data()

# 特定カテゴリの高品質データを取得
result = await agent_get_learning_data(
    category="web_design",
    min_quality=0.8,
    limit=10
)
```

### 知識ベースの管理

#### 知識アイテムの追加

```python
result = await agent_add_knowledge_item(
    fact="React.jsはFacebookが開発したJavaScriptライブラリです",
    category="programming",
    confidence=0.9,
    source_context="Web開発に関する質問への回答",
    applicability="フロントエンド開発の説明"
)
```

#### 知識ベースの確認

```python
# 全知識ベースを取得
result = await agent_get_knowledge_base()

# 特定カテゴリの知識を取得
result = await agent_get_knowledge_base(category="programming")
```

### プロンプト最適化

#### 特定プロンプトの最適化

```python
result = await agent_optimize_prompt(
    prompt_key="agent_web_design_prompt",
    prompt_content="現在のプロンプト内容..."
)
```

### パフォーマンス監視

#### パフォーマンスレポートの取得

```python
# 過去7日間のレポート
result = await agent_get_performance_report(days=7)

# 過去30日間のレポート
result = await agent_get_performance_report(days=30)
```

### データのエクスポート/インポート

#### 学習データのエクスポート

```python
result = await agent_export_learning_data(format="json")
```

## CLI からの操作

### 1. 学習システムの制御

```bash
# 学習システムを開始
> learn start

# 学習システムを停止
> learn stop

# 学習状態を確認
> learn status

# 手動で学習サイクルを実行
> learn cycle
```

### 2. 学習データの管理

```bash
# 学習データを追加
> data add <カテゴリ> <内容>

# 学習データを一覧表示
> data list [カテゴリ]

# 統計情報を表示
> data stats
```

### 3. システム状態の確認

```bash
# システム全体の状態確認
> status

# 統計情報の表示
> stats
```

## 設定オプション

### 学習設定の調整

`src/agent/core/config.py` の `LearningConfig` クラスで以下の設定を調整できます：

```python
@dataclass
class LearningConfig:
    # 品質評価設定
    auto_evaluation_enabled: bool = True
    min_quality_score_for_learning: float = 0.7
    max_conversations_per_learning_cycle: int = 100

    # プロンプト最適化設定
    prompt_optimization_enabled: bool = True
    ab_test_duration_hours: int = 24
    min_samples_for_statistical_significance: int = 30

    # 知識抽出設定
    knowledge_extraction_enabled: bool = True
    knowledge_confidence_threshold: float = 0.8
    max_knowledge_items_per_category: int = 1000

    # セーフガード設定
    safety_check_enabled: bool = True
    safety_threshold: float = 0.8
    harmful_content_detection: bool = True
```

### 環境変数での設定

```bash
# 学習機能の有効/無効
AGENT_LEARNING_ENABLED=true

# 学習間隔（分）
AGENT_LEARNING_INTERVAL_MINUTES=30

# 品質閾値
AGENT_QUALITY_THRESHOLD=0.8
```

## 学習プロセスの詳細

### 1. 学習データ分析ループ

- **実行間隔**: 設定された学習間隔（デフォルト 30 分）
- **処理内容**:
  - 高品質な会話（品質スコア 0.7 以上）から学習データを抽出
  - 既存の学習データの品質改善
  - 使用されていない学習データの削除

### 2. プロンプト最適化ループ

- **実行間隔**: 学習間隔の 2 倍（デフォルト 60 分）
- **処理内容**:
  - カスタムプロンプトファイルの読み込み
  - LLM を使用したプロンプト最適化
  - 改善効果の評価（10%以上の改善がある場合のみ適用）
  - バックアップ作成とプロンプト更新

### 3. 知識抽出ループ

- **実行間隔**: 学習間隔の 3 倍（デフォルト 90 分）
- **処理内容**:
  - 高品質な会話（品質スコア 0.8 以上）から知識を抽出
  - 重複する知識の統合
  - 低信頼度の知識の削除

### 4. パフォーマンス分析ループ

- **実行間隔**: 学習間隔の 4 倍（デフォルト 120 分）
- **処理内容**:
  - パフォーマンス指標の取得
  - 学習効果の傾向分析
  - 学習パラメータの適応的調整
  - 学習レポートの生成

## ベストプラクティス

### 1. 学習データの品質管理

- 高品質な会話のみを学習データとして使用
- 定期的に学習データの品質を評価
- 使用されていないデータは自動削除

### 2. プロンプト最適化の活用

- 重要なプロンプトは定期的に最適化
- 改善効果を確認してから適用
- バックアップを活用して安全に更新

### 3. 知識ベースの維持

- 信頼度の高い知識のみを保持
- 重複する知識は統合
- 定期的に知識ベースを整理

### 4. パフォーマンス監視

- 定期的にパフォーマンスレポートを確認
- 学習効果の傾向を分析
- 必要に応じて学習パラメータを調整

## トラブルシューティング

### よくある問題と解決方法

#### 1. 学習システムが開始されない

- **原因**: 設定で学習機能が無効になっている
- **解決**: `AGENT_LEARNING_ENABLED=true` を設定

#### 2. プロンプト最適化が実行されない

- **原因**: プロンプトファイルが見つからない、または権限問題
- **解決**: `data/prompts/` ディレクトリの存在と権限を確認

#### 3. 学習データが保存されない

- **原因**: データベースの権限問題
- **解決**: `data/agent.db` ファイルの書き込み権限を確認

#### 4. パフォーマンスが低下

- **原因**: 学習データが多すぎる、または OLLAMA の負荷
- **解決**: 学習データの品質閾値を上げるか、古いデータを削除

## 高度な使用方法

### カスタム学習アルゴリズムの実装

新しい学習アルゴリズムを追加する場合は、`src/agent/self_tuning/advanced_learning.py` を拡張してください：

```python
class CustomLearningAlgorithm:
    async def process(self, conversation_data):
        # カスタム学習ロジックを実装
        pass
```

### 学習データの外部連携

外部システムと学習データを連携する場合は、エクスポート/インポート機能を活用してください：

```python
# 学習データをエクスポート
export_result = await agent_export_learning_data()

# 外部システムで処理後、インポート
import_result = await agent_import_learning_data(processed_data)
```

## まとめ

この自己学習システムにより、AI エージェントは継続的に改善され、より良い応答を提供できるようになります。適切な設定と監視により、効果的な学習サイクルを実現してください。
