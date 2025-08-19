# ローカル自立学習 AI エージェント実装計画書

## 1. プロジェクト概要（更新版）

### 1.1 確定した要件

**開発環境**: Intel i7-13700H, 32GB RAM, RTX 4050 Laptop ✅
**開発言語**: Python
**コンテナ**: Docker 環境使用
**外部 API**: 無償サービスのみ使用
**主要目標**:

1. 自己チューニング機能優先実装
2. 最終的に Web デザインができるエージェント
3. 会話を通した自己学習
4. インターネット検索機能

## 2. 実装優先順位（更新版）

### Phase 1: 基盤構築（2 週間）

**優先度: 最高**

1. Docker 環境セットアップ
2. OLLAMA + qwen2:7b-instruct 環境構築
3. Open WebUI 導入・カスタマイズ
4. 基本的なエージェントフレームワーク

### Phase 2: 自己チューニング機能（4 週間）

**優先度: 最高**

1. 会話履歴管理システム
2. 応答品質評価機能
3. プロンプト自動最適化
4. 学習データ自動蓄積・更新

### Phase 3: エージェント基本機能（3 週間）

**優先度: 高**

1. インターネット検索機能（無償 API 使用）
2. ファイル操作機能
3. 基本的なコマンド実行機能

### Phase 4: Web デザインエージェント機能（6 週間）

**優先度: 高**

1. HTML/CSS 生成機能
2. Web デザイン知識ベース構築
3. デザインパターン学習・適用
4. レスポンシブデザイン対応

### Phase 5: 統合・最適化（3 週間）

**優先度: 中**

1. 全機能統合
2. パフォーマンス最適化
3. UI/UX 改善

## 3. 自己チューニング機能詳細設計

### 3.1 会話ベース学習システム

```
会話データ → 品質評価 → 知識抽出 → プロンプト更新 → 性能向上
    ↑                                                    ↓
    ←←←←←←←←←← フィードバックループ ←←←←←←←←←←←←
```

#### 3.1.1 会話履歴管理

- **データベース**: SQLite（軽量、ローカル）
- **保存内容**:
  - ユーザー入力
  - エージェント応答
  - 応答時間
  - ユーザーフィードバック（明示的・暗示的）
  - コンテキスト情報

#### 3.1.2 応答品質評価システム

- **自己評価機能**: LLM 自身による応答品質スコアリング
- **ユーザーフィードバック**: 👍👎、詳細コメント
- **客観的指標**: 応答時間、タスク成功率、エラー発生率
- **学習効果測定**: 同様の質問に対する応答改善度

#### 3.1.3 プロンプト自動最適化

- **A/B テスト機能**: 複数のプロンプトバリエーション自動テスト
- **成功パターン抽出**: 高評価を得たプロンプトの特徴分析
- **動的プロンプト生成**: 状況に応じた最適プロンプト自動選択

### 3.2 Web デザインエージェント特化機能

#### 3.2.1 デザイン知識ベース

- **デザイン原則**: レイアウト、色彩理論、タイポグラフィ
- **トレンド情報**: 最新の Web デザイントレンド（定期更新）
- **ベストプラクティス**: アクセシビリティ、UX 設計
- **コードテンプレート**: 再利用可能な HTML/CSS スニペット

#### 3.2.2 デザイン生成プロセス

1. **要件分析**: ユーザーの要望を構造化
2. **デザイン提案**: 複数のデザインオプション生成
3. **コード生成**: HTML/CSS/JavaScript 自動生成
4. **プレビュー機能**: リアルタイムデザイン確認
5. **反復改善**: フィードバックに基づく修正

## 4. 技術スタック詳細

### 4.1 Docker 構成

```dockerfile
# メインコンテナ構成
- ollama-container: OLLAMA + qwen2:7b-instruct
- webui-container: Open WebUI
- agent-container: Python エージェントコア
- db-container: SQLite + データ管理
- proxy-container: Nginx（リバースプロキシ）
```

### 4.2 Python 依存関係

```
# コア機能
ollama-python==0.1.7
langchain==0.1.0
langchain-community==0.0.13
streamlit==1.29.0  # 管理UI用

# Web関連
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.0
html5lib==1.1

# データベース
sqlite3 (標準ライブラリ)
sqlalchemy==2.0.23

# 検索・スクレイピング
googlesearch-python==1.2.3  # 無償Google検索
duckduckgo-search==3.9.6    # DuckDuckGo API
newspaper3k==0.2.8          # ニュース記事抽出

# Web デザイン
jinja2==3.1.2               # テンプレート生成
cssutils==2.7.1             # CSS操作
premailer==3.10.0           # CSS インライン化
```

### 4.3 無償 API 利用計画

- **検索**: DuckDuckGo API（無制限、無料）
- **バックアップ検索**: Google Custom Search（100 回/日無料）
- **Web スクレイピング**: requests + BeautifulSoup
- **デザインリソース**: Unsplash API（5000 回/時間無料）
- **フォント**: Google Fonts API（無制限）

## 5. Docker 環境セットアップ手順

### 5.1 ディレクトリ構造

```
008_LLM/
├── docker/
│   ├── docker-compose.yml
│   ├── ollama/
│   │   └── Dockerfile
│   ├── webui/
│   │   └── Dockerfile
│   ├── agent/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── nginx/
│       └── nginx.conf
├── src/
│   ├── agent/
│   │   ├── core/
│   │   ├── self_tuning/
│   │   ├── web_design/
│   │   └── tools/
│   └── data/
│       ├── conversations/
│       ├── knowledge_base/
│       └── models/
└── document/
    ├── requirements.md
    └── implementation_plan.md
```

### 5.2 docker-compose.yml 構成

```yaml
version: "3.8"

services:
  ollama:
    build: ./docker/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_MODELS=/root/.ollama/models

  webui:
    build: ./docker/webui
    ports:
      - "3000:8080"
    depends_on:
      - ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434

  agent:
    build: ./docker/agent
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
      - ./data:/app/data
    depends_on:
      - ollama
      - webui

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - webui
      - agent

volumes:
  ollama_data:
```

## 6. 開発マイルストーン

### Week 1-2: 環境構築

- [ ] Docker 環境セットアップ
- [ ] OLLAMA + qwen2:7b-instruct 動作確認
- [ ] Open WebUI 導入・カスタマイズ
- [ ] 基本的な Python エージェントフレームワーク

### Week 3-4: 自己チューニング基盤

- [ ] 会話履歴管理システム実装
- [ ] SQLite データベース設計・実装
- [ ] 基本的な品質評価機能

### Week 5-6: 自己チューニング高度化

- [ ] プロンプト自動最適化機能
- [ ] A/B テスト機能実装
- [ ] 学習効果測定システム

### Week 7-8: エージェント基本機能

- [ ] インターネット検索機能（DuckDuckGo API）
- [ ] Web スクレイピング機能
- [ ] ファイル操作機能

### Week 9-12: Web デザインエージェント

- [ ] HTML/CSS 生成エンジン
- [ ] デザイン知識ベース構築
- [ ] デザインパターン学習機能
- [ ] プレビュー機能実装

### Week 13-14: 統合・最適化

- [ ] 全機能統合テスト
- [ ] パフォーマンス最適化
- [ ] ドキュメント整備

## 7. 成功指標（更新版）

### 7.1 自己チューニング機能

- 同一質問に対する応答品質 20%向上（4 週間後）
- ユーザー満足度スコア 4.0 以上（5 点満点）
- プロンプト最適化による応答時間 10%短縮

### 7.2 Web デザインエージェント機能

- HTML/CSS 生成成功率 90%以上
- 生成された Web ページの表示エラー率 5%以下
- デザイン要求の理解精度 85%以上

### 7.3 システム全体

- 24 時間連続稼働安定性
- メモリ使用量 16GB 以内
- 応答時間平均 5 秒以内

## 8. 次のアクション

1. **Docker 環境セットアップ開始**
2. **基本ディレクトリ構造作成**
3. **OLLAMA 環境構築**
4. **Open WebUI 導入**
5. **基本エージェントフレームワーク実装**

---

**更新日**: 2024 年 12 月
**バージョン**: 1.1（ユーザー要求反映版）
**次回更新**: Phase 1 完了時
