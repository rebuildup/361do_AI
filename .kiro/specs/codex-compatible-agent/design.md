# 設計書

## 概要

現在の複雑な自己学習エージェント実装を、OpenAI Codex のシンプルで安定したエージェント機能に置き換えます。OLLAMA をバックエンドとして使用し、Codex 互換の API インターフェースを提供することで、既存の Codex ベースのツールやアプリケーションをそのまま利用できるようにします。

## アーキテクチャ

### システム構成

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Codex Agent   │───▶│ Compatibility   │───▶│   OLLAMA        │
│   Interface     │    │ Layer           │    │   Backend       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Existing Codex  │    │ API Translation │    │ Local LLM       │
│ Applications    │    │ & Error Handling│    │ Processing      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 主要コンポーネント

1. **Codex Agent Interface** - Codex エージェント機能のインターフェース
2. **Compatibility Layer** - Codex と OLLAMA 間の互換性レイヤー
3. **OLLAMA Backend** - ローカル LLM バックエンド
4. **Configuration Manager** - 最小限の設定管理

## コンポーネントと インターフェース

### 1. CodexAgentInterface

```python
class CodexAgentInterface:
    """Codex互換エージェントインターフェース"""

    def __init__(self, compatibility_layer: CompatibilityLayer):
        self.compatibility_layer = compatibility_layer

    async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """コード補完リクエスト処理"""
        pass

    async def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """チャットリクエスト処理"""
        pass
```

### 2. CompatibilityLayer

```python
class CompatibilityLayer:
    """CodexとOLLAMA間の互換性レイヤー"""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client

    async def translate_codex_to_ollama(self, request: Dict) -> Dict:
        """CodexリクエストをOLLAMA形式に変換"""
        pass

    async def translate_ollama_to_codex(self, response: Dict) -> Dict:
        """OLLAMAレスポンスをCodex形式に変換"""
        pass
```

### 3. SimpleOllamaClient

```python
class SimpleOllamaClient:
    """シンプルなOLLAMAクライアント"""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """テキスト生成"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        pass
```

### 4. SimpleConfig

```python
class SimpleConfig:
    """最小限の設定管理"""

    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2:7b-instruct")
        self.default_max_tokens = 1000
```

## データモデル

### リクエスト形式

```python
@dataclass
class CodexRequest:
    """Codex互換リクエスト"""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
```

### レスポンス形式

```python
@dataclass
class CodexResponse:
    """Codex互換レスポンス"""
    id: str
    object: str = "text_completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
```

## エラーハンドリング

### エラー分類

1. **接続エラー** - OLLAMA サービスへの接続失敗
2. **変換エラー** - リクエスト/レスポンス変換失敗
3. **設定エラー** - 不正な設定値
4. **モデルエラー** - 指定されたモデルが利用不可

### エラーレスポンス形式

```python
@dataclass
class ErrorResponse:
    """エラーレスポンス"""
    error: Dict[str, Any]

    @classmethod
    def create(cls, error_type: str, message: str, code: int = 500):
        return cls(error={
            "type": error_type,
            "message": message,
            "code": code
        })
```

## テスト戦略

### 単体テスト

1. **CompatibilityLayer** - リクエスト/レスポンス変換のテスト
2. **SimpleOllamaClient** - OLLAMA API 呼び出しのテスト
3. **CodexAgentInterface** - エージェントインターフェースのテスト
4. **SimpleConfig** - 設定管理のテスト

### 統合テスト

1. **エンドツーエンド** - Codex リクエストから最終レスポンスまでの流れ
2. **エラーハンドリング** - 各種エラーケースの処理
3. **設定変更** - 異なる設定での動作確認

### テストデータ

```python
TEST_CASES = [
    {
        "name": "simple_completion",
        "request": {"prompt": "def hello_world():", "max_tokens": 50},
        "expected_type": "code_completion"
    },
    {
        "name": "chat_message",
        "request": {"messages": [{"role": "user", "content": "Hello"}]},
        "expected_type": "chat_response"
    }
]
```

## 実装計画

### フェーズ 1: 基盤構築

- SimpleConfig 実装
- SimpleOllamaClient 実装
- 基本的なエラーハンドリング

### フェーズ 2: 互換性レイヤー

- CompatibilityLayer 実装
- リクエスト/レスポンス変換機能
- 基本的なテスト

### フェーズ 3: エージェントインターフェース

- CodexAgentInterface 実装
- 既存システムとの統合
- 包括的なテスト

### フェーズ 4: 置き換えと検証

- 既存の複雑なエージェント機能を無効化
- 新しい Codex 互換エージェントに切り替え
- 動作検証とパフォーマンステスト
