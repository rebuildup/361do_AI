# 自己チューニング機能詳細設計書

## 1. 概要

### 1.1 目的

エージェントが会話を通して自動的に学習し、応答品質を継続的に改善する自己チューニング機能を実装する。ユーザーとの対話から知識を抽出し、プロンプトを最適化し、より良い応答を生成できるようになることを目指す。

### 1.2 設計原則

- **継続的学習**: 全ての会話から学習機会を抽出
- **自動化**: 人間の介入なしに改善を実行
- **透明性**: 学習プロセスをユーザーに可視化
- **安全性**: 有害な学習を防ぐセーフガード実装

## 2. システムアーキテクチャ

### 2.1 コンポーネント構成

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Conversation   │───▶│  Quality        │───▶│  Knowledge      │
│  Manager        │    │  Evaluator      │    │  Extractor      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Conversation   │    │  Performance    │    │  Knowledge      │
│  Database       │    │  Metrics        │    │  Base           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  Prompt         │
                    │  Optimizer      │
                    └─────────────────┘
```

### 2.2 データフロー

1. **会話収集**: ユーザーとの全対話を記録
2. **品質評価**: 各応答の品質を自動評価
3. **知識抽出**: 成功パターンから知識を抽出
4. **プロンプト最適化**: 抽出した知識でプロンプトを改善
5. **性能測定**: 改善効果を継続的に測定

## 3. データベース設計

### 3.1 テーブル構成

#### conversations テーブル

```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_input TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    response_time REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    context_data JSON,
    user_feedback INTEGER DEFAULT NULL,  -- -1: 悪い, 0: 普通, 1: 良い
    feedback_comment TEXT DEFAULT NULL,
    quality_score REAL DEFAULT NULL,
    improvement_applied BOOLEAN DEFAULT FALSE
);
```

#### quality_metrics テーブル

```sql
CREATE TABLE quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations(id),
    relevance_score REAL NOT NULL,      -- 関連性スコア (0-1)
    accuracy_score REAL NOT NULL,       -- 正確性スコア (0-1)
    helpfulness_score REAL NOT NULL,    -- 有用性スコア (0-1)
    clarity_score REAL NOT NULL,        -- 明確性スコア (0-1)
    overall_score REAL NOT NULL,        -- 総合スコア (0-1)
    evaluation_method TEXT NOT NULL,    -- 'auto', 'user', 'hybrid'
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### knowledge_base テーブル

```sql
CREATE TABLE knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,              -- 'web_design', 'general', 'technical'
    knowledge_type TEXT NOT NULL,        -- 'pattern', 'fact', 'procedure'
    content TEXT NOT NULL,
    confidence_score REAL NOT NULL,      -- 信頼度 (0-1)
    source_conversations JSON,           -- 元となった会話のID配列
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE
);
```

#### prompt_templates テーブル

```sql
CREATE TABLE prompt_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_name TEXT UNIQUE NOT NULL,
    template_content TEXT NOT NULL,
    category TEXT NOT NULL,              -- 'system', 'user', 'web_design'
    version INTEGER NOT NULL DEFAULT 1,
    performance_score REAL DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE
);
```

#### learning_history テーブル

```sql
CREATE TABLE learning_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learning_type TEXT NOT NULL,         -- 'prompt_update', 'knowledge_addition'
    description TEXT NOT NULL,
    before_state JSON,
    after_state JSON,
    performance_impact REAL DEFAULT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 4. コア機能実装

### 4.1 会話管理システム (ConversationManager)

```python
class ConversationManager:
    """会話の記録と管理を行うクラス"""

    def __init__(self, db_connection):
        self.db = db_connection
        self.current_session = None

    async def start_session(self) -> str:
        """新しい会話セッションを開始"""
        session_id = str(uuid.uuid4())
        self.current_session = session_id
        return session_id

    async def record_conversation(
        self,
        user_input: str,
        agent_response: str,
        response_time: float,
        context_data: dict = None
    ) -> int:
        """会話を記録"""
        conversation_id = await self.db.insert_conversation(
            session_id=self.current_session,
            user_input=user_input,
            agent_response=agent_response,
            response_time=response_time,
            context_data=context_data
        )

        # 非同期で品質評価を開始
        asyncio.create_task(
            self._evaluate_response_quality(conversation_id)
        )

        return conversation_id

    async def record_user_feedback(
        self,
        conversation_id: int,
        feedback_score: int,
        feedback_comment: str = None
    ):
        """ユーザーフィードバックを記録"""
        await self.db.update_conversation_feedback(
            conversation_id, feedback_score, feedback_comment
        )

        # フィードバックに基づく学習を開始
        asyncio.create_task(
            self._learn_from_feedback(conversation_id)
        )
```

### 4.2 品質評価システム (QualityEvaluator)

```python
class QualityEvaluator:
    """応答品質を自動評価するクラス"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.evaluation_prompt = self._load_evaluation_prompt()

    async def evaluate_response(
        self,
        user_input: str,
        agent_response: str,
        context: dict = None
    ) -> QualityMetrics:
        """応答品質を評価"""

        evaluation_request = {
            "user_input": user_input,
            "agent_response": agent_response,
            "context": context or {}
        }

        # LLMを使用した品質評価
        evaluation_result = await self._llm_evaluate(evaluation_request)

        # 客観的指標の計算
        objective_metrics = self._calculate_objective_metrics(
            user_input, agent_response
        )

        # 総合スコア計算
        overall_score = self._calculate_overall_score(
            evaluation_result, objective_metrics
        )

        return QualityMetrics(
            relevance_score=evaluation_result['relevance'],
            accuracy_score=evaluation_result['accuracy'],
            helpfulness_score=evaluation_result['helpfulness'],
            clarity_score=evaluation_result['clarity'],
            overall_score=overall_score,
            evaluation_method='auto'
        )

    async def _llm_evaluate(self, request: dict) -> dict:
        """LLMを使用した品質評価"""
        prompt = self.evaluation_prompt.format(**request)

        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.1  # 一貫性のため低温度
        )

        # JSONレスポンスをパース
        return self._parse_evaluation_response(response)

    def _calculate_objective_metrics(
        self, user_input: str, agent_response: str
    ) -> dict:
        """客観的指標の計算"""
        return {
            "response_length": len(agent_response),
            "response_complexity": self._calculate_complexity(agent_response),
            "relevance_keywords": self._check_keyword_relevance(
                user_input, agent_response
            )
        }
```

### 4.3 知識抽出システム (KnowledgeExtractor)

```python
class KnowledgeExtractor:
    """成功パターンから知識を抽出するクラス"""

    def __init__(self, llm_client, db_connection):
        self.llm = llm_client
        self.db = db_connection
        self.extraction_threshold = 0.8  # 高品質な会話のみから抽出

    async def extract_knowledge_from_conversations(
        self,
        min_quality_score: float = 0.8,
        limit: int = 100
    ):
        """高品質な会話から知識を抽出"""

        # 高品質な会話を取得
        high_quality_conversations = await self.db.get_conversations_by_quality(
            min_score=min_quality_score,
            limit=limit
        )

        extracted_knowledge = []

        for conv in high_quality_conversations:
            # パターン抽出
            patterns = await self._extract_patterns(conv)

            # 事実抽出
            facts = await self._extract_facts(conv)

            # 手順抽出
            procedures = await self._extract_procedures(conv)

            extracted_knowledge.extend(patterns + facts + procedures)

        # 知識ベースに保存
        for knowledge in extracted_knowledge:
            await self._save_knowledge(knowledge)

        return len(extracted_knowledge)

    async def _extract_patterns(self, conversation: dict) -> list:
        """成功パターンを抽出"""
        extraction_prompt = f"""
        以下の会話から、成功したパターンを抽出してください。
        特に、ユーザーの質問タイプと効果的な回答構造に注目してください。

        ユーザー入力: {conversation['user_input']}
        エージェント応答: {conversation['agent_response']}
        品質スコア: {conversation['quality_score']}

        抽出すべき要素:
        1. 質問のカテゴリ
        2. 効果的だった回答の構造
        3. 使用された専門用語や説明方法
        4. ユーザーが満足した要因

        JSON形式で回答してください。
        """

        response = await self.llm.generate(
            prompt=extraction_prompt,
            max_tokens=800,
            temperature=0.2
        )

        return self._parse_extraction_response(response, 'pattern')
```

### 4.4 プロンプト最適化システム (PromptOptimizer)

```python
class PromptOptimizer:
    """プロンプトを自動最適化するクラス"""

    def __init__(self, llm_client, db_connection):
        self.llm = llm_client
        self.db = db_connection
        self.ab_test_manager = ABTestManager(db_connection)

    async def optimize_prompts(self):
        """プロンプトの最適化を実行"""

        # 現在のプロンプト性能を分析
        current_performance = await self._analyze_current_performance()

        # 改善候補を生成
        improvement_candidates = await self._generate_improvements()

        # A/Bテストを設定
        for candidate in improvement_candidates:
            await self.ab_test_manager.setup_test(
                original_prompt=candidate['original'],
                new_prompt=candidate['improved'],
                test_duration_hours=24
            )

        return len(improvement_candidates)

    async def _generate_improvements(self) -> list:
        """プロンプト改善案を生成"""

        # 最近の低品質な応答を分析
        poor_responses = await self.db.get_conversations_by_quality(
            max_score=0.5, limit=50
        )

        # 高品質な応答と比較
        good_responses = await self.db.get_conversations_by_quality(
            min_score=0.8, limit=50
        )

        improvement_prompt = f"""
        以下の情報を基に、プロンプトの改善案を提案してください。

        低品質な応答の特徴:
        {self._summarize_responses(poor_responses)}

        高品質な応答の特徴:
        {self._summarize_responses(good_responses)}

        現在のシステムプロンプト:
        {await self._get_current_system_prompt()}

        改善案を3つ提案し、それぞれの理由も説明してください。
        """

        response = await self.llm.generate(
            prompt=improvement_prompt,
            max_tokens=1200,
            temperature=0.3
        )

        return self._parse_improvement_suggestions(response)
```

### 4.5 A/B テスト管理システム (ABTestManager)

```python
class ABTestManager:
    """プロンプトのA/Bテストを管理するクラス"""

    def __init__(self, db_connection):
        self.db = db_connection
        self.active_tests = {}

    async def setup_test(
        self,
        original_prompt: str,
        new_prompt: str,
        test_duration_hours: int = 24
    ) -> str:
        """新しいA/Bテストを設定"""
        test_id = str(uuid.uuid4())

        test_config = {
            'test_id': test_id,
            'original_prompt': original_prompt,
            'new_prompt': new_prompt,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=test_duration_hours),
            'original_responses': [],
            'new_responses': [],
            'status': 'active'
        }

        await self.db.save_ab_test(test_config)
        self.active_tests[test_id] = test_config

        return test_id

    async def get_prompt_for_request(self, base_prompt: str) -> tuple:
        """リクエストに使用するプロンプトを決定（A/Bテスト考慮）"""

        # アクティブなテストがあるかチェック
        for test_id, test_config in self.active_tests.items():
            if test_config['original_prompt'] in base_prompt:
                # 50/50でプロンプトを選択
                if random.random() < 0.5:
                    return test_config['original_prompt'], test_id, 'A'
                else:
                    return test_config['new_prompt'], test_id, 'B'

        return base_prompt, None, None

    async def record_test_result(
        self,
        test_id: str,
        variant: str,
        conversation_id: int,
        quality_score: float
    ):
        """A/Bテストの結果を記録"""
        if test_id in self.active_tests:
            test_config = self.active_tests[test_id]

            result = {
                'conversation_id': conversation_id,
                'quality_score': quality_score,
                'timestamp': datetime.now()
            }

            if variant == 'A':
                test_config['original_responses'].append(result)
            else:
                test_config['new_responses'].append(result)

            await self.db.update_ab_test(test_id, test_config)

    async def analyze_completed_tests(self):
        """完了したテストを分析"""
        completed_tests = await self.db.get_completed_ab_tests()

        for test in completed_tests:
            analysis = self._statistical_analysis(test)

            if analysis['significant_improvement']:
                # 新しいプロンプトを採用
                await self._adopt_new_prompt(test)

            await self.db.mark_test_analyzed(test['test_id'])
```

## 5. 学習サイクル実装

### 5.1 継続的学習ループ

```python
class ContinuousLearningEngine:
    """継続的学習エンジン"""

    def __init__(self, components: dict):
        self.conversation_manager = components['conversation_manager']
        self.quality_evaluator = components['quality_evaluator']
        self.knowledge_extractor = components['knowledge_extractor']
        self.prompt_optimizer = components['prompt_optimizer']
        self.learning_scheduler = LearningScheduler()

    async def start_learning_cycle(self):
        """学習サイクルを開始"""

        # 定期的な学習タスクをスケジュール
        self.learning_scheduler.schedule_task(
            "quality_evaluation",
            self._evaluate_recent_conversations,
            interval_minutes=30
        )

        self.learning_scheduler.schedule_task(
            "knowledge_extraction",
            self._extract_new_knowledge,
            interval_hours=6
        )

        self.learning_scheduler.schedule_task(
            "prompt_optimization",
            self._optimize_prompts,
            interval_hours=24
        )

        self.learning_scheduler.schedule_task(
            "performance_analysis",
            self._analyze_performance_trends,
            interval_hours=12
        )

    async def _evaluate_recent_conversations(self):
        """最近の会話を評価"""
        recent_conversations = await self.conversation_manager.get_recent_unevaluated()

        for conv in recent_conversations:
            quality_metrics = await self.quality_evaluator.evaluate_response(
                conv['user_input'],
                conv['agent_response'],
                conv['context_data']
            )

            await self.conversation_manager.update_quality_metrics(
                conv['id'], quality_metrics
            )

    async def _extract_new_knowledge(self):
        """新しい知識を抽出"""
        extracted_count = await self.knowledge_extractor.extract_knowledge_from_conversations()

        if extracted_count > 0:
            logger.info(f"Extracted {extracted_count} new knowledge items")

            # 知識ベースの品質チェック
            await self._validate_knowledge_quality()

    async def _optimize_prompts(self):
        """プロンプトを最適化"""
        optimization_count = await self.prompt_optimizer.optimize_prompts()

        if optimization_count > 0:
            logger.info(f"Started {optimization_count} prompt optimization tests")
```

### 5.2 学習効果測定

```python
class LearningEffectMeasurement:
    """学習効果を測定するクラス"""

    def __init__(self, db_connection):
        self.db = db_connection

    async def measure_improvement_over_time(
        self,
        time_period_days: int = 30
    ) -> dict:
        """時系列での改善効果を測定"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)

        # 期間内の会話を取得
        conversations = await self.db.get_conversations_in_period(
            start_date, end_date
        )

        # 時系列データに変換
        daily_metrics = self._group_by_day(conversations)

        # トレンド分析
        trend_analysis = {
            'quality_trend': self._calculate_trend(daily_metrics, 'quality_score'),
            'response_time_trend': self._calculate_trend(daily_metrics, 'response_time'),
            'user_satisfaction_trend': self._calculate_trend(daily_metrics, 'user_feedback'),
            'learning_events': await self._get_learning_events_in_period(start_date, end_date)
        }

        return trend_analysis

    async def generate_learning_report(self) -> dict:
        """学習レポートを生成"""

        report = {
            'summary': await self._generate_summary(),
            'performance_metrics': await self._calculate_performance_metrics(),
            'knowledge_growth': await self._analyze_knowledge_growth(),
            'prompt_evolution': await self._analyze_prompt_evolution(),
            'recommendations': await self._generate_recommendations()
        }

        return report
```

## 6. セーフガード機能

### 6.1 有害学習防止

```python
class SafetyGuard:
    """有害な学習を防ぐセーフガードクラス"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.harmful_patterns = self._load_harmful_patterns()
        self.safety_threshold = 0.8

    async def validate_knowledge(self, knowledge_item: dict) -> bool:
        """知識アイテムの安全性を検証"""

        # 有害パターンチェック
        if self._contains_harmful_pattern(knowledge_item['content']):
            return False

        # LLMによる安全性評価
        safety_score = await self._evaluate_safety(knowledge_item)

        return safety_score >= self.safety_threshold

    async def validate_prompt_change(
        self,
        original_prompt: str,
        new_prompt: str
    ) -> bool:
        """プロンプト変更の安全性を検証"""

        validation_prompt = f"""
        以下のプロンプト変更が安全かどうか評価してください。

        元のプロンプト: {original_prompt}
        新しいプロンプト: {new_prompt}

        評価基準:
        1. 有害なコンテンツ生成の可能性
        2. プライバシー侵害のリスク
        3. 誤情報生成のリスク
        4. 倫理的な問題

        0-1のスコアで評価し、理由も説明してください。
        """

        response = await self.llm.generate(
            prompt=validation_prompt,
            max_tokens=500,
            temperature=0.1
        )

        safety_evaluation = self._parse_safety_evaluation(response)
        return safety_evaluation['score'] >= self.safety_threshold
```

## 7. 実装スケジュール

### Week 1: 基盤実装

- [ ] データベーススキーマ作成
- [ ] ConversationManager 実装
- [ ] 基本的な会話記録機能

### Week 2: 品質評価システム

- [ ] QualityEvaluator 実装
- [ ] 自動評価プロンプト作成
- [ ] 客観的指標計算機能

### Week 3: 知識抽出システム

- [ ] KnowledgeExtractor 実装
- [ ] パターン抽出機能
- [ ] 知識ベース管理機能

### Week 4: プロンプト最適化

- [ ] PromptOptimizer 実装
- [ ] A/B テスト機能
- [ ] 自動最適化ループ

### Week 5: 統合・テスト

- [ ] 全コンポーネント統合
- [ ] セーフガード機能実装
- [ ] パフォーマンステスト

## 8. 成功指標

### 8.1 定量的指標

- **品質スコア向上**: 4 週間で平均 20%向上
- **ユーザー満足度**: 👍 フィードバック率 80%以上
- **応答時間**: 自己チューニング後も 5 秒以内維持
- **学習効果**: 同様質問への応答品質継続的向上

### 8.2 定性的指標

- **適応性**: 新しいドメインへの適応速度
- **一貫性**: 学習後も一貫した品質維持
- **透明性**: 学習プロセスの可視化品質
- **安全性**: 有害学習の防止効果

---

**作成日**: 2024 年 12 月
**対象**: 自己チューニング機能実装
**優先度**: 最高（Phase 2）
