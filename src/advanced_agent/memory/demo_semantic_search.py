"""
Semantic Search System Demonstration

ChromaDB + Sentence-Transformers 意味的記憶検索システムのデモンストレーション
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from .persistent_memory import LangChainPersistentMemory
from .semantic_search import SentenceTransformersSearchEngine, ChromaDBSemanticMemory


async def demo_sentence_transformers_engine():
    """Sentence-Transformers 検索エンジンのデモ"""
    
    print("=== Sentence-Transformers 検索エンジン デモ ===\n")
    
    search_engine = SentenceTransformersSearchEngine(device="cpu")
    
    # 1. テキスト埋め込みと類似度計算
    print("1. テキスト埋め込みと類似度計算")
    
    test_texts = [
        "Pythonでの機械学習プログラミング",
        "機械学習アルゴリズムの実装",
        "JavaScriptでのWeb開発",
        "データベース設計の基本原則",
        "GPU メモリ最適化の手法"
    ]
    
    query = "Python 機械学習"
    print(f"   クエリ: '{query}'\n")
    
    similarities = []
    for text in test_texts:
        similarity = search_engine.calculate_similarity(query, text)
        similarities.append((text, similarity))
        print(f"   '{text[:30]}...' : {similarity:.3f}")
    
    print()
    
    # 2. 類似記憶検索
    print("2. 類似記憶検索")
    
    memory_metadata = [{"id": i, "type": "conversation"} for i in range(len(test_texts))]
    
    search_results = search_engine.find_similar_memories(
        query=query,
        memory_texts=test_texts,
        memory_metadata=memory_metadata,
        max_results=3
    )
    
    print(f"   検索結果数: {len(search_results)}")
    for i, result in enumerate(search_results, 1):
        print(f"   {i}. 類似度: {result['similarity']:.3f}")
        print(f"      内容: {result['text'][:50]}...")
    
    print()
    
    # 3. 記憶クラスタリング
    print("3. 記憶クラスタリング")
    
    extended_texts = test_texts + [
        "Python データ分析ライブラリ",
        "React コンポーネント設計",
        "SQL クエリ最適化",
        "深層学習モデルの訓練",
        "Node.js サーバー開発"
    ]
    
    clustering_result = search_engine.cluster_memories(
        memory_texts=extended_texts,
        n_clusters=3
    )
    
    print(f"   総記憶数: {len(extended_texts)}")
    print(f"   クラスタ数: {len(clustering_result['clusters'])}")
    
    for cluster_id, memories in clustering_result["clusters"].items():
        representative = clustering_result["representatives"][cluster_id]
        print(f"   クラスタ {cluster_id}: {len(memories)} 件")
        print(f"     代表: {representative['text'][:40]}...")
    
    print()


async def demo_semantic_memory_system():
    """意味的記憶システムのデモ"""
    
    print("=== ChromaDB + Sentence-Transformers 統合記憶システム デモ ===\n")
    
    # システム初期化
    persistent_memory = LangChainPersistentMemory(
        db_path="data/demo_semantic_memory.db",
        chroma_path="data/demo_semantic_chroma"
    )
    
    semantic_memory = ChromaDBSemanticMemory(
        persistent_memory=persistent_memory
    )
    
    try:
        # セッション初期化
        session_id = await persistent_memory.initialize_session(user_id="semantic_demo_user")
        print(f"1. セッション初期化: {session_id[:20]}...\n")
        
        # 多様な会話データを追加
        conversations = [
            ("Pythonでの機械学習について教えて", "scikit-learn, TensorFlow, PyTorchなどのライブラリが人気です。"),
            ("機械学習アルゴリズムの種類は？", "教師あり学習、教師なし学習、強化学習に分類されます。"),
            ("GPU メモリ不足の解決方法", "バッチサイズ削減、勾配蓄積、モデル量子化が効果的です。"),
            ("JavaScriptでのDOM操作", "document.querySelector()やaddEventListener()を使用します。"),
            ("React コンポーネントの設計", "関数コンポーネントとHooksを使用することを推奨します。"),
            ("データベース正規化について", "第一正規形から第三正規形まで段階的に正規化します。"),
            ("SQL インデックスの最適化", "適切なインデックス設計でクエリ性能が向上します。"),
            ("Python エラーのデバッグ方法", "トレースバックを読み、ログを活用してエラーを特定します。"),
            ("機械学習モデルの評価指標", "精度、再現率、F1スコア、AUCなどを使用します。"),
            ("Web セキュリティの基本", "HTTPS、CSRF対策、XSS対策が重要です。")
        ]
        
        print("2. 会話データの追加")
        for i, (user_input, agent_response) in enumerate(conversations, 1):
            await persistent_memory.store_conversation(
                user_input=user_input,
                agent_response=agent_response,
                metadata={"demo_step": i, "timestamp": datetime.now().isoformat()}
            )
            print(f"   会話 {i}: {user_input[:30]}...")
        
        print()
        
        # 拡張類似度検索のデモ
        print("3. 拡張類似度検索")
        
        search_queries = [
            "Python 機械学習 エラー",
            "JavaScript React 開発",
            "データベース SQL 最適化"
        ]
        
        for query in search_queries:
            print(f"\n   クエリ: '{query}'")
            
            search_result = await semantic_memory.enhanced_similarity_search(
                query=query,
                session_id=session_id,
                max_results=3
            )
            
            enhanced_results = search_result["enhanced_results"]
            print(f"     検索結果数: {len(enhanced_results)}")
            
            for j, result in enumerate(enhanced_results[:2], 1):
                print(f"       {j}. 統合スコア: {result['combined_score']:.3f}")
                print(f"          類似度: {result['similarity']:.3f}")
                print(f"          内容: {result['text'][:60]}...")
        
        print()
        
        # パターン学習のデモ
        print("4. 検索パターン学習")
        
        # 成功した検索結果を使ってパターン学習
        successful_query = "Python 機械学習"
        search_result = await semantic_memory.enhanced_similarity_search(
            query=successful_query,
            session_id=session_id,
            max_results=3
        )
        
        if search_result["enhanced_results"]:
            pattern_name = await semantic_memory.learn_search_pattern(
                query=successful_query,
                successful_results=search_result["enhanced_results"],
                pattern_name="python_ml_pattern"
            )
            
            print(f"   学習パターン: {pattern_name}")
            print(f"   関連記憶数: {len(search_result['enhanced_results'])}")
            
            # パターンを使った検索
            pattern_search = await semantic_memory.enhanced_similarity_search(
                query="Python での AI 開発",  # 類似クエリ
                session_id=session_id,
                max_results=3,
                use_patterns=True
            )
            
            pattern_matches = pattern_search["pattern_matches"]
            print(f"   パターンマッチ数: {len(pattern_matches)}")
            
            for match in pattern_matches:
                print(f"     パターン: {match['pattern_name']}")
                print(f"     類似度: {match['pattern_similarity']:.3f}")
        
        print()
        
        # セッションクラスタリング
        print("5. セッション記憶クラスタリング")
        
        clustering_result = await semantic_memory.cluster_session_memories(
            session_id=session_id,
            n_clusters=4
        )
        
        print(f"   総記憶数: {clustering_result['total_memories']}")
        print(f"   クラスタ数: {clustering_result['n_clusters']}")
        
        for cluster_id, cluster_data in clustering_result["clusters"].items():
            print(f"   クラスタ {cluster_id}: {cluster_data['size']} 件")
            representative = cluster_data["representative"]
            print(f"     代表記憶: {representative['text'][:50]}...")
        
        print()
        
        # 記憶進化パターン検出
        print("6. 記憶進化パターン検出")
        
        # 時系列データを追加（過去の会話として）
        evolution_conversations = [
            (datetime.now() - timedelta(days=7), "Python基礎学習", "変数と関数について学習中"),
            (datetime.now() - timedelta(days=5), "Python中級", "クラスとモジュールを理解"),
            (datetime.now() - timedelta(days=3), "Python応用", "機械学習ライブラリの使用開始"),
            (datetime.now() - timedelta(days=1), "Python実践", "実際のプロジェクトで機械学習を適用")
        ]
        
        for timestamp, user_input, agent_response in evolution_conversations:
            await persistent_memory.store_conversation(
                user_input=user_input,
                agent_response=agent_response,
                metadata={"timestamp": timestamp.isoformat(), "evolution_demo": True}
            )
        
        evolution_result = await semantic_memory.find_memory_evolution(
            topic="Python学習",
            session_id=session_id,
            time_window_days=10
        )
        
        print(f"   分析期間: {evolution_result['time_window_days']} 日")
        print(f"   対象記憶数: {evolution_result['total_memories']}")
        
        evolution_patterns = evolution_result["evolution_patterns"]
        print(f"   進化パターン数: {len(evolution_patterns)}")
        
        for i, pattern in enumerate(evolution_patterns[:3], 1):
            print(f"   パターン {i}:")
            print(f"     変化タイプ: {pattern['evolution_type']}")
            print(f"     内容類似度: {pattern['content_similarity']:.3f}")
            print(f"     時間差: {pattern['time_difference_hours']:.1f} 時間")
        
        summary = evolution_result["summary"]
        print(f"   変化サマリー:")
        print(f"     大きな変化: {summary['significant_changes']} 件")
        print(f"     段階的変化: {summary['gradual_changes']} 件")
        print(f"     小さな更新: {summary['minor_updates']} 件")
        
        print()
        
        # 統計情報
        print("7. システム統計")
        
        stats = semantic_memory.get_search_statistics()
        
        search_perf = stats["search_performance"]
        print(f"   検索パフォーマンス:")
        print(f"     総検索数: {search_perf['total_searches']}")
        print(f"     成功検索数: {search_perf['successful_searches']}")
        print(f"     成功率: {search_perf['success_rate']:.1%}")
        print(f"     パターンマッチ: {search_perf['pattern_matches']}")
        
        learned_patterns = stats["learned_patterns"]
        print(f"   学習パターン:")
        print(f"     総パターン数: {learned_patterns['total_patterns']}")
        if learned_patterns["most_used_pattern"]:
            print(f"     最頻使用: {learned_patterns['most_used_pattern']}")
        
        model_info = stats["model_info"]
        print(f"   モデル情報:")
        print(f"     埋め込み次元: {model_info['embedding_dimension']}")
        print(f"     類似度閾値: {model_info['similarity_threshold']}")
        print(f"     キャッシュサイズ: {model_info['cache_size']}")
        
    finally:
        persistent_memory.close()


async def demo_advanced_search_features():
    """高度な検索機能のデモ"""
    
    print("\n=== 高度な検索機能デモ ===\n")
    
    # システム初期化
    persistent_memory = LangChainPersistentMemory(
        db_path="data/demo_advanced_search.db",
        chroma_path="data/demo_advanced_chroma"
    )
    
    semantic_memory = ChromaDBSemanticMemory(
        persistent_memory=persistent_memory
    )
    
    try:
        session_id = await persistent_memory.initialize_session(user_id="advanced_demo_user")
        
        # 専門的な会話データを追加
        specialized_conversations = [
            ("深層学習の最適化手法", "Adam、RMSprop、SGDなどの最適化アルゴリズムがあります。"),
            ("Transformer アーキテクチャ", "Attention機構により長距離依存関係を効率的に学習します。"),
            ("BERT の事前訓練", "Masked Language ModelとNext Sentence Predictionを使用します。"),
            ("GPT の生成メカニズム", "自己回帰的に次のトークンを予測して文章を生成します。"),
            ("量子化技術の種類", "Post-training量子化とQuantization-aware trainingがあります。"),
            ("LoRA の仕組み", "低ランク行列分解により効率的にファインチューニングします。"),
            ("RLHF の手順", "Reward Model訓練、PPO最適化の段階的プロセスです。"),
            ("マルチモーダル学習", "テキスト、画像、音声を統合的に処理する技術です。")
        ]
        
        print("1. 専門的会話データの追加")
        for i, (user_input, agent_response) in enumerate(specialized_conversations, 1):
            await persistent_memory.store_conversation(
                user_input=user_input,
                agent_response=agent_response,
                metadata={
                    "domain": "deep_learning",
                    "complexity": "advanced",
                    "timestamp": datetime.now().isoformat()
                }
            )
            print(f"   専門会話 {i}: {user_input[:40]}...")
        
        print()
        
        # 複雑なクエリでの検索
        print("2. 複雑クエリ検索")
        
        complex_queries = [
            "Transformer Attention 最適化",
            "量子化 LoRA 効率化",
            "BERT GPT 比較分析"
        ]
        
        for query in complex_queries:
            print(f"\n   複雑クエリ: '{query}'")
            
            search_result = await semantic_memory.enhanced_similarity_search(
                query=query,
                session_id=session_id,
                max_results=3
            )
            
            enhanced_results = search_result["enhanced_results"]
            print(f"     マッチ数: {len(enhanced_results)}")
            
            for j, result in enumerate(enhanced_results, 1):
                print(f"       {j}. スコア: {result['combined_score']:.3f}")
                print(f"          内容: {result['text'][:50]}...")
        
        print()
        
        # 高度なクラスタリング
        print("3. 高度なクラスタリング分析")
        
        clustering_result = await semantic_memory.cluster_session_memories(
            session_id=session_id,
            n_clusters=3
        )
        
        clusters = clustering_result["clusters"]
        print(f"   専門分野クラスタ数: {len(clusters)}")
        
        for cluster_id, cluster_data in clusters.items():
            memories = cluster_data["memories"]
            print(f"   クラスタ {cluster_id}: {len(memories)} 件")
            
            # クラスタ内の専門用語分析
            cluster_texts = [memory["text"] for memory in memories]
            technical_terms = []
            
            for text in cluster_texts:
                # 簡単な専門用語検出
                terms = ["Transformer", "BERT", "GPT", "LoRA", "量子化", "Attention", "RLHF"]
                for term in terms:
                    if term in text and term not in technical_terms:
                        technical_terms.append(term)
            
            print(f"     専門用語: {', '.join(technical_terms[:3])}")
        
        print()
        
        # パフォーマンス分析
        print("4. 検索パフォーマンス分析")
        
        stats = semantic_memory.get_search_statistics()
        model_info = stats["model_info"]
        
        print(f"   埋め込みモデル次元: {model_info['embedding_dimension']}")
        print(f"   キャッシュ効率: {model_info['cache_size']} エントリ")
        print(f"   類似度閾値: {model_info['similarity_threshold']}")
        
        # 検索精度テスト
        precision_queries = [
            ("深層学習", ["深層学習", "Transformer", "BERT"]),
            ("最適化", ["最適化", "Adam", "SGD"]),
            ("量子化", ["量子化", "LoRA", "効率"])
        ]
        
        print(f"   精度テスト結果:")
        for query, expected_terms in precision_queries:
            search_result = await semantic_memory.enhanced_similarity_search(
                query=query,
                session_id=session_id,
                max_results=3
            )
            
            found_terms = 0
            total_results = len(search_result["enhanced_results"])
            
            for result in search_result["enhanced_results"]:
                for term in expected_terms:
                    if term in result["text"]:
                        found_terms += 1
                        break
            
            precision = found_terms / total_results if total_results > 0 else 0
            print(f"     '{query}': 精度 {precision:.1%} ({found_terms}/{total_results})")
        
    finally:
        persistent_memory.close()


async def main():
    """メインデモ実行"""
    
    # データディレクトリ作成
    Path("data").mkdir(exist_ok=True)
    
    try:
        await demo_sentence_transformers_engine()
        await demo_semantic_memory_system()
        await demo_advanced_search_features()
        
        print("\n=== デモ完了 ===")
        print("ChromaDB + Sentence-Transformers 意味的記憶検索システムが正常に動作することを確認しました。")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())