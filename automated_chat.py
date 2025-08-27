#!/usr/bin/env python3
"""
Automated Chat System
事前定義された質問を自動実行してエージェントの応答を取得
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient
from agent.core.agent_manager import AgentManager


class AutomatedChatSystem:
    """自動化チャットシステム"""

    def __init__(self):
        self.config = None
        self.db_manager = None
        self.agent_manager = None
        self.results = []

    async def initialize(self):
        """システム初期化"""
        print("[AUTO] 自動化チャットシステムを初期化中...")
        
        try:
            self.config = Config()
            
            # データベース初期化
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            
            # エージェントマネージャー初期化
            self.agent_manager = AgentManager(self.config, self.db_manager)
            await self.agent_manager.initialize()
            
            print("[AUTO] 初期化完了")
            return True
            
        except Exception as e:
            print(f"[ERROR] 初期化エラー: {e}")
            return False

    async def shutdown(self):
        """システム終了処理"""
        print("[AUTO] システムを終了中...")
        
        if self.agent_manager:
            await self.agent_manager.shutdown()
        
        if self.db_manager:
            await self.db_manager.close()
        
        print("[AUTO] 終了完了")

    async def execute_chat_sequence(self, questions: List[str], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """質問シーケンスの自動実行"""
        results = []
        
        print(f"[AUTO] {len(questions)}個の質問を自動実行開始...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[AUTO] 質問 {i}/{len(questions)}: {question}")
            
            start_time = time.time()
            
            try:
                response = await self.agent_manager.process_message(
                    user_input=question,
                    session_id=session_id
                )
                
                execution_time = time.time() - start_time
                
                result = {
                    'question_number': i,
                    'question': question,
                    'response': response.get('response', ''),
                    'session_id': response.get('session_id', session_id),
                    'execution_time': execution_time,
                    'intent': response.get('intent', {}),
                    'tools_used': response.get('tools_used', []),
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
                print(f"[AUTO] 応答: {response.get('response', '')[:100]}...")
                print(f"[AUTO] 実行時間: {execution_time:.2f}秒")
                
                if response.get('tools_used'):
                    print(f"[AUTO] 使用ツール: {', '.join(response['tools_used'])}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = {
                    'question_number': i,
                    'question': question,
                    'response': f"エラー: {str(e)}",
                    'session_id': session_id,
                    'execution_time': execution_time,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'success': False
                }
                
                print(f"[ERROR] 質問 {i} でエラー: {e}")
            
            results.append(result)
            
            # 質問間の間隔（システム負荷軽減）
            if i < len(questions):
                await asyncio.sleep(1)
        
        print(f"\n[AUTO] 全質問実行完了 ({len(results)}件)")
        return results

    def save_results(self, results: List[Dict[str, Any]], filename: Optional[str] = None):
        """結果をファイルに保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"automated_chat_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"[AUTO] 結果を保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] 結果保存エラー: {e}")
            return None

    def print_summary(self, results: List[Dict[str, Any]]):
        """結果サマリーを表示"""
        if not results:
            print("[AUTO] 結果がありません")
            return
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_time = sum(r['execution_time'] for r in results)
        avg_time = total_time / len(results)
        
        print(f"\n{'='*60}")
        print(f"[AUTO] 実行サマリー")
        print(f"{'='*60}")
        print(f"総質問数: {len(results)}")
        print(f"成功: {successful}")
        print(f"失敗: {failed}")
        print(f"総実行時間: {total_time:.2f}秒")
        print(f"平均実行時間: {avg_time:.2f}秒")
        
        # ツール使用統計
        all_tools = []
        for r in results:
            all_tools.extend(r.get('tools_used', []))
        
        if all_tools:
            from collections import Counter
            tool_counts = Counter(all_tools)
            print(f"\nツール使用統計:")
            for tool, count in tool_counts.most_common():
                print(f"  {tool}: {count}回")
        
        print(f"{'='*60}")


# 事前定義された質問セット
PREDEFINED_QUESTION_SETS = {
    "basic_functionality": [
        "こんにちは",
        "あなたの機能について教えてください",
        "help",
        "利用可能なツールを教えて"
    ],
    
    "command_execution": [
        "systeminfoコマンドを実行してください",
        "tasklistでプロセス一覧を表示",
        "現在のディレクトリ内容を確認",
        "ホスト名を表示してください"
    ],
    
    "web_search": [
        "最新のAI技術について調べて",
        "Python プログラミング チュートリアルを探して",
        "機械学習 入門 情報を検索",
        "ボーカロイド 楽曲投稿祭 ボカコレ 情報を検索"
    ],
    
    "learning_system": [
        "学習データの統計を表示",
        "一番古い学習データを教えて",
        "学習システムの状態を確認",
        "プロンプト一覧を表示"
    ],
    
    "comprehensive_test": [
        "こんにちは、機能テストを開始します",
        "systeminfoコマンドを実行してシステム情報を取得",
        "先ほどの結果について簡潔に説明してください",
        "最新のPython情報について検索",
        "学習データの統計情報を教えて",
        "help tools",
        "tasklistを実行してメモリ使用量の多いプロセスを特定",
        "ありがとうございました、テスト完了です"
    ]
}


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自動化チャットシステム")
    parser.add_argument("--set", choices=list(PREDEFINED_QUESTION_SETS.keys()), 
                       default="basic_functionality", help="使用する質問セット")
    parser.add_argument("--questions", nargs="+", help="カスタム質問リスト")
    parser.add_argument("--output", help="結果出力ファイル名")
    parser.add_argument("--no-save", action="store_true", help="結果を保存しない")
    
    args = parser.parse_args()
    
    # 質問リストの決定
    if args.questions:
        questions = args.questions
        print(f"[AUTO] カスタム質問を使用: {len(questions)}個")
    else:
        questions = PREDEFINED_QUESTION_SETS[args.set]
        print(f"[AUTO] 事前定義セット '{args.set}' を使用: {len(questions)}個")
    
    # システム実行
    chat_system = AutomatedChatSystem()
    
    try:
        if await chat_system.initialize():
            results = await chat_system.execute_chat_sequence(questions)
            
            # 結果表示
            chat_system.print_summary(results)
            
            # 結果保存
            if not args.no_save:
                chat_system.save_results(results, args.output)
            
        else:
            print("[ERROR] システム初期化に失敗しました")
            
    except KeyboardInterrupt:
        print("\n[AUTO] ユーザーによって中断されました")
    except Exception as e:
        print(f"[ERROR] 予期しないエラー: {e}")
    finally:
        await chat_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())