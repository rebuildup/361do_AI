#!/usr/bin/env python3
"""
Batch Chat Execution
複数の質問セットを連続実行するバッチシステム
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from automated_chat import AutomatedChatSystem, PREDEFINED_QUESTION_SETS


class BatchChatSystem:
    """バッチチャット実行システム"""

    def __init__(self):
        self.chat_system = AutomatedChatSystem()
        self.batch_results = []

    async def run_batch_tests(self, test_sets: List[str] = None) -> Dict[str, Any]:
        """複数のテストセットを連続実行"""
        if test_sets is None:
            test_sets = list(PREDEFINED_QUESTION_SETS.keys())
        
        print(f"[BATCH] {len(test_sets)}個のテストセットを実行開始...")
        
        batch_start_time = datetime.now()
        all_results = {}
        
        try:
            if not await self.chat_system.initialize():
                raise Exception("システム初期化に失敗")
            
            for i, test_set in enumerate(test_sets, 1):
                print(f"\n[BATCH] テストセット {i}/{len(test_sets)}: {test_set}")
                print("-" * 50)
                
                if test_set not in PREDEFINED_QUESTION_SETS:
                    print(f"[ERROR] 不明なテストセット: {test_set}")
                    continue
                
                questions = PREDEFINED_QUESTION_SETS[test_set]
                results = await self.chat_system.execute_chat_sequence(questions)
                
                all_results[test_set] = {
                    'questions': questions,
                    'results': results,
                    'summary': self._generate_summary(results)
                }
                
                print(f"[BATCH] テストセット '{test_set}' 完了")
                
                # テストセット間の間隔
                if i < len(test_sets):
                    await asyncio.sleep(2)
            
            batch_end_time = datetime.now()
            batch_duration = (batch_end_time - batch_start_time).total_seconds()
            
            # バッチサマリー生成
            batch_summary = {
                'start_time': batch_start_time.isoformat(),
                'end_time': batch_end_time.isoformat(),
                'duration_seconds': batch_duration,
                'test_sets_executed': len(test_sets),
                'total_questions': sum(len(PREDEFINED_QUESTION_SETS.get(ts, [])) for ts in test_sets),
                'results': all_results
            }
            
            return batch_summary
            
        finally:
            await self.chat_system.shutdown()

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """個別テストセットのサマリー生成"""
        if not results:
            return {}
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_time = sum(r['execution_time'] for r in results)
        avg_time = total_time / len(results) if results else 0
        
        # ツール使用統計
        all_tools = []
        for r in results:
            all_tools.extend(r.get('tools_used', []))
        
        tool_stats = {}
        if all_tools:
            from collections import Counter
            tool_counts = Counter(all_tools)
            tool_stats = dict(tool_counts)
        
        return {
            'total_questions': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) if results else 0,
            'total_execution_time': total_time,
            'average_execution_time': avg_time,
            'tools_used': tool_stats
        }

    def save_batch_results(self, batch_summary: Dict[str, Any], filename: str = None):
        """バッチ結果を保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_chat_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, ensure_ascii=False, indent=2)
            
            print(f"[BATCH] バッチ結果を保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] バッチ結果保存エラー: {e}")
            return None

    def print_batch_summary(self, batch_summary: Dict[str, Any]):
        """バッチサマリーを表示"""
        print(f"\n{'='*80}")
        print(f"[BATCH] バッチ実行サマリー")
        print(f"{'='*80}")
        print(f"実行開始: {batch_summary['start_time']}")
        print(f"実行終了: {batch_summary['end_time']}")
        print(f"総実行時間: {batch_summary['duration_seconds']:.2f}秒")
        print(f"テストセット数: {batch_summary['test_sets_executed']}")
        print(f"総質問数: {batch_summary['total_questions']}")
        
        print(f"\n[BATCH] テストセット別結果:")
        for test_set, data in batch_summary['results'].items():
            summary = data['summary']
            print(f"\n  {test_set}:")
            print(f"    質問数: {summary['total_questions']}")
            print(f"    成功率: {summary['success_rate']:.1%}")
            print(f"    平均実行時間: {summary['average_execution_time']:.2f}秒")
            
            if summary['tools_used']:
                tools = ', '.join([f"{tool}({count})" for tool, count in summary['tools_used'].items()])
                print(f"    使用ツール: {tools}")
        
        print(f"{'='*80}")


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="バッチチャット実行システム")
    parser.add_argument("--sets", nargs="+", choices=list(PREDEFINED_QUESTION_SETS.keys()),
                       help="実行するテストセット（指定しない場合は全て実行）")
    parser.add_argument("--output", help="結果出力ファイル名")
    parser.add_argument("--no-save", action="store_true", help="結果を保存しない")
    
    args = parser.parse_args()
    
    batch_system = BatchChatSystem()
    
    try:
        batch_summary = await batch_system.run_batch_tests(args.sets)
        
        # 結果表示
        batch_system.print_batch_summary(batch_summary)
        
        # 結果保存
        if not args.no_save:
            batch_system.save_batch_results(batch_summary, args.output)
            
    except KeyboardInterrupt:
        print("\n[BATCH] ユーザーによって中断されました")
    except Exception as e:
        print(f"[ERROR] バッチ実行エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())