#!/usr/bin/env python3
"""
Configuration-based Test System
設定ファイルベースのテスト実行システム
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from automated_chat import AutomatedChatSystem


class ConfigBasedTestSystem:
    """設定ファイルベーステストシステム"""

    def __init__(self, config_file: str = "test_config.json"):
        self.config_file = config_file
        self.config = None
        self.chat_system = AutomatedChatSystem()

    def load_config(self) -> bool:
        """設定ファイル読み込み"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            print(f"[CONFIG] 設定ファイル読み込み完了: {self.config_file}")
            return True
            
        except FileNotFoundError:
            print(f"[ERROR] 設定ファイルが見つかりません: {self.config_file}")
            return False
        except json.JSONDecodeError as e:
            print(f"[ERROR] 設定ファイルの形式が不正です: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] 設定ファイル読み込みエラー: {e}")
            return False

    async def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """指定されたシナリオを実行"""
        if scenario_name not in self.config['test_scenarios']:
            raise ValueError(f"不明なシナリオ: {scenario_name}")
        
        scenario = self.config['test_scenarios'][scenario_name]
        questions = scenario['questions']
        description = scenario.get('description', scenario_name)
        
        print(f"[CONFIG] シナリオ実行: {description}")
        print(f"[CONFIG] 質問数: {len(questions)}")
        
        results = await self.chat_system.execute_chat_sequence(questions)
        
        return {
            'scenario_name': scenario_name,
            'description': description,
            'questions': questions,
            'results': results,
            'summary': self._generate_summary(results)
        }

    async def run_all_scenarios(self) -> Dict[str, Any]:
        """全シナリオを実行"""
        if not await self.chat_system.initialize():
            raise Exception("チャットシステム初期化に失敗")
        
        try:
            all_results = {}
            scenarios = list(self.config['test_scenarios'].keys())
            
            print(f"[CONFIG] {len(scenarios)}個のシナリオを実行開始...")
            
            for i, scenario_name in enumerate(scenarios, 1):
                print(f"\n[CONFIG] シナリオ {i}/{len(scenarios)}: {scenario_name}")
                print("-" * 60)
                
                scenario_result = await self.run_scenario(scenario_name)
                all_results[scenario_name] = scenario_result
                
                # シナリオ間の間隔
                interval = self.config.get('execution_settings', {}).get('question_interval_seconds', 1)
                if i < len(scenarios):
                    await asyncio.sleep(interval)
            
            return {
                'execution_time': datetime.now().isoformat(),
                'config_file': self.config_file,
                'scenarios_executed': len(scenarios),
                'results': all_results
            }
            
        finally:
            await self.chat_system.shutdown()

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """結果サマリー生成"""
        if not results:
            return {}
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_time = sum(r['execution_time'] for r in results)
        
        return {
            'total_questions': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) if results else 0,
            'total_execution_time': total_time,
            'average_execution_time': total_time / len(results) if results else 0
        }

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """結果保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"config_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"[CONFIG] 結果保存完了: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] 結果保存エラー: {e}")
            return None

    def print_results_summary(self, results: Dict[str, Any]):
        """結果サマリー表示"""
        print(f"\n{'='*80}")
        print(f"[CONFIG] テスト実行結果サマリー")
        print(f"{'='*80}")
        print(f"実行時刻: {results['execution_time']}")
        print(f"設定ファイル: {results['config_file']}")
        print(f"実行シナリオ数: {results['scenarios_executed']}")
        
        print(f"\n[CONFIG] シナリオ別結果:")
        for scenario_name, scenario_data in results['results'].items():
            summary = scenario_data['summary']
            print(f"\n  📋 {scenario_name}:")
            print(f"     説明: {scenario_data['description']}")
            print(f"     質問数: {summary['total_questions']}")
            print(f"     成功率: {summary['success_rate']:.1%}")
            print(f"     実行時間: {summary['total_execution_time']:.2f}秒")
        
        print(f"{'='*80}")


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="設定ファイルベーステスト実行")
    parser.add_argument("--config", default="test_config.json", help="設定ファイルパス")
    parser.add_argument("--scenario", help="実行する特定のシナリオ名")
    parser.add_argument("--output", help="結果出力ファイル名")
    parser.add_argument("--no-save", action="store_true", help="結果を保存しない")
    
    args = parser.parse_args()
    
    test_system = ConfigBasedTestSystem(args.config)
    
    try:
        if not test_system.load_config():
            return
        
        if args.scenario:
            # 特定シナリオのみ実行
            if not await test_system.chat_system.initialize():
                print("[ERROR] システム初期化に失敗")
                return
            
            try:
                result = await test_system.run_scenario(args.scenario)
                print(f"\n[CONFIG] シナリオ '{args.scenario}' 実行完了")
                
                # 簡易サマリー表示
                summary = result['summary']
                print(f"成功率: {summary['success_rate']:.1%}")
                print(f"実行時間: {summary['total_execution_time']:.2f}秒")
                
            finally:
                await test_system.chat_system.shutdown()
        else:
            # 全シナリオ実行
            results = await test_system.run_all_scenarios()
            
            # 結果表示
            test_system.print_results_summary(results)
            
            # 結果保存
            if not args.no_save:
                test_system.save_results(results, args.output)
            
    except KeyboardInterrupt:
        print("\n[CONFIG] ユーザーによって中断されました")
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())