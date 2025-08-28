#!/usr/bin/env python3
"""
Launch Multi-Agent Learning System
マルチエージェント学習システム起動スクリプト
事前チェック → 最適化 → 実行の統合ランチャー
"""

import asyncio
import sys
import os
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class MultiAgentLauncher:
    """マルチエージェント学習システムランチャー"""
    
    def __init__(self):
        self.launch_time = datetime.now()
        self.pre_check_passed = False
        self.optimization_completed = False
        
    def print_banner(self):
        """起動バナー表示"""
        print("🚀" * 30)
        print("🤖 マルチエージェント学習システム 統合ランチャー 🤖")
        print("🚀" * 30)
        print(f"起動時刻: {self.launch_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print("4つの専門エージェントが8時間にわたって相互学習を行います")
        print("=" * 80)
        print()
    
    def get_user_confirmation(self, message: str) -> bool:
        """ユーザー確認"""
        while True:
            response = input(f"{message} (y/n): ").lower().strip()
            if response in ['y', 'yes', 'はい']:
                return True
            elif response in ['n', 'no', 'いいえ']:
                return False
            else:
                print("y または n で回答してください")
    
    async def run_pre_check(self) -> bool:
        """事前チェック実行"""
        print("🔍 STEP 1: システム事前チェック")
        print("-" * 40)
        
        try:
            # 事前チェックスクリプト実行
            result = subprocess.run([
                sys.executable, "pre_launch_checklist.py"
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ 事前チェック完了")
                
                # 重要エラーの有無を確認
                if "重要エラー: 0" in result.stdout:
                    self.pre_check_passed = True
                    print("✅ 全ての重要チェックに合格しました")
                    return True
                else:
                    print("❌ 重要エラーが検出されました")
                    print(result.stdout)
                    return False
            else:
                print("❌ 事前チェックでエラーが発生しました")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 事前チェック実行エラー: {e}")
            return False
    
    async def run_optimization(self) -> bool:
        """システム最適化実行"""
        print("\n⚡ STEP 2: システム最適化")
        print("-" * 40)
        
        if not self.get_user_confirmation("システム最適化を実行しますか？"):
            print("⏭️ 最適化をスキップします")
            return True
        
        try:
            # 最適化スクリプト実行
            result = subprocess.run([
                sys.executable, "system_optimizer.py"
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ システム最適化完了")
                self.optimization_completed = True
                return True
            else:
                print("❌ システム最適化でエラーが発生しました")
                print(result.stderr)
                
                # 最適化失敗でも続行するか確認
                return self.get_user_confirmation("最適化に失敗しましたが、続行しますか？")
                
        except Exception as e:
            print(f"❌ 最適化実行エラー: {e}")
            return self.get_user_confirmation("最適化でエラーが発生しましたが、続行しますか？")
    
    def show_execution_plan(self):
        """実行計画表示"""
        print("\n📋 STEP 3: 実行計画確認")
        print("-" * 40)
        print("実行予定:")
        print("  🤖 エージェント数: 4つ")
        print("  ⏰ 実行時間: 8時間")
        print("  🔄 学習サイクル: 約96回 (5分間隔)")
        print("  💬 予想会話数: 約192回")
        print("  📊 予想学習データ: 数百件")
        print()
        print("エージェント構成:")
        print("  🔍 リサーチャー: 情報収集と探求")
        print("  📊 アナライザー: 論理的分析と構造化")
        print("  💡 クリエイター: 創造的思考とアイデア生成")
        print("  ⚡ オプティマイザー: 効率化と品質向上")
        print()
        print("停止方法:")
        print("  - Ctrl+C で安全に停止")
        print("  - 8時間経過で自動停止")
        print("  - システムエラー時は自動停止")
        print()
    
    def create_launch_log(self):
        """起動ログ作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"launch_log_{timestamp}.txt"
        
        log_content = f"""マルチエージェント学習システム 起動ログ
==========================================

起動時刻: {self.launch_time.strftime('%Y-%m-%d %H:%M:%S')}
事前チェック: {'合格' if self.pre_check_passed else '失敗'}
最適化実行: {'完了' if self.optimization_completed else 'スキップ'}

実行設定:
- エージェント数: 4
- 実行時間制限: 8時間
- 学習サイクル間隔: 5分
- 自動停止: 有効

システム情報:
- Python: {sys.version}
- プラットフォーム: {sys.platform}
- 作業ディレクトリ: {os.getcwd()}

注意事項:
- 実行中は他の重いタスクを避けてください
- 定期的にログファイルを確認してください
- 異常を検知したらCtrl+Cで停止してください

==========================================
"""
        
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(log_content)
            print(f"📄 起動ログを作成: {log_filename}")
        except Exception as e:
            print(f"⚠️ 起動ログ作成エラー: {e}")
    
    async def launch_main_system(self):
        """メインシステム起動"""
        print("\n🚀 STEP 4: マルチエージェント学習システム起動")
        print("-" * 40)
        print("システムを起動しています...")
        print("停止するには Ctrl+C を押してください")
        print("=" * 80)
        
        try:
            # メインシステム実行
            process = subprocess.Popen([
                sys.executable, "multi_agent_learning_system.py", "--hours", "8.0"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding='utf-8', bufsize=1, universal_newlines=True)
            
            # リアルタイム出力表示
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # 終了コード確認
            return_code = process.poll()
            
            if return_code == 0:
                print("\n✅ マルチエージェント学習システムが正常に終了しました")
            else:
                print(f"\n⚠️ システムが終了コード {return_code} で終了しました")
            
            return return_code == 0
            
        except KeyboardInterrupt:
            print("\n👋 ユーザーによって中断されました")
            if 'process' in locals():
                process.terminate()
            return True
        except Exception as e:
            print(f"\n❌ システム起動エラー: {e}")
            return False
    
    def show_completion_summary(self, success: bool):
        """完了サマリー表示"""
        end_time = datetime.now()
        total_duration = end_time - self.launch_time
        
        print("\n" + "🎉" * 30)
        print("📊 実行完了サマリー")
        print("🎉" * 30)
        print(f"開始時刻: {self.launch_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総実行時間: {total_duration}")
        print(f"実行結果: {'成功' if success else '異常終了'}")
        print()
        
        print("📁 生成されたファイル:")
        
        # ログファイルの確認
        log_files = []
        for file in os.listdir('.'):
            if (file.startswith('multi_agent_learning_') and file.endswith('.log')) or \
               (file.startswith('multi_agent_learning_session_') and file.endswith('.json')):
                log_files.append(file)
        
        if log_files:
            for log_file in sorted(log_files):
                print(f"  📄 {log_file}")
        else:
            print("  ⚠️ ログファイルが見つかりません")
        
        print("\n💡 次のステップ:")
        if success:
            print("  1. 生成されたログファイルを確認")
            print("  2. 学習結果JSONファイルを分析")
            print("  3. システムパフォーマンスを評価")
            print("  4. 必要に応じて再実行や設定調整")
        else:
            print("  1. エラーログを確認")
            print("  2. システム状態をチェック")
            print("  3. 問題を修正後に再実行")
        
        print("=" * 80)


async def main():
    """メイン関数"""
    launcher = MultiAgentLauncher()
    
    try:
        # バナー表示
        launcher.print_banner()
        
        # STEP 1: 事前チェック
        if not await launcher.run_pre_check():
            print("\n❌ 事前チェックに失敗しました。問題を修正してから再実行してください。")
            return
        
        # STEP 2: システム最適化
        if not await launcher.run_optimization():
            print("\n❌ システム最適化で問題が発生しました。")
            if not launcher.get_user_confirmation("それでも続行しますか？"):
                return
        
        # STEP 3: 実行計画確認
        launcher.show_execution_plan()
        
        if not launcher.get_user_confirmation("8時間のマルチエージェント学習を開始しますか？"):
            print("👋 実行をキャンセルしました")
            return
        
        # 起動ログ作成
        launcher.create_launch_log()
        
        # STEP 4: メインシステム起動
        success = await launcher.launch_main_system()
        
        # 完了サマリー
        launcher.show_completion_summary(success)
        
    except KeyboardInterrupt:
        print("\n👋 ランチャーが中断されました")
    except Exception as e:
        print(f"\n❌ ランチャーエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())