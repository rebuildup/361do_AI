#!/usr/bin/env python3
"""
Quick Self-Learning Test
自己学習機能の簡単なテストと問題特定
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Tests should not contact a local ollama daemon; force skip mode
os.environ.setdefault('AGENT_SKIP_OLLAMA', '1')


async def quick_test():
    """クイック自己学習テスト"""
    print("🔍 自己学習機能クイック診断開始...")
    
    issues = []
    recommendations = []
    
    try:
        # 1. 基本インポートテスト
        print("\n📦 1. 基本インポートテスト")
        try:
            from agent.core.config import Config
            from agent.core.database import DatabaseManager
            from agent.core.agent_manager import AgentManager
            from agent.tools.file_tool import FileTool
            from agent.tools.learning_tool import LearningTool
            print("  ✅ 全ての必要なモジュールのインポート成功")
        except ImportError as e:
            print(f"  ❌ インポートエラー: {e}")
            issues.append(f"モジュールインポート失敗: {e}")
            recommendations.append("必要なモジュールが正しくインストールされているか確認してください")
            return
        
        # 2. 設定初期化テスト
        print("\n⚙️ 2. 設定初期化テスト")
        try:
            config = Config()
            print("  ✅ 設定初期化成功")
        except Exception as e:
            print(f"  ❌ 設定初期化エラー: {e}")
            issues.append(f"設定初期化失敗: {e}")
            recommendations.append("設定ファイルを確認してください")
            return
        
        # 3. データベース接続テスト
        print("\n🗄️ 3. データベース接続テスト")
        try:
            db_manager = DatabaseManager(config.database_url)
            await db_manager.initialize()
            print("  ✅ データベース接続成功")
        except Exception as e:
            print(f"  ❌ データベース接続エラー: {e}")
            issues.append(f"データベース接続失敗: {e}")
            recommendations.append("データベース設定とファイルパスを確認してください")
            return
        
        # 4. エージェントマネージャー初期化テスト
        print("\n🤖 4. エージェントマネージャー初期化テスト")
        try:
            agent_manager = AgentManager(config, db_manager)
            print("  ✅ エージェントマネージャー初期化成功")
        except Exception as e:
            print(f"  ❌ エージェントマネージャー初期化エラー: {e}")
            issues.append(f"エージェントマネージャー初期化失敗: {e}")
            recommendations.append("AgentManagerクラスの実装を確認してください")
            await db_manager.close()
            return
        
        # 5. 自己編集メソッド存在確認
        print("\n✏️ 5. 自己編集メソッド存在確認")
        if hasattr(agent_manager, '_handle_self_edit'):
            print("  ✅ _handle_self_edit メソッドが存在")
        else:
            print("  ❌ _handle_self_edit メソッドが存在しない")
            issues.append("_handle_self_edit メソッドが実装されていない")
            recommendations.append("AgentManagerに_handle_self_editメソッドを実装してください")
        
        # 6. 意図分析メソッド存在確認
        print("\n🎯 6. 意図分析メソッド存在確認")
        if hasattr(agent_manager, '_analyze_intent'):
            print("  ✅ _analyze_intent メソッドが存在")
        else:
            print("  ❌ _analyze_intent メソッドが存在しない")
            issues.append("_analyze_intent メソッドが実装されていない")
            recommendations.append("AgentManagerに_analyze_intentメソッドを実装してください")
        
        # 7. ファイルツール初期化テスト
        print("\n📁 7. ファイルツール初期化テスト")
        try:
            project_root = os.getcwd()
            file_tool = FileTool(project_root=project_root)
            await file_tool.initialize()
            print("  ✅ ファイルツール初期化成功")
            
            # ファイルツールをエージェントに追加
            agent_manager.tools = getattr(agent_manager, 'tools', {})
            agent_manager.tools['file'] = file_tool
            
        except Exception as e:
            print(f"  ❌ ファイルツール初期化エラー: {e}")
            issues.append(f"ファイルツール初期化失敗: {e}")
            recommendations.append("FileToolクラスの実装を確認してください")
        
        # 8. 学習ツール初期化テスト
        print("\n📚 8. 学習ツール初期化テスト")
        try:
            # ダミーOllamaクライアント作成
            class DummyOllama:
                def __init__(self, config):
                    self.config = config
                    self.base_url = config.get('base_url', '')
                    self.model = config.get('model', '')
                    self.session = None

                async def initialize(self):
                    return

                async def close(self):
                    return

                async def generate(self, *args, **kwargs):
                    return "0.5"

                async def generate_response(self, *args, **kwargs):
                    return "テスト応答"

            dummy_ollama = DummyOllama(config.ollama_config)
            learning_tool = LearningTool(
                db_manager=db_manager,
                config=config,
                ollama_client=dummy_ollama,
                agent_manager=agent_manager
            )
            agent_manager.learning_tool = learning_tool
            print("  ✅ 学習ツール初期化成功")
            
        except Exception as e:
            print(f"  ❌ 学習ツール初期化エラー: {e}")
            issues.append(f"学習ツール初期化失敗: {e}")
            recommendations.append("LearningToolクラスの実装を確認してください")
        
        # 9. 簡単な自己編集テスト
        print("\n✏️ 9. 簡単な自己編集テスト")
        if hasattr(agent_manager, '_handle_self_edit'):
            try:
                test_file = "test_quick_self_edit.txt"
                test_content = f"クイックテスト - {datetime.now().isoformat()}"
                
                # ファイル書き込みテスト
                write_cmd = f"write file {test_file}\n{test_content}"
                write_result = await agent_manager._handle_self_edit(write_cmd, {})
                
                # ファイル存在確認
                if os.path.exists(test_file):
                    with open(test_file, 'r', encoding='utf-8') as f:
                        actual_content = f.read().strip()
                    
                    if actual_content == test_content:
                        print("  ✅ ファイル書き込み・読み取りテスト成功")
                    else:
                        print(f"  ⚠️ ファイル内容不一致")
                        issues.append("ファイル書き込み時の内容が期待値と異なる")
                    
                    # クリーンアップ
                    os.remove(test_file)
                else:
                    print("  ❌ ファイルが作成されていない")
                    issues.append("ファイル書き込み機能が動作していない")
                    recommendations.append("FileToolの書き込み機能を確認してください")
                    
            except Exception as e:
                print(f"  ❌ 自己編集テストエラー: {e}")
                issues.append(f"自己編集テストでエラー: {e}")
        else:
            print("  ⏭️ _handle_self_edit メソッドが存在しないためスキップ")
        
        # 10. 学習データアクセステスト
        print("\n📊 10. 学習データアクセステスト")
        if hasattr(agent_manager, 'learning_tool') and agent_manager.learning_tool:
            try:
                learning_data = await agent_manager.learning_tool.get_learning_data(limit=1)
                
                if learning_data.get('status') == 'success':
                    print("  ✅ 学習データアクセス成功")
                else:
                    print(f"  ⚠️ 学習データアクセス部分的成功: {learning_data.get('message', '')}")
                    issues.append("学習データアクセス機能に問題がある可能性")
                    
            except Exception as e:
                print(f"  ❌ 学習データアクセスエラー: {e}")
                issues.append(f"学習データアクセスでエラー: {e}")
                recommendations.append("LearningToolの実装を確認してください")
        else:
            print("  ⏭️ 学習ツールが利用できないためスキップ")
        
        # クリーンアップ
        try:
            if 'file_tool' in locals():
                await file_tool.close()
            await db_manager.close()
        except Exception as e:
            print(f"  ⚠️ クリーンアップエラー: {e}")
        
        # 結果サマリー
        print(f"\n{'='*60}")
        print("🔍 クイック診断結果サマリー")
        print(f"{'='*60}")
        
        if not issues:
            print("✅ 問題は検出されませんでした！")
            print("自己学習機能は正常に動作している可能性が高いです。")
        else:
            print(f"⚠️ {len(issues)}個の問題が検出されました:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            
            if recommendations:
                print(f"\n💡 推奨事項:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ クイック診断でエラー: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """メイン関数"""
    print("🤖 自己学習機能クイック診断ツール")
    print("=" * 50)
    
    await quick_test()
    
    print("\n🎉 クイック診断完了！")
    print("\nより詳細な診断を実行するには:")
    print("  python self_learning_diagnostic_test.py")


if __name__ == "__main__":
    asyncio.run(main())