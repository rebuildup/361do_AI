#!/usr/bin/env python3
"""
Self-Learning Diagnostic Test
自己学習機能の詳細診断テスト - 問題の特定と解決策の提案
"""

import asyncio
import json
import sys
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Tests should not contact a local ollama daemon; force skip mode
os.environ.setdefault('AGENT_SKIP_OLLAMA', '1')

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient
from agent.core.agent_manager import AgentManager
from agent.tools.file_tool import FileTool
from agent.tools.learning_tool import LearningTool


class SelfLearningDiagnostic:
    """自己学習機能診断クラス"""

    def __init__(self):
        self.config = None
        self.db_manager = None
        self.agent_manager = None
        self.file_tool = None
        self.learning_tool = None
        self.project_root = os.getcwd()
        self.test_results = []
        self.issues_found = []
        self.recommendations = []

    async def initialize(self):
        """システム初期化と基本チェック"""
        print("[DIAGNOSTIC] 自己学習機能診断システム初期化中...")
        
        try:
            # 設定初期化
            self.config = Config()
            print(f"✅ 設定読み込み完了")
            
            # データベース初期化
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            print(f"✅ データベース接続完了")
            
            # エージェントマネージャー初期化
            self.agent_manager = AgentManager(self.config, self.db_manager)
            
            # ファイルツール手動初期化
            self.file_tool = FileTool(project_root=self.project_root)
            await self.file_tool.initialize()
            self.agent_manager.tools['file'] = self.file_tool
            print(f"✅ ファイルツール初期化完了")
            
            # 学習ツール初期化
            try:
                # ダミーOllamaクライアント作成（テスト用）
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

                dummy_ollama = DummyOllama(self.config.ollama_config)
                self.learning_tool = LearningTool(
                    db_manager=self.db_manager,
                    config=self.config,
                    ollama_client=dummy_ollama,
                    agent_manager=self.agent_manager
                )
                self.agent_manager.learning_tool = self.learning_tool
                print(f"✅ 学習ツール初期化完了")
                
            except Exception as e:
                print(f"⚠️ 学習ツール初期化エラー: {e}")
                self.issues_found.append(f"学習ツール初期化失敗: {e}")
            
            print("[DIAGNOSTIC] 初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            self.issues_found.append(f"システム初期化失敗: {e}")
            return False

    async def shutdown(self):
        """システム終了処理"""
        print("[DIAGNOSTIC] システム終了中...")
        
        try:
            if self.file_tool:
                await self.file_tool.close()
            
            if self.agent_manager:
                await self.agent_manager.shutdown()
            
            if self.db_manager:
                await self.db_manager.close()
                
        except Exception as e:
            print(f"⚠️ 終了処理エラー: {e}")
        
        print("[DIAGNOSTIC] 終了完了")

    async def diagnose_self_edit_functionality(self) -> Dict[str, Any]:
        """自己編集機能の詳細診断"""
        print("\n[DIAGNOSTIC] 自己編集機能診断開始...")
        
        results = {
            'test_name': 'self_edit_functionality',
            'tests': [],
            'issues': [],
            'recommendations': []
        }
        
        # テスト1: _handle_self_edit メソッドの存在確認
        print("  📋 テスト1: _handle_self_edit メソッド存在確認")
        try:
            has_method = hasattr(self.agent_manager, '_handle_self_edit')
            if has_method:
                print("    ✅ _handle_self_edit メソッドが存在")
                results['tests'].append({
                    'name': 'handle_self_edit_method_exists',
                    'status': 'success',
                    'details': 'メソッドが存在'
                })
            else:
                print("    ❌ _handle_self_edit メソッドが存在しない")
                results['tests'].append({
                    'name': 'handle_self_edit_method_exists',
                    'status': 'failed',
                    'details': 'メソッドが存在しない'
                })
                results['issues'].append('_handle_self_edit メソッドが実装されていない')
                results['recommendations'].append('AgentManagerに_handle_self_editメソッドを実装する必要があります')
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'handle_self_edit_method_exists',
                'status': 'error',
                'details': str(e)
            })
        
        # テスト2: ファイル書き込みテスト
        print("  📋 テスト2: ファイル書き込み機能テスト")
        test_file_path = "src/data/prompts/diagnostic_test.txt"
        test_content = f"診断テスト - {datetime.now().isoformat()}"
        
        try:
            if hasattr(self.agent_manager, '_handle_self_edit'):
                write_cmd = f"write file {test_file_path}\n{test_content}"
                write_result = await self.agent_manager._handle_self_edit(write_cmd, {})
                
                # ファイルが実際に作成されたか確認
                full_path = os.path.join(self.project_root, test_file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        actual_content = f.read().strip()
                    
                    if actual_content == test_content:
                        print("    ✅ ファイル書き込み成功")
                        results['tests'].append({
                            'name': 'file_write_test',
                            'status': 'success',
                            'details': f'ファイル正常作成: {test_file_path}'
                        })
                    else:
                        print(f"    ⚠️ ファイル内容不一致: 期待='{test_content}', 実際='{actual_content}'")
                        results['tests'].append({
                            'name': 'file_write_test',
                            'status': 'partial',
                            'details': 'ファイル作成されたが内容が不一致'
                        })
                        results['issues'].append('ファイル書き込み時の内容が期待値と異なる')
                else:
                    print("    ❌ ファイルが作成されていない")
                    results['tests'].append({
                        'name': 'file_write_test',
                        'status': 'failed',
                        'details': 'ファイルが作成されていない'
                    })
                    results['issues'].append('ファイル書き込み機能が動作していない')
                    results['recommendations'].append('FileToolの書き込み機能を確認してください')
            else:
                print("    ⏭️ _handle_self_edit メソッドが存在しないためスキップ")
                results['tests'].append({
                    'name': 'file_write_test',
                    'status': 'skipped',
                    'details': '_handle_self_edit メソッドが存在しない'
                })
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'file_write_test',
                'status': 'error',
                'details': str(e)
            })
            results['issues'].append(f'ファイル書き込みテストでエラー: {e}')
        
        # テスト3: ファイル読み取りテスト
        print("  📋 テスト3: ファイル読み取り機能テスト")
        try:
            if hasattr(self.agent_manager, '_handle_self_edit') and os.path.exists(os.path.join(self.project_root, test_file_path)):
                read_cmd = f"read file {test_file_path}"
                read_result = await self.agent_manager._handle_self_edit(read_cmd, {})
                
                if test_content in str(read_result):
                    print("    ✅ ファイル読み取り成功")
                    results['tests'].append({
                        'name': 'file_read_test',
                        'status': 'success',
                        'details': 'ファイル内容正常読み取り'
                    })
                else:
                    print(f"    ⚠️ 読み取り内容不一致: {read_result}")
                    results['tests'].append({
                        'name': 'file_read_test',
                        'status': 'partial',
                        'details': '読み取り結果が期待値と異なる'
                    })
                    results['issues'].append('ファイル読み取り結果が期待値と異なる')
            else:
                print("    ⏭️ 前提条件が満たされていないためスキップ")
                results['tests'].append({
                    'name': 'file_read_test',
                    'status': 'skipped',
                    'details': '前提条件未満足'
                })
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'file_read_test',
                'status': 'error',
                'details': str(e)
            })
        
        # テスト4: プロンプト更新テスト
        print("  📋 テスト4: プロンプト更新機能テスト")
        try:
            if hasattr(self.agent_manager, '_handle_self_edit') and self.learning_tool:
                # まずテスト用プロンプトを追加
                test_prompt_name = "diagnostic_test_prompt"
                initial_content = "初期プロンプト内容"
                
                await self.learning_tool.add_prompt_template(
                    name=test_prompt_name,
                    content=initial_content,
                    description="診断テスト用プロンプト"
                )
                
                # プロンプト更新テスト
                updated_content = "更新されたプロンプト内容"
                update_cmd = f"update prompt {test_prompt_name}: {updated_content}"
                update_result = await self.agent_manager._handle_self_edit(update_cmd, {})
                
                # データベースから更新されたプロンプトを確認
                updated_prompt = await self.db_manager.get_prompt_template_by_name(test_prompt_name)
                
                if updated_prompt and updated_content in updated_prompt.get('template_content', ''):
                    print("    ✅ プロンプト更新成功")
                    results['tests'].append({
                        'name': 'prompt_update_test',
                        'status': 'success',
                        'details': 'プロンプト正常更新'
                    })
                else:
                    print("    ❌ プロンプト更新失敗")
                    results['tests'].append({
                        'name': 'prompt_update_test',
                        'status': 'failed',
                        'details': 'プロンプト更新が反映されていない'
                    })
                    results['issues'].append('プロンプト更新機能が動作していない')
                    results['recommendations'].append('プロンプト更新処理の実装を確認してください')
            else:
                print("    ⏭️ 前提条件が満たされていないためスキップ")
                results['tests'].append({
                    'name': 'prompt_update_test',
                    'status': 'skipped',
                    'details': '前提条件未満足'
                })
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'prompt_update_test',
                'status': 'error',
                'details': str(e)
            })
            results['issues'].append(f'プロンプト更新テストでエラー: {e}')
        
        # テスト5: 学習データ追加テスト
        print("  📋 テスト5: 学習データ追加機能テスト")
        try:
            if hasattr(self.agent_manager, '_handle_self_edit') and self.learning_tool:
                test_learning_content = "診断テスト用学習データ"
                add_cmd = f"add learning data: {test_learning_content}"
                add_result = await self.agent_manager._handle_self_edit(add_cmd, {})
                
                # データベースから追加された学習データを確認
                learning_items = await self.db_manager.get_learning_data(category='custom', limit=10)
                found = any(test_learning_content in item.get('content', '') for item in learning_items)
                
                if found:
                    print("    ✅ 学習データ追加成功")
                    results['tests'].append({
                        'name': 'learning_data_add_test',
                        'status': 'success',
                        'details': '学習データ正常追加'
                    })
                else:
                    print("    ❌ 学習データ追加失敗")
                    results['tests'].append({
                        'name': 'learning_data_add_test',
                        'status': 'failed',
                        'details': '学習データが追加されていない'
                    })
                    results['issues'].append('学習データ追加機能が動作していない')
                    results['recommendations'].append('学習データ追加処理の実装を確認してください')
            else:
                print("    ⏭️ 前提条件が満たされていないためスキップ")
                results['tests'].append({
                    'name': 'learning_data_add_test',
                    'status': 'skipped',
                    'details': '前提条件未満足'
                })
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'learning_data_add_test',
                'status': 'error',
                'details': str(e)
            })
        
        # クリーンアップ
        try:
            test_file_full_path = os.path.join(self.project_root, test_file_path)
            if os.path.exists(test_file_full_path):
                os.remove(test_file_full_path)
        except Exception as e:
            print(f"    ⚠️ クリーンアップエラー: {e}")
        
        return results

    async def diagnose_learning_data_access(self) -> Dict[str, Any]:
        """学習データアクセス機能の診断"""
        print("\n[DIAGNOSTIC] 学習データアクセス機能診断開始...")
        
        results = {
            'test_name': 'learning_data_access',
            'tests': [],
            'issues': [],
            'recommendations': []
        }
        
        # テスト1: 学習ツールの存在確認
        print("  📋 テスト1: 学習ツール存在確認")
        if self.learning_tool:
            print("    ✅ 学習ツールが利用可能")
            results['tests'].append({
                'name': 'learning_tool_availability',
                'status': 'success',
                'details': '学習ツール利用可能'
            })
        else:
            print("    ❌ 学習ツールが利用できない")
            results['tests'].append({
                'name': 'learning_tool_availability',
                'status': 'failed',
                'details': '学習ツール未初期化'
            })
            results['issues'].append('学習ツールが初期化されていない')
            results['recommendations'].append('LearningToolの初期化処理を確認してください')
            return results
        
        # テスト2: 学習データ取得テスト
        print("  📋 テスト2: 学習データ取得テスト")
        try:
            learning_data = await self.learning_tool.get_learning_data(limit=5)
            
            if learning_data.get('status') == 'success':
                data_count = len(learning_data.get('data', []))
                print(f"    ✅ 学習データ取得成功: {data_count}件")
                results['tests'].append({
                    'name': 'get_learning_data',
                    'status': 'success',
                    'details': f'{data_count}件のデータを取得'
                })
            else:
                print(f"    ❌ 学習データ取得失敗: {learning_data.get('message', '')}")
                results['tests'].append({
                    'name': 'get_learning_data',
                    'status': 'failed',
                    'details': learning_data.get('message', '')
                })
                results['issues'].append('学習データ取得機能が動作していない')
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'get_learning_data',
                'status': 'error',
                'details': str(e)
            })
            results['issues'].append(f'学習データ取得でエラー: {e}')
        
        # テスト3: 学習状態取得テスト
        print("  📋 テスト3: 学習状態取得テスト")
        try:
            learning_status = await self.learning_tool.get_learning_status()
            
            if learning_status.get('status') == 'success':
                print(f"    ✅ 学習状態取得成功")
                results['tests'].append({
                    'name': 'get_learning_status',
                    'status': 'success',
                    'details': '学習状態正常取得'
                })
            else:
                print(f"    ❌ 学習状態取得失敗: {learning_status.get('message', '')}")
                results['tests'].append({
                    'name': 'get_learning_status',
                    'status': 'failed',
                    'details': learning_status.get('message', '')
                })
                results['issues'].append('学習状態取得機能が動作していない')
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'get_learning_status',
                'status': 'error',
                'details': str(e)
            })
        
        # テスト4: カスタム学習データ追加テスト
        print("  📋 テスト4: カスタム学習データ追加テスト")
        try:
            test_content = f"診断テスト学習データ - {datetime.now().isoformat()}"
            add_result = await self.learning_tool.add_custom_learning_data(
                content=test_content,
                category="diagnostic_test",
                tags=["diagnostic", "test"]
            )
            
            if add_result.get('status') == 'success':
                print(f"    ✅ カスタム学習データ追加成功")
                results['tests'].append({
                    'name': 'add_custom_learning_data',
                    'status': 'success',
                    'details': 'カスタム学習データ正常追加'
                })
            else:
                print(f"    ❌ カスタム学習データ追加失敗: {add_result.get('message', '')}")
                results['tests'].append({
                    'name': 'add_custom_learning_data',
                    'status': 'failed',
                    'details': add_result.get('message', '')
                })
                results['issues'].append('カスタム学習データ追加機能が動作していない')
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'add_custom_learning_data',
                'status': 'error',
                'details': str(e)
            })
        
        return results

    async def diagnose_intent_analysis(self) -> Dict[str, Any]:
        """意図分析機能の診断"""
        print("\n[DIAGNOSTIC] 意図分析機能診断開始...")
        
        results = {
            'test_name': 'intent_analysis',
            'tests': [],
            'issues': [],
            'recommendations': []
        }
        
        # テスト用ケース
        test_cases = [
            {
                'input': '学習データの統計を表示',
                'expected_intent': 'learning_data_access',
                'description': '学習データアクセス意図'
            },
            {
                'input': 'read file test.txt',
                'expected_intent': 'file_operation',
                'description': 'ファイル操作意図'
            },
            {
                'input': 'write file output.txt\ncontent',
                'expected_intent': 'file_operation',
                'description': 'ファイル書き込み意図'
            },
            {
                'input': 'update prompt test: content',
                'expected_intent': 'file_operation',
                'description': 'プロンプト更新意図'
            }
        ]
        
        # テスト1: _analyze_intent メソッドの存在確認
        print("  📋 テスト1: _analyze_intent メソッド存在確認")
        try:
            has_method = hasattr(self.agent_manager, '_analyze_intent')
            if has_method:
                print("    ✅ _analyze_intent メソッドが存在")
                results['tests'].append({
                    'name': 'analyze_intent_method_exists',
                    'status': 'success',
                    'details': 'メソッドが存在'
                })
            else:
                print("    ❌ _analyze_intent メソッドが存在しない")
                results['tests'].append({
                    'name': 'analyze_intent_method_exists',
                    'status': 'failed',
                    'details': 'メソッドが存在しない'
                })
                results['issues'].append('_analyze_intent メソッドが実装されていない')
                results['recommendations'].append('AgentManagerに_analyze_intentメソッドを実装する必要があります')
                return results
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'analyze_intent_method_exists',
                'status': 'error',
                'details': str(e)
            })
            return results
        
        # テスト2: 各テストケースの意図分析
        for i, test_case in enumerate(test_cases, 1):
            print(f"  📋 テスト{i+1}: {test_case['description']}")
            try:
                context = []
                intent = await self.agent_manager._analyze_intent(test_case['input'], context)
                
                detected_intent = intent.get('primary_intent', 'unknown')
                expected_intent = test_case['expected_intent']
                confidence = intent.get('confidence', 0)
                
                if detected_intent == expected_intent:
                    print(f"    ✅ 意図分析成功: {detected_intent} (信頼度: {confidence})")
                    results['tests'].append({
                        'name': f'intent_analysis_{i}',
                        'status': 'success',
                        'details': f'期待: {expected_intent}, 検出: {detected_intent}, 信頼度: {confidence}'
                    })
                else:
                    print(f"    ⚠️ 意図分析不一致: 期待={expected_intent}, 検出={detected_intent}")
                    results['tests'].append({
                        'name': f'intent_analysis_{i}',
                        'status': 'partial',
                        'details': f'期待: {expected_intent}, 検出: {detected_intent}, 信頼度: {confidence}'
                    })
                    results['issues'].append(f'意図分析が不正確: {test_case["input"]}')
                    
            except Exception as e:
                print(f"    ❌ エラー: {e}")
                results['tests'].append({
                    'name': f'intent_analysis_{i}',
                    'status': 'error',
                    'details': str(e)
                })
                results['issues'].append(f'意図分析でエラー: {e}')
        
        return results

    async def diagnose_database_integration(self) -> Dict[str, Any]:
        """データベース統合機能の診断"""
        print("\n[DIAGNOSTIC] データベース統合機能診断開始...")
        
        results = {
            'test_name': 'database_integration',
            'tests': [],
            'issues': [],
            'recommendations': []
        }
        
        # テスト1: データベース接続確認
        print("  📋 テスト1: データベース接続確認")
        try:
            if self.db_manager:
                print("    ✅ データベースマネージャーが利用可能")
                results['tests'].append({
                    'name': 'database_connection',
                    'status': 'success',
                    'details': 'データベース接続正常'
                })
            else:
                print("    ❌ データベースマネージャーが利用できない")
                results['tests'].append({
                    'name': 'database_connection',
                    'status': 'failed',
                    'details': 'データベース未接続'
                })
                results['issues'].append('データベース接続が確立されていない')
                return results
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'database_connection',
                'status': 'error',
                'details': str(e)
            })
            return results
        
        # テスト2: 学習統計取得テスト
        print("  📋 テスト2: 学習統計取得テスト")
        try:
            stats = await self.db_manager.get_learning_statistics()
            
            if isinstance(stats, dict):
                print(f"    ✅ 学習統計取得成功: {stats.get('total_learning_data', 0)}件")
                results['tests'].append({
                    'name': 'get_learning_statistics',
                    'status': 'success',
                    'details': f"学習データ: {stats.get('total_learning_data', 0)}件"
                })
            else:
                print("    ❌ 学習統計取得失敗")
                results['tests'].append({
                    'name': 'get_learning_statistics',
                    'status': 'failed',
                    'details': '統計データが取得できない'
                })
                results['issues'].append('学習統計取得機能が動作していない')
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'get_learning_statistics',
                'status': 'error',
                'details': str(e)
            })
        
        # テスト3: プロンプトテンプレート取得テスト
        print("  📋 テスト3: プロンプトテンプレート取得テスト")
        try:
            system_prompt = await self.db_manager.get_prompt_template("system_prompt")
            
            if system_prompt:
                print("    ✅ システムプロンプト取得成功")
                results['tests'].append({
                    'name': 'get_prompt_template',
                    'status': 'success',
                    'details': 'システムプロンプト存在'
                })
            else:
                print("    ⚠️ システムプロンプトが存在しない")
                results['tests'].append({
                    'name': 'get_prompt_template',
                    'status': 'partial',
                    'details': 'システムプロンプト不存在'
                })
                results['issues'].append('システムプロンプトが設定されていない')
                results['recommendations'].append('システムプロンプトを設定してください')
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results['tests'].append({
                'name': 'get_prompt_template',
                'status': 'error',
                'details': str(e)
            })
        
        return results

    async def run_comprehensive_diagnostic(self) -> Dict[str, Any]:
        """包括的診断実行"""
        print("[DIAGNOSTIC] 自己学習機能包括診断開始...")
        
        diagnostic_start_time = datetime.now()
        all_results = {}
        
        # 各診断を実行
        diagnostic_functions = [
            self.diagnose_database_integration,
            self.diagnose_intent_analysis,
            self.diagnose_learning_data_access,
            self.diagnose_self_edit_functionality
        ]
        
        for diagnostic_func in diagnostic_functions:
            try:
                result = await diagnostic_func()
                all_results[result['test_name']] = result
                
                # 問題と推奨事項を統合
                self.issues_found.extend(result.get('issues', []))
                self.recommendations.extend(result.get('recommendations', []))
                
            except Exception as e:
                print(f"❌ 診断 {diagnostic_func.__name__} でエラー: {e}")
                all_results[diagnostic_func.__name__] = {
                    'test_name': diagnostic_func.__name__,
                    'error': str(e),
                    'success': False
                }
        
        diagnostic_end_time = datetime.now()
        diagnostic_duration = (diagnostic_end_time - diagnostic_start_time).total_seconds()
        
        # 包括的診断結果
        comprehensive_result = {
            'start_time': diagnostic_start_time.isoformat(),
            'end_time': diagnostic_end_time.isoformat(),
            'duration_seconds': diagnostic_duration,
            'diagnostics_executed': len(all_results),
            'results': all_results,
            'summary': {
                'total_issues': len(self.issues_found),
                'total_recommendations': len(self.recommendations),
                'issues_found': list(set(self.issues_found)),  # 重複除去
                'recommendations': list(set(self.recommendations))  # 重複除去
            }
        }
        
        return comprehensive_result

    def print_diagnostic_summary(self, results: Dict[str, Any]):
        """診断結果サマリー表示"""
        print(f"\n{'='*80}")
        print(f"[DIAGNOSTIC] 自己学習機能診断結果サマリー")
        print(f"{'='*80}")
        print(f"実行開始: {results['start_time']}")
        print(f"実行終了: {results['end_time']}")
        print(f"総実行時間: {results['duration_seconds']:.2f}秒")
        print(f"実行診断数: {results['diagnostics_executed']}")
        
        summary = results.get('summary', {})
        print(f"\n[SUMMARY] 診断サマリー:")
        print(f"  🔍 発見された問題数: {summary.get('total_issues', 0)}")
        print(f"  💡 推奨事項数: {summary.get('total_recommendations', 0)}")
        
        # 診断別結果
        print(f"\n[RESULTS] 診断別結果:")
        for test_name, test_data in results['results'].items():
            if 'tests' in test_data:
                tests = test_data['tests']
                success_count = sum(1 for t in tests if t.get('status') == 'success')
                failed_count = sum(1 for t in tests if t.get('status') == 'failed')
                error_count = sum(1 for t in tests if t.get('status') == 'error')
                
                print(f"\n  📋 {test_name}:")
                print(f"     総テスト数: {len(tests)}")
                print(f"     成功: {success_count}")
                print(f"     失敗: {failed_count}")
                print(f"     エラー: {error_count}")
                
                # 失敗したテストの詳細
                if failed_count > 0 or error_count > 0:
                    failed_tests = [t for t in tests if t.get('status') in ['failed', 'error']]
                    print(f"     問題詳細:")
                    for failed in failed_tests[:3]:  # 最大3件まで表示
                        status = failed.get('status', 'unknown')
                        name = failed.get('name', 'unknown')
                        details = failed.get('details', 'No details')
                        print(f"       - [{status.upper()}] {name}: {details}")
        
        # 発見された問題
        if summary.get('issues_found'):
            print(f"\n[ISSUES] 発見された問題:")
            for i, issue in enumerate(summary['issues_found'], 1):
                print(f"  {i}. {issue}")
        
        # 推奨事項
        if summary.get('recommendations'):
            print(f"\n[RECOMMENDATIONS] 推奨事項:")
            for i, recommendation in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {recommendation}")
        
        print(f"{'='*80}")

    def save_diagnostic_results(self, results: Dict[str, Any], filename: str = None):
        """診断結果保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"self_learning_diagnostic_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"[DIAGNOSTIC] 診断結果を保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ 診断結果保存エラー: {e}")
            return None


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自己学習機能診断システム")
    parser.add_argument("--diagnostic", choices=[
        'self_edit', 'learning_access', 'intent', 'database', 'all'
    ], default='all', help="実行する診断種類")
    parser.add_argument("--output", help="結果出力ファイル名")
    parser.add_argument("--no-save", action="store_true", help="結果を保存しない")
    
    args = parser.parse_args()
    
    diagnostic = SelfLearningDiagnostic()
    
    try:
        if await diagnostic.initialize():
            if args.diagnostic == 'all':
                results = await diagnostic.run_comprehensive_diagnostic()
                diagnostic.print_diagnostic_summary(results)
            else:
                # 個別診断実行
                diagnostic_map = {
                    'self_edit': diagnostic.diagnose_self_edit_functionality,
                    'learning_access': diagnostic.diagnose_learning_data_access,
                    'intent': diagnostic.diagnose_intent_analysis,
                    'database': diagnostic.diagnose_database_integration
                }
                
                if args.diagnostic in diagnostic_map:
                    result = await diagnostic_map[args.diagnostic]()
                    print(f"\n[RESULT] {result['test_name']} 診断完了")
                    print(f"問題数: {len(result.get('issues', []))}")
                    print(f"推奨事項数: {len(result.get('recommendations', []))}")
            
            # 結果保存
            if not args.no_save and 'results' in locals():
                diagnostic.save_diagnostic_results(results, args.output)
            
        else:
            print("❌ システム初期化に失敗しました")
            
    except KeyboardInterrupt:
        print("\n[DIAGNOSTIC] ユーザーによって中断されました")
    except Exception as e:
        print(f"❌ 診断実行エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await diagnostic.shutdown()


if __name__ == "__main__":
    asyncio.run(main())