#!/usr/bin/env python3
"""
Self-Learning Fix
自己学習機能の問題修正スクリプト
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class SelfLearningFix:
    """自己学習機能修正クラス"""
    
    def __init__(self):
        self.fixes_applied = []
        self.recommendations = []
    
    def analyze_issues(self):
        """問題分析"""
        print("🔍 自己学習機能の問題分析...")
        
        issues = [
            {
                'issue': '意図分析の不正確性',
                'description': '自己学習関連コマンドが正しく認識されない',
                'severity': 'high',
                'fix_method': 'improve_intent_analysis'
            },
            {
                'issue': 'ツール選択の問題',
                'description': '期待されるツールが使用されない',
                'severity': 'high',
                'fix_method': 'fix_tool_selection'
            },
            {
                'issue': '自己編集コマンド認識不良',
                'description': 'update prompt, add learning dataが正しく処理されない',
                'severity': 'critical',
                'fix_method': 'fix_self_edit_commands'
            },
            {
                'issue': 'システムプロンプト未設定',
                'description': 'システムプロンプトが設定されていない',
                'severity': 'medium',
                'fix_method': 'setup_system_prompt'
            }
        ]
        
        print(f"📊 {len(issues)}個の問題を特定:")
        for i, issue in enumerate(issues, 1):
            severity_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            print(f"  {i}. {severity_icon.get(issue['severity'], '⚪')} {issue['issue']}")
            print(f"     {issue['description']}")
        
        return issues
    
    def generate_intent_analysis_fix(self):
        """意図分析修正コード生成"""
        print("\n🔧 意図分析修正コード生成中...")
        
        fix_code = '''
# AgentManagerクラスの_analyze_intentメソッドを改善
def _analyze_intent_improved(self, message: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """改善された意図分析"""
    message_lower = message.lower().strip()
    
    # 自己学習関連の意図パターン
    learning_patterns = [
        r'学習データ.*統計',
        r'学習データ.*表示',
        r'学習データ.*一覧',
        r'学習データ.*取得',
        r'学習.*状態',
        r'最近.*学習データ',
        r'古い.*学習データ'
    ]
    
    # 自己編集関連の意図パターン
    self_edit_patterns = [
        r'^(write|read|append)\\s+file\\s+',
        r'^update\\s+prompt\\s+',
        r'^add\\s+learning\\s+data:',
        r'^optimize\\s+prompt\\s+'
    ]
    
    # ファイル操作パターン
    file_patterns = [
        r'^(write|read|append)\\s+file\\s+',
        r'ファイル.*書き込み',
        r'ファイル.*読み取り',
        r'ファイル.*作成'
    ]
    
    import re
    
    # パターンマッチング
    for pattern in learning_patterns:
        if re.search(pattern, message_lower):
            return {
                'primary_intent': 'learning_data_access',
                'confidence': 0.9,
                'suggested_tools': ['learning'],
                'reasoning': f'学習データ関連パターンにマッチ: {pattern}'
            }
    
    for pattern in self_edit_patterns:
        if re.search(pattern, message):
            return {
                'primary_intent': 'self_edit',
                'confidence': 0.95,
                'suggested_tools': ['file'],
                'reasoning': f'自己編集パターンにマッチ: {pattern}'
            }
    
    for pattern in file_patterns:
        if re.search(pattern, message):
            return {
                'primary_intent': 'file_operation',
                'confidence': 0.9,
                'suggested_tools': ['file'],
                'reasoning': f'ファイル操作パターンにマッチ: {pattern}'
            }
    
    # デフォルトの意図分析にフォールバック
    return self._analyze_intent_original(message, context)
'''
        
        return fix_code
    
    def generate_self_edit_fix(self):
        """自己編集機能修正コード生成"""
        print("🔧 自己編集機能修正コード生成中...")
        
        fix_code = '''
# AgentManagerクラスの_handle_self_editメソッドを改善
async def _handle_self_edit_improved(self, message: str, context: Dict[str, Any]) -> str:
    """改善された自己編集処理"""
    import re
    
    message = message.strip()
    
    # ファイル書き込みパターン
    write_match = re.match(r'^write\\s+file\\s+([^\\n]+)\\n(.*)$', message, re.DOTALL)
    if write_match:
        file_path = write_match.group(1).strip()
        content = write_match.group(2)
        
        if hasattr(self, 'tools') and 'file' in self.tools:
            try:
                result = await self.tools['file'].write_file(file_path, content)
                return f"ファイル '{file_path}' に正常に書き込みました。"
            except Exception as e:
                return f"ファイル書き込みエラー: {e}"
        else:
            return "ファイルツールが利用できません。"
    
    # ファイル読み取りパターン
    read_match = re.match(r'^read\\s+file\\s+(.+)$', message)
    if read_match:
        file_path = read_match.group(1).strip()
        
        if hasattr(self, 'tools') and 'file' in self.tools:
            try:
                content = await self.tools['file'].read_file(file_path)
                return content
            except Exception as e:
                return f"ファイル読み取りエラー: {e}"
        else:
            return "ファイルツールが利用できません。"
    
    # プロンプト更新パターン
    prompt_match = re.match(r'^update\\s+prompt\\s+([^:]+):\\s*(.*)$', message, re.DOTALL)
    if prompt_match:
        prompt_name = prompt_match.group(1).strip()
        prompt_content = prompt_match.group(2).strip()
        
        if hasattr(self, 'learning_tool') and self.learning_tool:
            try:
                # 既存のプロンプトを更新または新規作成
                result = await self.learning_tool.add_prompt_template(
                    name=prompt_name,
                    content=prompt_content,
                    description=f"自己編集により更新: {datetime.now().isoformat()}"
                )
                return f"プロンプト '{prompt_name}' を正常に更新しました。"
            except Exception as e:
                return f"プロンプト更新エラー: {e}"
        else:
            return "学習ツールが利用できません。"
    
    # 学習データ追加パターン
    learning_match = re.match(r'^add\\s+learning\\s+data:\\s*(.*)$', message, re.DOTALL)
    if learning_match:
        learning_content = learning_match.group(1).strip()
        
        if hasattr(self, 'learning_tool') and self.learning_tool:
            try:
                result = await self.learning_tool.add_custom_learning_data(
                    content=learning_content,
                    category="self_edit",
                    tags=["self_edit", "manual_addition"]
                )
                return f"学習データを正常に追加しました: {learning_content[:50]}..."
            except Exception as e:
                return f"学習データ追加エラー: {e}"
        else:
            return "学習ツールが利用できません。"
    
    return f"サポートされていない自己編集コマンドです: {message}"
'''
        
        return fix_code
    
    def generate_tool_selection_fix(self):
        """ツール選択修正コード生成"""
        print("🔧 ツール選択修正コード生成中...")
        
        fix_code = '''
# AgentManagerクラスのprocess_messageメソッドを改善
async def process_message_improved(self, message: str) -> Dict[str, Any]:
    """改善されたメッセージ処理"""
    start_time = time.time()
    
    try:
        # 意図分析
        intent = await self._analyze_intent_improved(message, [])
        primary_intent = intent.get('primary_intent', 'unknown')
        suggested_tools = intent.get('suggested_tools', [])
        
        # 自己編集コマンドの直接処理
        if primary_intent == 'self_edit':
            response = await self._handle_self_edit_improved(message, {})
            return {
                'response': response,
                'intent': intent,
                'tools_used': ['file'],
                'response_time': time.time() - start_time
            }
        
        # 学習データアクセスの直接処理
        if primary_intent == 'learning_data_access':
            if hasattr(self, 'learning_tool') and self.learning_tool:
                try:
                    if '統計' in message:
                        stats = await self.db_manager.get_learning_statistics()
                        response = f"学習データ統計:\\n"
                        response += f"- 総学習データ数: {stats.get('total_learning_data', 0)}件\\n"
                        response += f"- 知識アイテム数: {stats.get('total_knowledge_items', 0)}件\\n"
                        response += f"- 平均品質スコア: {stats.get('average_quality_score', 0):.2f}\\n"
                        response += f"- 高品質データ数: {stats.get('high_quality_count', 0)}件"
                    elif '一覧' in message or '表示' in message:
                        learning_data = await self.learning_tool.get_learning_data(limit=5)
                        if learning_data.get('status') == 'success':
                            data = learning_data.get('data', [])
                            response = f"最近の学習データ ({len(data)}件):\\n"
                            for i, item in enumerate(data, 1):
                                content = item.get('content', '')[:100]
                                category = item.get('category', 'unknown')
                                response += f"{i}. [{category}] {content}...\\n"
                        else:
                            response = "学習データの取得に失敗しました。"
                    else:
                        response = "学習データに関する具体的な要求を指定してください。"
                    
                    return {
                        'response': response,
                        'intent': intent,
                        'tools_used': ['learning'],
                        'response_time': time.time() - start_time
                    }
                except Exception as e:
                    return {
                        'response': f"学習データアクセスエラー: {e}",
                        'intent': intent,
                        'tools_used': [],
                        'response_time': time.time() - start_time
                    }
        
        # 通常の処理にフォールバック
        return await self.process_message_original(message)
        
    except Exception as e:
        return {
            'response': f"メッセージ処理エラー: {e}",
            'intent': {'primary_intent': 'error'},
            'tools_used': [],
            'response_time': time.time() - start_time
        }
'''
        
        return fix_code
    
    def create_system_prompt_fix(self):
        """システムプロンプト設定修正"""
        print("🔧 システムプロンプト設定修正中...")
        
        system_prompt = """あなたは高度な自己学習機能を持つAIエージェントです。

主な機能:
1. 学習データの管理と分析
2. ファイルの読み書き操作
3. プロンプトテンプレートの更新
4. 自己改善と最適化

自己編集コマンド:
- write file <path>\\n<content> : ファイル書き込み
- read file <path> : ファイル読み取り
- update prompt <name>: <content> : プロンプト更新
- add learning data: <content> : 学習データ追加

学習データアクセス:
- 学習データの統計表示
- 学習データの一覧表示
- 学習状態の確認

常に正確で有用な応答を提供し、ユーザーの要求に応じて適切なツールを使用してください。"""
        
        return system_prompt
    
    def generate_comprehensive_fix(self):
        """包括的修正コード生成"""
        print("\n🔧 包括的修正コード生成中...")
        
        fixes = {
            'intent_analysis': self.generate_intent_analysis_fix(),
            'self_edit': self.generate_self_edit_fix(),
            'tool_selection': self.generate_tool_selection_fix(),
            'system_prompt': self.create_system_prompt_fix()
        }
        
        return fixes
    
    def save_fixes_to_file(self, fixes):
        """修正コードをファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"self_learning_fixes_{timestamp}.py"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("#!/usr/bin/env python3\n")
                f.write('"""\n')
                f.write("Self-Learning Fixes\n")
                f.write("自己学習機能修正コード\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write('"""\n\n')
                
                for fix_name, fix_code in fixes.items():
                    f.write(f"# {fix_name.upper()} FIX\n")
                    f.write("# " + "="*50 + "\n")
                    f.write(fix_code)
                    f.write("\n\n")
            
            print(f"💾 修正コードを保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ 修正コード保存エラー: {e}")
            return None
    
    def print_recommendations(self):
        """推奨事項表示"""
        print("\n💡 修正推奨事項:")
        
        recommendations = [
            "1. AgentManagerクラスの_analyze_intentメソッドを改善版に置き換える",
            "2. _handle_self_editメソッドを改善版に置き換える",
            "3. process_messageメソッドに改善されたツール選択ロジックを追加",
            "4. システムプロンプトを設定してエージェントの動作を明確化",
            "5. 意図分析のパターンマッチングを強化",
            "6. 自己編集コマンドの正規表現パターンを改善",
            "7. エラーハンドリングを強化してより詳細なエラーメッセージを提供"
        ]
        
        for rec in recommendations:
            print(f"  {rec}")
        
        print("\n🔧 実装手順:")
        print("  1. 生成された修正コードを確認")
        print("  2. src/agent/core/agent_manager.py を修正")
        print("  3. システムプロンプトをデータベースに追加")
        print("  4. テストを実行して動作確認")
        print("  5. 必要に応じて追加調整")


async def main():
    """メイン関数"""
    print("🔧 自己学習機能修正システム")
    print("=" * 50)
    
    fix_system = SelfLearningFix()
    
    # 問題分析
    issues = fix_system.analyze_issues()
    
    # 修正コード生成
    fixes = fix_system.generate_comprehensive_fix()
    
    # 修正コード保存
    filename = fix_system.save_fixes_to_file(fixes)
    
    # 推奨事項表示
    fix_system.print_recommendations()
    
    print("\n🎉 修正コード生成完了！")
    if filename:
        print(f"📁 修正コードファイル: {filename}")
    
    print("\n次のステップ:")
    print("1. 生成された修正コードを確認")
    print("2. AgentManagerクラスに修正を適用")
    print("3. テストを再実行して改善を確認")


if __name__ == "__main__":
    asyncio.run(main())