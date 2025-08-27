"""
Help System
エージェントの全機能に関するヘルプとガイダンスを提供
"""

from typing import Dict, List, Any
from agent.tools.tool_manager import ToolManager


class HelpSystem:
    """包括的ヘルプシステム"""

    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager

    def get_comprehensive_help(self) -> str:
        """包括的なヘルプ情報を取得"""
        help_text = """
🤖 AIエージェント - 全機能ガイド

## 📋 利用可能な機能

### 🔍 Web検索
- 「〜について調べて」「〜を検索して」
- 例: "最新のAI技術について調べて"

### 💻 システムコマンド実行  
- 「systeminfoを実行」「tasklistコマンド」
- 例: "システム情報を表示して"

### 📁 ファイル操作
- 「ファイルを読み取り」「ファイルに書き込み」
- 例: "read file data/test.txt"

### 🧠 学習システム
- 「学習データを表示」「最古のデータ」
- 例: "学習データの統計を教えて"

## 🎯 使用例
- "tasklistを実行してプロセス一覧を表示"
- "ボカロ 最新情報を検索"
- "学習データの中で一番古いものを教えて"

## ❓ ヘルプコマンド
- help tools - ツール一覧
- help examples - 使用例
- help commands - コマンド一覧
        """
        return help_text.strip()

    def get_tool_help(self) -> str:
        """ツール固有のヘルプ"""
        if not self.tool_manager:
            return "ツールマネージャーが利用できません。"

        tools = self.tool_manager.get_available_tools()
        help_text = "🛠️ 利用可能なツール:\n\n"

        for tool_name, info in tools.items():
            if info['available']:
                capability = info['capability']
                help_text += f"**{capability.get('name', tool_name)}**\n"
                help_text += f"  説明: {capability.get('description', '')}\n"
                help_text += f"  キーワード: {', '.join(capability.get('keywords', []))}\n\n"

        return help_text

    def get_examples(self) -> str:
        """使用例を取得"""
        return """
📝 使用例:

**コマンド実行:**
- "systeminfoコマンドを実行して"
- "現在のプロセス一覧を表示"
- "ディレクトリの内容を確認"

**Web検索:**
- "最新のAI技術について調べて"
- "Python プログラミング 情報を検索"

**ファイル操作:**
- "read file data/config.txt"
- "write file output.txt\\n新しい内容"

**学習システム:**
- "学習データの統計を表示"
- "一番古い学習データを教えて"
        """