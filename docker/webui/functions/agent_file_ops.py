"""
title: AI Agent File Operations
author: AI Agent Studio
version: 1.0.0
description: AI エージェントを使用したファイル操作機能
"""

import json
import requests
from typing import Dict, Any
from pydantic import BaseModel, Field


class FileOperationRequest(BaseModel):
    operation: str = Field(..., description="操作タイプ (read, write, list)")
    path: str = Field(..., description="ファイル/ディレクトリパス")
    content: str = Field("", description="書き込み内容（write操作の場合）")


class Tools:
    def __init__(self):
        self.agent_base_url = "http://agent:8000"
    
    def file_operation(self, operation: str, path: str, content: str = "") -> str:
        """
        AI エージェントを使用してファイル操作を実行
        
        Args:
            operation: 操作タイプ (read, write, list)
            path: ファイル/ディレクトリパス
            content: 書き込み内容（write操作の場合）
            
        Returns:
            操作結果のテキスト
        """
        try:
            # 操作に応じてメッセージを構築
            if operation == "read":
                message = f"ファイル読み込み: {path}"
            elif operation == "write":
                message = f"ファイル書き込み: {path}\n内容:\n{content}"
            elif operation == "list":
                message = f"ディレクトリ一覧: {path}"
            else:
                return f"未対応の操作: {operation}"
            
            # エージェントAPIにリクエストを送信
            response = requests.post(
                f"{self.agent_base_url}/api/chat",
                json={
                    "message": message,
                    "session_id": "webui_file_ops"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "ファイル操作を実行できませんでした。")
            else:
                return f"ファイル操作エラー: HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"ファイル操作エラー: {str(e)}"
        except Exception as e:
            return f"予期しないエラー: {str(e)}"


# Tools インスタンスを作成
tools = Tools()
