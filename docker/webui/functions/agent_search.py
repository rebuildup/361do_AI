"""
title: AI Agent Web Search
author: AI Agent Studio
version: 1.0.0
description: AI エージェントを使用したWeb検索機能
"""

import json
import requests
from typing import Dict, Any
from pydantic import BaseModel, Field


class WebSearchRequest(BaseModel):
    query: str = Field(..., description="検索クエリ")
    max_results: int = Field(5, description="最大結果数")


class Tools:
    def __init__(self):
        self.agent_base_url = "http://agent:8000"
    
    def web_search(self, query: str, max_results: int = 5) -> str:
        """
        AI エージェントを使用してWeb検索を実行
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            
        Returns:
            検索結果のテキスト
        """
        try:
            # エージェントAPIに検索リクエストを送信
            response = requests.post(
                f"{self.agent_base_url}/api/chat",
                json={
                    "message": f"Web検索: {query}",
                    "session_id": "webui_search"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "検索結果を取得できませんでした。")
            else:
                return f"検索エラー: HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"検索エラー: {str(e)}"
        except Exception as e:
            return f"予期しないエラー: {str(e)}"


# Tools インスタンスを作成
tools = Tools()
