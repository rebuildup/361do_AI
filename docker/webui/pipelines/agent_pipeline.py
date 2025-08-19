# -*- coding: utf-8 -*-
"""
AI Agent Pipeline for Open WebUI
エージェント機能をOpen WebUIに統合するパイプライン
"""

import json
import requests
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        AGENT_API_BASE_URL: str = "http://agent:8000"
        AGENT_API_KEY: str = ""

    def __init__(self):
        self.name = "AI Agent Pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        """パイプライン起動時の処理"""
        print(f"AI Agent Pipeline が起動しました")

    async def on_shutdown(self):
        """パイプライン終了時の処理"""
        print(f"AI Agent Pipeline が終了しました")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """メインのパイプライン処理"""
        
        # エージェント機能が必要かどうかを判定
        if self._needs_agent_function(user_message):
            return self._call_agent_api(user_message, messages)
        
        # 通常のLLM処理に委譲
        return self._call_ollama_direct(user_message, model_id, messages, body)

    def _needs_agent_function(self, message: str) -> bool:
        """エージェント機能が必要かどうかを判定"""
        agent_keywords = [
            "検索", "調べ", "探し", "search", "find",
            "ファイル", "file", "保存", "save", "読み込み", "load",
            "作成", "create", "生成", "generate",
            "webデザイン", "web design", "html", "css",
            "機能", "function", "できること", "help"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in agent_keywords)

    def _call_agent_api(self, user_message: str, messages: List[dict]) -> str:
        """エージェントAPIを呼び出し"""
        try:
            # セッションIDを生成（メッセージ履歴から）
            session_id = f"webui_{hash(str(messages))}"
            
            response = requests.post(
                f"{self.valves.AGENT_API_BASE_URL}/api/chat",
                json={
                    "message": user_message,
                    "session_id": session_id
                },
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                agent_response = result.get("response", "エージェント機能からの応答を取得できませんでした。")
                
                # エージェント機能の詳細情報を含める
                if result.get("intent"):
                    agent_response += f"\n\n*[検出された意図: {result['intent']}]*"
                
                if result.get("processing_time"):
                    agent_response += f"\n*[処理時間: {result['processing_time']:.2f}秒]*"
                
                return agent_response
            else:
                return f"エージェント機能でエラーが発生しました (HTTP {response.status_code})"
                
        except requests.exceptions.RequestException as e:
            return f"エージェント機能への接続に失敗しました: {str(e)}"
        except Exception as e:
            return f"エージェント機能で予期しないエラーが発生しました: {str(e)}"

    def _call_ollama_direct(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        """OLLAMAに直接リクエスト（フォールバック）"""
        try:
            # OLLAMAのAPIエンドポイントを呼び出し
            ollama_response = requests.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": model_id,
                    "prompt": user_message,
                    "stream": False
                },
                timeout=60
            )
            
            if ollama_response.status_code == 200:
                result = ollama_response.json()
                return result.get("response", "応答を生成できませんでした。")
            else:
                return "LLMからの応答を取得できませんでした。"
                
        except Exception as e:
            return f"LLMとの通信でエラーが発生しました: {str(e)}"


