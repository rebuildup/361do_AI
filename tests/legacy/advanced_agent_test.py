#!/usr/bin/env python3
"""
高度な自己学習AIエージェントの詳細テスト
"""

import asyncio
import time
import subprocess
import sys
from playwright.async_api import async_playwright
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAgentTester:
    def __init__(self):
        self.streamlit_process = None
        self.base_url = "http://localhost:8501"
        
    async def start_streamlit(self):
        """Streamlitアプリを起動"""
        try:
            logger.info("Streamlitアプリを起動中...")
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "webui.py", 
                "--server.port", "8501", "--server.headless", "true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # アプリの起動を待つ
            await asyncio.sleep(8)
            logger.info("Streamlitアプリが起動しました")
            return True
            
        except Exception as e:
            logger.error(f"Streamlit起動エラー: {e}")
            return False
    
    async def test_agent_capabilities(self):
        """エージェントの能力を詳細テスト"""
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                # チャット入力欄を取得
                chat_input = await page.wait_for_selector('textarea[placeholder*="自然言語"]', timeout=10000)
                
                # プロジェクト目標に基づくテストケース
                test_cases = [
                    {
                        "name": "基本挨拶",
                        "message": "こんにちは、自己学習AIエージェントさん",
                        "expected_keywords": ["こんにちは", "エージェント", "自己学習"]
                    },
                    {
                        "name": "自己学習機能の確認",
                        "message": "あなたの自己学習機能について詳しく教えてください",
                        "expected_keywords": ["自己学習", "プロンプト", "チューニング", "進化"]
                    },
                    {
                        "name": "永続セッションの確認",
                        "message": "セッションが永続的に続くかテストしてください",
                        "expected_keywords": ["セッション", "永続", "記憶"]
                    },
                    {
                        "name": "Web検索機能のテスト",
                        "message": "最新のAI技術についてWeb検索して情報を教えてください",
                        "expected_keywords": ["検索", "AI技術", "情報"]
                    },
                    {
                        "name": "コマンド実行機能のテスト",
                        "message": "現在のシステムの状態を確認するコマンドを実行してください",
                        "expected_keywords": ["システム", "状態", "コマンド"]
                    },
                    {
                        "name": "ファイル操作機能のテスト",
                        "message": "プロジェクトの構造を確認するファイル操作を実行してください",
                        "expected_keywords": ["ファイル", "プロジェクト", "構造"]
                    },
                    {
                        "name": "MCP機能のテスト",
                        "message": "MCPツールを使用して何か操作を実行してください",
                        "expected_keywords": ["MCP", "ツール"]
                    },
                    {
                        "name": "進化システムのテスト",
                        "message": "SAKANA AIスタイルの進化システムを実行してください",
                        "expected_keywords": ["進化", "SAKANA", "世代"]
                    },
                    {
                        "name": "報酬システムのテスト",
                        "message": "報酬システムの状態を確認してください",
                        "expected_keywords": ["報酬", "学習", "スコア"]
                    },
                    {
                        "name": "Deepseekレベルの推論テスト",
                        "message": "複雑な論理問題を解いてください：AはBより背が高く、BはCより背が高い。AとCのどちらが背が高いですか？",
                        "expected_keywords": ["論理", "推論", "A", "C", "背が高い"]
                    }
                ]
                
                results = []
                
                for i, test_case in enumerate(test_cases):
                    logger.info(f"=== テストケース {i+1}: {test_case['name']} ===")
                    
                    try:
                        # メッセージを送信
                        await chat_input.click()
                        await asyncio.sleep(0.5)
                        await page.keyboard.type(test_case['message'])
                        await asyncio.sleep(0.5)
                        await page.keyboard.press('Enter')
                        
                        # 応答を待つ（エージェントの初期化時間を考慮）
                        await asyncio.sleep(10)
                        
                        # 応答を取得
                        messages = await page.query_selector_all('[data-testid="stChatMessage"]')
                        if messages:
                            last_message = messages[-1]
                            response_content = await last_message.text_content()
                            
                            logger.info(f"エージェントの応答: {response_content[:300]}...")
                            
                            # 期待されるキーワードの確認
                            found_keywords = []
                            for keyword in test_case['expected_keywords']:
                                if keyword in response_content:
                                    found_keywords.append(keyword)
                            
                            test_result = {
                                "name": test_case['name'],
                                "success": len(found_keywords) > 0,
                                "found_keywords": found_keywords,
                                "expected_keywords": test_case['expected_keywords'],
                                "response_length": len(response_content),
                                "response_preview": response_content[:200]
                            }
                            
                            results.append(test_result)
                            
                            if test_result['success']:
                                logger.info(f"✅ {test_case['name']}: 成功 (キーワード: {found_keywords})")
                            else:
                                logger.warning(f"⚠️ {test_case['name']}: 期待されるキーワードが見つかりません")
                        else:
                            logger.error(f"❌ {test_case['name']}: 応答が取得できませんでした")
                            results.append({
                                "name": test_case['name'],
                                "success": False,
                                "error": "応答が取得できませんでした"
                            })
                        
                    except Exception as e:
                        logger.error(f"❌ {test_case['name']}: エラー - {e}")
                        results.append({
                            "name": test_case['name'],
                            "success": False,
                            "error": str(e)
                        })
                    
                    # 次のテストの前に待機
                    await asyncio.sleep(2)
                
                # 結果サマリー
                logger.info("=== テスト結果サマリー ===")
                successful_tests = [r for r in results if r.get('success', False)]
                failed_tests = [r for r in results if not r.get('success', False)]
                
                logger.info(f"成功: {len(successful_tests)}/{len(results)}")
                logger.info(f"失敗: {len(failed_tests)}/{len(results)}")
                
                for result in successful_tests:
                    logger.info(f"✅ {result['name']}: {result['found_keywords']}")
                
                for result in failed_tests:
                    logger.info(f"❌ {result['name']}: {result.get('error', 'キーワード不一致')}")
                
                # プロジェクト目標の達成度評価
                project_goals = [
                    "永続的会話セッション",
                    "エージェントの自己プロンプト書き換え", 
                    "チューニングデータ操作",
                    "Web検索機能",
                    "コマンド実行機能",
                    "ファイル変更機能",
                    "MCP使用",
                    "AI進化システム",
                    "報酬構造確立"
                ]
                
                logger.info("=== プロジェクト目標達成度 ===")
                for goal in project_goals:
                    # 関連するテストケースの成功を確認
                    related_tests = [r for r in results if goal.lower().replace(" ", "") in r['name'].lower().replace(" ", "")]
                    if related_tests and any(r.get('success', False) for r in related_tests):
                        logger.info(f"✅ {goal}")
                    else:
                        logger.info(f"❌ {goal}")
                
                return len(successful_tests) >= len(results) * 0.7  # 70%以上の成功率
                
            except Exception as e:
                logger.error(f"テスト実行エラー: {e}")
                return False
                
            finally:
                await browser.close()
    
    def stop_streamlit(self):
        """Streamlitアプリを停止"""
        if self.streamlit_process:
            logger.info("Streamlitアプリを停止中...")
            self.streamlit_process.terminate()
            self.streamlit_process.wait()
            logger.info("Streamlitアプリが停止しました")

async def main():
    """メイン実行関数"""
    tester = AdvancedAgentTester()
    
    try:
        # Streamlitを起動
        if not await tester.start_streamlit():
            logger.error("Streamlitの起動に失敗しました")
            return
        
        # エージェントの能力をテスト
        logger.info("=== 高度な自己学習AIエージェントテスト開始 ===")
        success = await tester.test_agent_capabilities()
        
        if success:
            logger.info("🎉 自己学習AIエージェントのテストが成功しました！")
            logger.info("プロジェクト目標の大部分が達成されています。")
        else:
            logger.warning("⚠️ 一部のテストに問題がありました。")
            logger.info("エージェントの機能改善が必要です。")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        
    finally:
        # Streamlitを停止
        tester.stop_streamlit()

if __name__ == "__main__":
    asyncio.run(main())
