#!/usr/bin/env python3
"""
シンプルなエージェントテスト
基本的な応答機能を確認
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

class SimpleAgentTester:
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
            await asyncio.sleep(10)
            logger.info("Streamlitアプリが起動しました")
            return True
            
        except Exception as e:
            logger.error(f"Streamlit起動エラー: {e}")
            return False
    
    async def test_basic_conversation(self):
        """基本的な会話テスト"""
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                logger.info("WebUIにアクセス中...")
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                # ページの状態を確認
                title = await page.title()
                logger.info(f"ページタイトル: {title}")
                
                # チャット入力欄を探す
                logger.info("チャット入力欄を探しています...")
                chat_input = await page.wait_for_selector('textarea[placeholder*="自然言語"]', timeout=15000)
                
                if not chat_input:
                    logger.error("チャット入力欄が見つかりません")
                    return False
                
                logger.info("チャット入力欄を発見しました")
                
                # シンプルなテストメッセージ
                test_message = "こんにちは"
                logger.info(f"テストメッセージを送信: {test_message}")
                
                # メッセージを入力
                await chat_input.click()
                await asyncio.sleep(0.5)
                await page.keyboard.type(test_message)
                await asyncio.sleep(0.5)
                await page.keyboard.press('Enter')
                
                # 応答を待つ（最大30秒）
                logger.info("エージェントの応答を待機中...")
                start_time = time.time()
                last_response_length = 0
                
                while time.time() - start_time < 30:
                    try:
                        messages = await page.query_selector_all('[data-testid="stChatMessage"]')
                        if messages:
                            current_response = await messages[-1].text_content()
                            current_length = len(current_response)
                            
                            # 応答が変更された場合
                            if current_length != last_response_length:
                                logger.info(f"応答更新: {current_length}文字")
                                last_response_length = current_length
                                
                                # "思考中"や"初期化中"がなくなったら完了
                                if "思考中" not in current_response and "初期化中" not in current_response:
                                    logger.info("エージェントの応答が完了しました")
                                    logger.info(f"=== エージェントの応答 ===")
                                    logger.info(f"{current_response}")
                                    logger.info("=" * 50)
                                    return True
                        
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"応答確認エラー: {e}")
                        await asyncio.sleep(2)
                
                logger.warning("応答待機タイムアウト")
                return False
                
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
    tester = SimpleAgentTester()
    
    try:
        # Streamlitを起動
        if not await tester.start_streamlit():
            logger.error("Streamlitの起動に失敗しました")
            return
        
        # 基本的な会話テスト
        logger.info("=== 基本的な会話テスト ===")
        success = await tester.test_basic_conversation()
        
        if success:
            logger.info("✅ 基本的な会話テストが成功しました！")
        else:
            logger.warning("❌ 基本的な会話テストに失敗しました")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        
    finally:
        # Streamlitを停止
        tester.stop_streamlit()

if __name__ == "__main__":
    asyncio.run(main())
