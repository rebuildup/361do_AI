#!/usr/bin/env python3
"""
自己学習AIエージェントのPlaywrightテスト
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

class AgentTester:
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
            await asyncio.sleep(5)
            logger.info("Streamlitアプリが起動しました")
            return True
            
        except Exception as e:
            logger.error(f"Streamlit起動エラー: {e}")
            return False
    
    async def test_agent_conversation(self):
        """エージェントとの会話をテスト"""
        
        async with async_playwright() as p:
            # ブラウザを起動
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # ページにアクセス
                logger.info("WebUIにアクセス中...")
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                # ページのタイトルを確認
                title = await page.title()
                logger.info(f"ページタイトル: {title}")
                
                # チャット入力欄を探す
                logger.info("チャット入力欄を探しています...")
                
                # チャット入力欄のセレクタを試す（Streamlitのchat_inputに対応）
                chat_selectors = [
                    'textarea[data-testid="stChatInput"]',
                    'input[data-testid="stChatInput"]',
                    'textarea[placeholder*="自然言語"]',
                    'input[placeholder*="自然言語"]',
                    'textarea[placeholder*="AI"]',
                    'input[placeholder*="AI"]',
                    'textarea[placeholder*="チャット"]',
                    'input[placeholder*="チャット"]',
                    '.stChatInput textarea',
                    '.stChatInput input',
                    '[data-testid="stChatInput"]',
                    'textarea',
                    'input[type="text"]'
                ]
                
                chat_input = None
                for selector in chat_selectors:
                    try:
                        chat_input = await page.wait_for_selector(selector, timeout=5000)
                        if chat_input:
                            logger.info(f"チャット入力欄を発見: {selector}")
                            break
                    except:
                        continue
                
                if not chat_input:
                    logger.warning("チャット入力欄が見つかりません。ページの内容を確認します。")
                    # ページの内容を取得
                    content = await page.content()
                    logger.info(f"ページ内容の一部: {content[:500]}...")
                    
                    # 利用可能な入力要素を探す
                    inputs = await page.query_selector_all('input')
                    logger.info(f"見つかったinput要素数: {len(inputs)}")
                    
                    for i, inp in enumerate(inputs):
                        placeholder = await inp.get_attribute('placeholder')
                        input_type = await inp.get_attribute('type')
                        logger.info(f"Input {i}: type={input_type}, placeholder={placeholder}")
                    
                    return False
                
                # テストメッセージを送信
                test_messages = [
                    "こんにちは、エージェント",
                    "自己学習機能について教えて"
                ]
                
                for i, message in enumerate(test_messages):
                    logger.info(f"テストメッセージ {i+1}: {message}")
                    
                    try:
                        # チャット入力欄をクリックしてフォーカス
                        await chat_input.click()
                        await asyncio.sleep(0.5)
                        
                        # メッセージを入力（typeメソッドを使用）
                        await page.keyboard.type(message)
                        await asyncio.sleep(0.5)
                        
                        # Enterキーで送信
                        await page.keyboard.press('Enter')
                        
                        # 応答を待つ
                        await asyncio.sleep(5)
                        
                    except Exception as e:
                        logger.error(f"メッセージ送信エラー: {e}")
                        # 代替方法：直接input要素を探す
                        try:
                            # 他のinput要素を試す
                            inputs = await page.query_selector_all('input')
                            for inp in inputs:
                                try:
                                    await inp.click()
                                    await page.keyboard.type(message)
                                    await page.keyboard.press('Enter')
                                    await asyncio.sleep(5)
                                    break
                                except:
                                    continue
                        except Exception as e2:
                            logger.error(f"代替メソッドも失敗: {e2}")
                    
                    # 応答を確認
                    try:
                        # チャットメッセージを探す（複数のセレクタを試す）
                        message_selectors = [
                            '[data-testid="stChatMessage"]',
                            '.stChatMessage',
                            '.chat-message',
                            '[role="message"]',
                            '.message'
                        ]
                        
                        messages = []
                        for selector in message_selectors:
                            try:
                                found_messages = await page.query_selector_all(selector)
                                if found_messages:
                                    messages = found_messages
                                    logger.info(f"メッセージを発見: {selector}")
                                    break
                            except:
                                continue
                        
                        if messages:
                            last_message = messages[-1]
                            message_content = await last_message.text_content()
                            logger.info(f"エージェントの応答: {message_content[:200]}...")
                        else:
                            logger.warning("応答メッセージが見つかりません")
                            
                            # ページの内容を確認
                            page_content = await page.content()
                            if "エラー" in page_content or "error" in page_content.lower():
                                logger.warning("ページにエラーが含まれている可能性があります")
                            
                    except Exception as e:
                        logger.error(f"応答確認エラー: {e}")
                    
                    # 次のメッセージの前に少し待機
                    await asyncio.sleep(3)
                
                logger.info("会話テストが完了しました")
                return True
                
            except Exception as e:
                logger.error(f"テスト実行エラー: {e}")
                return False
                
            finally:
                await browser.close()
    
    async def test_agent_features(self):
        """エージェントの機能をテスト"""
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                # 各ページをテスト
                pages_to_test = [
                    ("💬 チャット", "chat"),
                    ("🔧 ツール", "tools"),
                    ("📝 プロンプト", "prompts"),
                    ("🧠 学習・進化", "learning"),
                    ("🏆 報酬システム", "rewards"),
                    ("🌐 API管理", "api"),
                    ("📖 ヘルプ", "help")
                ]
                
                for page_name, page_id in pages_to_test:
                    logger.info(f"{page_name}ページをテスト中...")
                    
                    try:
                        # サイドバーのボタンをクリック
                        button = await page.wait_for_selector(f'button:has-text("{page_name}")', timeout=5000)
                        if button:
                            await button.click()
                            await page.wait_for_load_state('networkidle')
                            
                            # ページの内容を確認
                            title = await page.query_selector('h1, h2, h3')
                            if title:
                                title_text = await title.text_content()
                                logger.info(f"{page_name}ページのタイトル: {title_text}")
                            
                            logger.info(f"{page_name}ページのテストが完了しました")
                        else:
                            logger.warning(f"{page_name}ページのボタンが見つかりません")
                            
                    except Exception as e:
                        logger.error(f"{page_name}ページのテストエラー: {e}")
                    
                    await asyncio.sleep(1)
                
                return True
                
            except Exception as e:
                logger.error(f"機能テストエラー: {e}")
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
    tester = AgentTester()
    
    try:
        # Streamlitを起動
        if not await tester.start_streamlit():
            logger.error("Streamlitの起動に失敗しました")
            return
        
        # エージェントとの会話をテスト
        logger.info("=== エージェントとの会話テスト開始 ===")
        conversation_success = await tester.test_agent_conversation()
        
        if conversation_success:
            logger.info("✅ 会話テストが成功しました")
        else:
            logger.warning("⚠️ 会話テストに問題がありました")
        
        # エージェントの機能をテスト
        logger.info("=== エージェント機能テスト開始 ===")
        features_success = await tester.test_agent_features()
        
        if features_success:
            logger.info("✅ 機能テストが成功しました")
        else:
            logger.warning("⚠️ 機能テストに問題がありました")
        
        # 結果のサマリー
        logger.info("=== テスト結果サマリー ===")
        logger.info(f"会話テスト: {'✅ 成功' if conversation_success else '❌ 失敗'}")
        logger.info(f"機能テスト: {'✅ 成功' if features_success else '❌ 失敗'}")
        
        if conversation_success and features_success:
            logger.info("🎉 すべてのテストが成功しました！")
        else:
            logger.warning("⚠️ 一部のテストに問題がありました")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        
    finally:
        # Streamlitを停止
        tester.stop_streamlit()

if __name__ == "__main__":
    asyncio.run(main())
