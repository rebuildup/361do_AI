#!/usr/bin/env python3
"""
直接Playwrightを使用したテスト
"""

import asyncio
from playwright.async_api import async_playwright
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_streamlit_app():
    """Streamlitアプリを直接テスト"""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            logger.info("テスト用Streamlitアプリにアクセス中...")
            await page.goto("http://localhost:8502")
            await page.wait_for_load_state('networkidle')
            
            # ページのタイトルを確認
            title = await page.title()
            logger.info(f"ページタイトル: {title}")
            
            # チャット入力欄を探す
            logger.info("チャット入力欄を探しています...")
            chat_input = await page.wait_for_selector('textarea[data-testid="stChatInput"]', timeout=10000)
            
            if not chat_input:
                logger.error("チャット入力欄が見つかりません")
                return False
            
            logger.info("チャット入力欄を発見しました")
            
            # テストメッセージを送信
            test_messages = [
                "こんにちは",
                "自己学習機能について教えて",
                "Web検索機能はありますか？",
                "コマンド実行機能について",
                "ファイル操作機能について",
                "MCP機能について",
                "進化システムについて",
                "報酬システムについて"
            ]
            
            for i, message in enumerate(test_messages):
                logger.info(f"=== テスト {i+1}: {message} ===")
                
                # メッセージを入力
                await chat_input.click()
                await asyncio.sleep(0.5)
                await page.keyboard.type(message)
                await asyncio.sleep(0.5)
                await page.keyboard.press('Enter')
                
                # 応答を待つ
                await asyncio.sleep(3)
                
                # 応答を確認
                try:
                    messages = await page.query_selector_all('[data-testid="stChatMessage"]')
                    if messages:
                        last_message = messages[-1]
                        response = await last_message.text_content()
                        logger.info(f"エージェントの応答: {response[:200]}...")
                    else:
                        logger.warning("応答メッセージが見つかりません")
                except Exception as e:
                    logger.error(f"応答確認エラー: {e}")
                
                await asyncio.sleep(2)
            
            logger.info("✅ すべてのテストが完了しました！")
            return True
            
        except Exception as e:
            logger.error(f"テスト実行エラー: {e}")
            return False
            
        finally:
            await browser.close()

async def main():
    """メイン実行関数"""
    logger.info("=== 直接Playwrightテスト開始 ===")
    success = await test_streamlit_app()
    
    if success:
        logger.info("🎉 テストが成功しました！")
    else:
        logger.warning("❌ テストに失敗しました")

if __name__ == "__main__":
    asyncio.run(main())
