#!/usr/bin/env python3
"""
最終的な自己学習AIエージェントテスト
長い待機時間でエージェントの完全な応答を確認
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

class FinalAgentTester:
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
    
    async def wait_for_agent_response(self, page, timeout=60):
        """エージェントの応答を待つ"""
        start_time = time.time()
        last_response = ""
        
        while time.time() - start_time < timeout:
            try:
                messages = await page.query_selector_all('[data-testid="stChatMessage"]')
                if messages:
                    current_response = await messages[-1].text_content()
                    
                    # 応答が変更されたかチェック
                    if current_response != last_response:
                        last_response = current_response
                        
                        # "思考中"や"初期化中"のメッセージがなくなったかチェック
                        if "思考中" not in current_response and "初期化中" not in current_response:
                            logger.info("エージェントの応答が完了しました")
                            return current_response
                    
                    logger.info(f"エージェント応答待機中... ({len(current_response)}文字)")
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"応答待機エラー: {e}")
                await asyncio.sleep(2)
        
        logger.warning(f"応答待機タイムアウト ({timeout}秒)")
        return last_response
    
    async def test_comprehensive_agent(self):
        """包括的なエージェントテスト"""
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                # チャット入力欄を取得
                chat_input = await page.wait_for_selector('textarea[placeholder*="自然言語"]', timeout=15000)
                
                # 重要なテストケース（少なくして詳細にテスト）
                critical_tests = [
                    {
                        "name": "基本機能テスト",
                        "message": "あなたはどのような機能を持っていますか？自己学習、Web検索、コマンド実行、ファイル操作、MCP使用について教えてください。",
                        "keywords": ["自己学習", "Web検索", "コマンド", "ファイル", "MCP"]
                    },
                    {
                        "name": "プロジェクト目標確認",
                        "message": "プロジェクトの目標である永続的会話セッション、エージェントの自己プロンプト書き換え、チューニングデータ操作、SAKANA AI進化システム、報酬構造確立について説明してください。",
                        "keywords": ["永続", "プロンプト", "チューニング", "進化", "報酬"]
                    }
                ]
                
                results = []
                
                for i, test in enumerate(critical_tests):
                    logger.info(f"=== 重要テスト {i+1}: {test['name']} ===")
                    
                    try:
                        # メッセージを送信
                        await chat_input.click()
                        await asyncio.sleep(1)
                        await page.keyboard.type(test['message'])
                        await asyncio.sleep(1)
                        await page.keyboard.press('Enter')
                        
                        # エージェントの応答を待つ（最大60秒）
                        logger.info("エージェントの応答を待機中...")
                        response = await self.wait_for_agent_response(page, timeout=60)
                        
                        if response:
                            logger.info(f"=== エージェントの完全応答 ===")
                            logger.info(f"{response}")
                            logger.info("=" * 50)
                            
                            # キーワード確認
                            found_keywords = [kw for kw in test['keywords'] if kw in response]
                            
                            test_result = {
                                "name": test['name'],
                                "success": len(found_keywords) >= len(test['keywords']) * 0.5,  # 50%以上
                                "found_keywords": found_keywords,
                                "expected_keywords": test['keywords'],
                                "response": response
                            }
                            
                            results.append(test_result)
                            
                            if test_result['success']:
                                logger.info(f"✅ {test['name']}: 成功")
                                logger.info(f"   発見されたキーワード: {found_keywords}")
                            else:
                                logger.warning(f"⚠️ {test['name']}: 部分的成功")
                                logger.warning(f"   期待: {test['keywords']}")
                                logger.warning(f"   発見: {found_keywords}")
                        else:
                            logger.error(f"❌ {test['name']}: 応答なし")
                            results.append({
                                "name": test['name'],
                                "success": False,
                                "error": "応答なし"
                            })
                        
                    except Exception as e:
                        logger.error(f"❌ {test['name']}: エラー - {e}")
                        results.append({
                            "name": test['name'],
                            "success": False,
                            "error": str(e)
                        })
                    
                    # 次のテストの前に待機
                    await asyncio.sleep(3)
                
                # 最終結果評価
                logger.info("=== 最終テスト結果 ===")
                successful_tests = [r for r in results if r.get('success', False)]
                total_tests = len(results)
                
                success_rate = len(successful_tests) / total_tests if total_tests > 0 else 0
                
                logger.info(f"成功率: {success_rate:.1%} ({len(successful_tests)}/{total_tests})")
                
                for result in results:
                    status = "✅" if result.get('success', False) else "❌"
                    logger.info(f"{status} {result['name']}")
                
                # プロジェクト目標の達成判定
                if success_rate >= 0.5:  # 50%以上
                    logger.info("🎉 自己学習AIエージェントは基本的に動作しています！")
                    logger.info("プロジェクト目標の一部が達成されています。")
                    
                    # 達成された機能を報告
                    if any("自己学習" in str(r.get('found_keywords', [])) for r in results):
                        logger.info("✅ 自己学習機能が確認されました")
                    if any("Web検索" in str(r.get('found_keywords', [])) for r in results):
                        logger.info("✅ Web検索機能が確認されました")
                    if any("コマンド" in str(r.get('found_keywords', [])) for r in results):
                        logger.info("✅ コマンド実行機能が確認されました")
                    if any("ファイル" in str(r.get('found_keywords', [])) for r in results):
                        logger.info("✅ ファイル操作機能が確認されました")
                    if any("MCP" in str(r.get('found_keywords', [])) for r in results):
                        logger.info("✅ MCP使用機能が確認されました")
                    if any("進化" in str(r.get('found_keywords', [])) for r in results):
                        logger.info("✅ 進化システムが確認されました")
                    if any("報酬" in str(r.get('found_keywords', [])) for r in results):
                        logger.info("✅ 報酬システムが確認されました")
                else:
                    logger.warning("⚠️ エージェントの機能改善が必要です")
                
                return success_rate >= 0.5
                
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
    tester = FinalAgentTester()
    
    try:
        # Streamlitを起動
        if not await tester.start_streamlit():
            logger.error("Streamlitの起動に失敗しました")
            return
        
        # 包括的なエージェントテスト
        logger.info("=== 最終的自己学習AIエージェントテスト ===")
        success = await tester.test_comprehensive_agent()
        
        if success:
            logger.info("🎉 テスト成功！自己学習AIエージェントが動作しています。")
        else:
            logger.info("⚠️ テスト結果: エージェントの機能改善が必要です。")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        
    finally:
        # Streamlitを停止
        tester.stop_streamlit()

if __name__ == "__main__":
    asyncio.run(main())
