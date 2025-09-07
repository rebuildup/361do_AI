#!/usr/bin/env python3
"""
改善されたWebUIのテストスクリプト
John Carmack、Robert C. Martin、Rob Pikeの設計思想を意識した改善点をテスト
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

class ImprovedWebUITester:
    def __init__(self):
        self.streamlit_process = None
        self.base_url = "http://localhost:8501"
        
    async def start_streamlit(self):
        """Streamlitアプリを起動"""
        try:
            logger.info("改善されたWebUIを起動中...")
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "webui.py", 
                "--server.port", "8501", "--server.headless", "true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # アプリの起動を待つ
            await asyncio.sleep(5)
            logger.info("改善されたWebUIが起動しました")
            return True
            
        except Exception as e:
            logger.error(f"WebUI起動エラー: {e}")
            return False
    
    async def test_sidebar_functionality(self):
        """サイドバーの機能性をテスト"""
        logger.info("🎛️ サイドバー機能テスト開始")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                # サイドバーの要素をテスト
                sidebar_tests = [
                    ("モデル選択", "selectbox"),
                    ("エージェント起動", "button"),
                    ("学習制御", "button"),
                    ("ツール実行", "button"),
                    ("プロンプト管理", "button"),
                    ("システム監視", "metric"),
                    ("セッション管理", "button"),
                    ("設定", "slider")
                ]
                
                results = {}
                for test_name, element_type in sidebar_tests:
                    try:
                        # 要素の存在確認
                        if element_type == "button":
                            elements = await page.query_selector_all('button')
                            found = any(await el.text_content() for el in elements if test_name in (await el.text_content() or ""))
                        elif element_type == "selectbox":
                            elements = await page.query_selector_all('select, [role="combobox"]')
                            found = len(elements) > 0
                        elif element_type == "slider":
                            elements = await page.query_selector_all('input[type="range"]')
                            found = len(elements) > 0
                        elif element_type == "metric":
                            elements = await page.query_selector_all('[data-testid="stMetric"]')
                            found = len(elements) > 0
                        
                        results[test_name] = found
                        logger.info(f"  {test_name}: {'✅' if found else '❌'}")
                        
                    except Exception as e:
                        results[test_name] = False
                        logger.error(f"  {test_name}: ❌ エラー - {e}")
                
                return results
                
            except Exception as e:
                logger.error(f"サイドバーテストエラー: {e}")
                return {}
                
            finally:
                await browser.close()
    
    async def test_chat_improvements(self):
        """チャット機能の改善点をテスト"""
        logger.info("💬 チャット改善テスト開始")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                # チャット入力欄を探す
                chat_input = await page.wait_for_selector('textarea[data-testid="stChatInput"], input[data-testid="stChatInput"]', timeout=10000)
                
                if chat_input:
                    logger.info("✅ チャット入力欄を発見")
                    
                    # 長いプロンプトをテスト
                    long_prompt = "これは非常に長いプロンプトです。" * 20
                    await chat_input.fill(long_prompt)
                    await page.keyboard.press('Enter')
                    
                    # 応答を待つ
                    await asyncio.sleep(3)
                    
                    # プロンプト全文表示の確認
                    expander = await page.query_selector('div[data-testid="stExpander"]')
                    if expander:
                        logger.info("✅ プロンプト全文表示機能を確認")
                    else:
                        logger.warning("⚠️ プロンプト全文表示機能が見つかりません")
                    
                    # 応答品質インジケーターの確認
                    quality_indicators = await page.query_selector_all('div[data-testid="stSuccess"], div[data-testid="stInfo"], div[data-testid="stWarning"]')
                    if quality_indicators:
                        logger.info("✅ 応答品質インジケーターを確認")
                    else:
                        logger.warning("⚠️ 応答品質インジケーターが見つかりません")
                    
                    return True
                else:
                    logger.error("❌ チャット入力欄が見つかりません")
                    return False
                    
            except Exception as e:
                logger.error(f"チャット改善テストエラー: {e}")
                return False
                
            finally:
                await browser.close()
    
    async def test_responsive_design(self):
        """レスポンシブデザインをテスト"""
        logger.info("📱 レスポンシブデザインテスト開始")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # デスクトップサイズ
                await page.set_viewport_size({"width": 1200, "height": 800})
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                
                # サイドバーの幅を確認
                sidebar = await page.query_selector('.stSidebar')
                if sidebar:
                    sidebar_width = await sidebar.evaluate('el => el.offsetWidth')
                    logger.info(f"✅ デスクトップサイドバー幅: {sidebar_width}px")
                
                # タブレットサイズ
                await page.set_viewport_size({"width": 768, "height": 1024})
                await page.wait_for_load_state('networkidle')
                
                if sidebar:
                    sidebar_width = await sidebar.evaluate('el => el.offsetWidth')
                    logger.info(f"✅ タブレットサイドバー幅: {sidebar_width}px")
                
                # モバイルサイズ
                await page.set_viewport_size({"width": 375, "height": 667})
                await page.wait_for_load_state('networkidle')
                
                if sidebar:
                    sidebar_width = await sidebar.evaluate('el => el.offsetWidth')
                    logger.info(f"✅ モバイルサイドバー幅: {sidebar_width}px")
                
                return True
                
            except Exception as e:
                logger.error(f"レスポンシブテストエラー: {e}")
                return False
                
            finally:
                await browser.close()
    
    async def test_performance(self):
        """パフォーマンステスト"""
        logger.info("⚡ パフォーマンステスト開始")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # ページ読み込み時間を測定
                start_time = time.time()
                await page.goto(self.base_url)
                await page.wait_for_load_state('networkidle')
                load_time = time.time() - start_time
                
                logger.info(f"✅ ページ読み込み時間: {load_time:.2f}秒")
                
                # インタラクション応答時間を測定
                chat_input = await page.wait_for_selector('textarea[data-testid="stChatInput"], input[data-testid="stChatInput"]')
                if chat_input:
                    start_time = time.time()
                    await chat_input.click()
                    response_time = time.time() - start_time
                    logger.info(f"✅ インタラクション応答時間: {response_time:.3f}秒")
                
                # メモリ使用量の確認
                memory_usage = await page.evaluate('performance.memory ? performance.memory.usedJSHeapSize : 0')
                logger.info(f"✅ JavaScript メモリ使用量: {memory_usage / 1024 / 1024:.2f}MB")
                
                return True
                
            except Exception as e:
                logger.error(f"パフォーマンステストエラー: {e}")
                return False
                
            finally:
                await browser.close()
    
    async def run_all_tests(self):
        """全テストを実行"""
        logger.info("🚀 改善されたWebUIテストスイート開始")
        logger.info("=" * 60)
        
        # Streamlitを起動
        if not await self.start_streamlit():
            logger.error("WebUIの起動に失敗しました")
            return
        
        tests = [
            ("サイドバー機能", self.test_sidebar_functionality),
            ("チャット改善", self.test_chat_improvements),
            ("レスポンシブデザイン", self.test_responsive_design),
            ("パフォーマンス", self.test_performance)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n--- {test_name} ---")
                result = await test_func()
                results[test_name] = result
                
                if isinstance(result, dict):
                    success_count = sum(1 for v in result.values() if v)
                    total_count = len(result)
                    logger.info(f"✅ {test_name}完了: {success_count}/{total_count} 成功")
                elif result:
                    logger.info(f"✅ {test_name}完了: 成功")
                else:
                    logger.warning(f"⚠️ {test_name}完了: 問題あり")
                
            except Exception as e:
                logger.error(f"❌ {test_name}でエラー: {e}")
                results[test_name] = False
        
        # 結果サマリー
        logger.info("\n" + "=" * 60)
        logger.info("📋 テスト結果サマリー")
        logger.info("=" * 60)
        
        for test_name, result in results.items():
            if isinstance(result, dict):
                success_count = sum(1 for v in result.values() if v)
                total_count = len(result)
                status = f"{success_count}/{total_count} 成功"
            elif result:
                status = "✅ 成功"
            else:
                status = "❌ 失敗"
            
            logger.info(f"{test_name}: {status}")
        
        logger.info("\n🎉 改善されたWebUIテスト完了")
        
        return results
    
    def stop_streamlit(self):
        """Streamlitアプリを停止"""
        if self.streamlit_process:
            logger.info("WebUIを停止中...")
            self.streamlit_process.terminate()
            self.streamlit_process.wait()
            logger.info("WebUIが停止しました")

async def main():
    """メイン実行関数"""
    tester = ImprovedWebUITester()
    
    try:
        await tester.run_all_tests()
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
    finally:
        tester.stop_streamlit()

if __name__ == "__main__":
    asyncio.run(main())
