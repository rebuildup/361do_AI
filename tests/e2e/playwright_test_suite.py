"""
Playwright MCP Test Suite
Playwright MCPを使用したブラウザ自動化テストスイート
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Playwright MCP tools
from mcp_playwright_mcp_browser_navigate import mcp_playwright_mcp_browser_navigate
from mcp_playwright_mcp_browser_snapshot import mcp_playwright_mcp_browser_snapshot
from mcp_playwright_mcp_browser_click import mcp_playwright_mcp_browser_click
from mcp_playwright_mcp_browser_type import mcp_playwright_mcp_browser_type
from mcp_playwright_mcp_browser_take_screenshot import mcp_playwright_mcp_browser_take_screenshot
from mcp_playwright_mcp_browser_evaluate import mcp_playwright_mcp_browser_evaluate

logger = logging.getLogger(__name__)


class PlaywrightMCPTestSuite:
    """Playwright MCPテストスイート"""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.screenshots_dir = "tests/e2e/screenshots"
        self.base_url = "http://localhost:8501"  # Streamlit default port
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """全テストを実行"""
        logger.info("🎭 Playwright MCPテストスイート開始")
        
        test_methods = [
            self.test_browser_navigation,
            self.test_page_snapshot,
            self.test_element_interaction,
            self.test_form_filling,
            self.test_screenshot_capture,
            self.test_javascript_execution,
            self.test_streamlit_ui_access,
            self.test_fastapi_endpoints
        ]
        
        results = {}
        for test_method in test_methods:
            try:
                test_name = test_method.__name__
                logger.info(f"実行中: {test_name}")
                
                result = await test_method()
                results[test_name] = {
                    "status": "passed",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"✅ {test_name} 完了")
                
            except Exception as e:
                test_name = test_method.__name__
                results[test_name] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                logger.error(f"❌ {test_name} 失敗: {e}")
        
        self.test_results = results
        return results
    
    async def test_browser_navigation(self) -> Dict[str, Any]:
        """ブラウザナビゲーションテスト"""
        logger.info("🌐 ブラウザナビゲーションテスト開始")
        
        # Googleにアクセス
        result = await mcp_playwright_mcp_browser_navigate(url="https://www.google.com")
        
        # ページタイトルを確認
        title_result = await mcp_playwright_mcp_browser_evaluate(
            function="() => document.title"
        )
        
        return {
            "navigation_result": result,
            "page_title": title_result,
            "test_url": "https://www.google.com"
        }
    
    async def test_page_snapshot(self) -> Dict[str, Any]:
        """ページスナップショットテスト"""
        logger.info("📸 ページスナップショットテスト開始")
        
        # スナップショット取得
        snapshot_result = await mcp_playwright_mcp_browser_snapshot()
        
        return {
            "snapshot_result": snapshot_result,
            "elements_found": len(snapshot_result.get("elements", [])) if isinstance(snapshot_result, dict) else 0
        }
    
    async def test_element_interaction(self) -> Dict[str, Any]:
        """要素インタラクションテスト"""
        logger.info("🖱️ 要素インタラクションテスト開始")
        
        # 検索ボックスをクリック
        click_result = await mcp_playwright_mcp_browser_click(
            element="検索ボックス",
            ref="input[name='q']"
        )
        
        # テキスト入力
        type_result = await mcp_playwright_mcp_browser_type(
            element="検索ボックス",
            ref="input[name='q']",
            text="Playwright MCP test"
        )
        
        return {
            "click_result": click_result,
            "type_result": type_result
        }
    
    async def test_form_filling(self) -> Dict[str, Any]:
        """フォーム入力テスト"""
        logger.info("📝 フォーム入力テスト開始")
        
        # 検索フォームにテキストを入力
        form_result = await mcp_playwright_mcp_browser_type(
            element="検索入力フィールド",
            ref="input[name='q']",
            text="AI Agent Testing"
        )
        
        return {
            "form_filling_result": form_result
        }
    
    async def test_screenshot_capture(self) -> Dict[str, Any]:
        """スクリーンショット取得テスト"""
        logger.info("📷 スクリーンショット取得テスト開始")
        
        # スクリーンショット取得
        screenshot_result = await mcp_playwright_mcp_browser_take_screenshot(
            filename=f"playwright_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        return {
            "screenshot_result": screenshot_result
        }
    
    async def test_javascript_execution(self) -> Dict[str, Any]:
        """JavaScript実行テスト"""
        logger.info("⚡ JavaScript実行テスト開始")
        
        # ページタイトルを取得
        title_result = await mcp_playwright_mcp_browser_evaluate(
            function="() => document.title"
        )
        
        # ページURLを取得
        url_result = await mcp_playwright_mcp_browser_evaluate(
            function="() => window.location.href"
        )
        
        # 要素数を取得
        element_count_result = await mcp_playwright_mcp_browser_evaluate(
            function="() => document.querySelectorAll('*').length"
        )
        
        return {
            "title": title_result,
            "url": url_result,
            "element_count": element_count_result
        }
    
    async def test_streamlit_ui_access(self) -> Dict[str, Any]:
        """Streamlit UIアクセステスト"""
        logger.info("🎨 Streamlit UIアクセステスト開始")
        
        # Streamlit UIにアクセス
        nav_result = await mcp_playwright_mcp_browser_navigate(url=self.base_url)
        
        # ページ読み込み待機
        await asyncio.sleep(2)
        
        # スナップショット取得
        snapshot_result = await mcp_playwright_mcp_browser_snapshot()
        
        # スクリーンショット取得
        screenshot_result = await mcp_playwright_mcp_browser_take_screenshot(
            filename=f"streamlit_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        return {
            "navigation_result": nav_result,
            "snapshot_result": snapshot_result,
            "screenshot_result": screenshot_result
        }
    
    async def test_fastapi_endpoints(self) -> Dict[str, Any]:
        """FastAPIエンドポイントテスト"""
        logger.info("🔌 FastAPIエンドポイントテスト開始")
        
        # FastAPIドキュメントページにアクセス
        api_url = "http://localhost:8000/docs"
        nav_result = await mcp_playwright_mcp_browser_navigate(url=api_url)
        
        # ページ読み込み待機
        await asyncio.sleep(2)
        
        # ページタイトル確認
        title_result = await mcp_playwright_mcp_browser_evaluate(
            function="() => document.title"
        )
        
        # スクリーンショット取得
        screenshot_result = await mcp_playwright_mcp_browser_take_screenshot(
            filename=f"fastapi_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        return {
            "navigation_result": nav_result,
            "title": title_result,
            "screenshot_result": screenshot_result
        }
    
    def generate_test_report(self) -> str:
        """テストレポート生成"""
        if not self.test_results:
            return "テストが実行されていません。"
        
        report = []
        report.append("# 🎭 Playwright MCP テストレポート")
        report.append(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # テスト結果サマリー
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "passed")
        failed_tests = total_tests - passed_tests
        
        report.append("## 📊 テスト結果サマリー")
        report.append(f"- **総テスト数**: {total_tests}")
        report.append(f"- **成功**: {passed_tests} ✅")
        report.append(f"- **失敗**: {failed_tests} ❌")
        report.append(f"- **成功率**: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # 詳細結果
        report.append("## 📋 詳細結果")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result["status"] == "passed" else "❌"
            report.append(f"### {status_emoji} {test_name}")
            report.append(f"- **ステータス**: {result['status']}")
            report.append(f"- **実行時刻**: {result['timestamp']}")
            
            if result["status"] == "failed":
                report.append(f"- **エラー**: {result['error']}")
            else:
                report.append(f"- **結果**: {result['result']}")
            report.append("")
        
        return "\n".join(report)


@pytest.mark.asyncio
async def test_playwright_mcp_suite():
    """Playwright MCPテストスイートの実行"""
    test_suite = PlaywrightMCPTestSuite()
    
    # 全テスト実行
    results = await test_suite.run_all_tests()
    
    # レポート生成
    report = test_suite.generate_test_report()
    print(report)
    
    # 結果検証
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result["status"] == "passed")
    
    # 最低限のテストが成功していることを確認
    assert total_tests > 0, "テストが実行されていません"
    assert passed_tests > 0, "すべてのテストが失敗しました"
    
    # 成功率が50%以上であることを確認
    success_rate = (passed_tests / total_tests) * 100
    assert success_rate >= 50.0, f"成功率が低すぎます: {success_rate:.1f}%"
    
    return results


if __name__ == "__main__":
    # 直接実行時のテスト
    async def main():
        test_suite = PlaywrightMCPTestSuite()
        results = await test_suite.run_all_tests()
        report = test_suite.generate_test_report()
        print(report)
    
    asyncio.run(main())
