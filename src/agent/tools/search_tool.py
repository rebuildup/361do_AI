"""
Search Tool
Web検索機能を提供するツール
"""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from loguru import logger


class SearchTool:
    """Web検索ツール"""

    def __init__(self):
        self.ddgs = None
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def initialize(self):
        """ツール初期化"""
        logger.info("Initializing Search Tool...")

        self.ddgs = DDGS()
        self.session = aiohttp.ClientSession(timeout=self.timeout)

        logger.info("Search Tool initialized")

    async def close(self):
        """ツール終了"""
        if self.session:
            await self.session.close()

    async def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "jp-jp",
        safesearch: str = "moderate"
    ) -> Dict[str, Any]:
        """Web検索実行"""
        try:
            logger.info(f"Searching for: {query}")

            # DuckDuckGo検索
            search_results = await self._duckduckgo_search(
                query, max_results, region, safesearch
            )

            # 結果が少ない場合は追加検索
            if len(search_results) < max_results:
                additional_results = await self._fallback_search(query, max_results - len(search_results))
                search_results.extend(additional_results)

            # コンテンツ取得（上位3件）
            enriched_results = []
            for i, result in enumerate(search_results[:3]):
                try:
                    content = await self._fetch_page_content(result['href'])
                    result['content_preview'] = content[:500] + "..." if len(content) > 500 else content
                    enriched_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to fetch content for {result['href']}: {e}")
                    enriched_results.append(result)

            # 残りの結果も追加
            enriched_results.extend(search_results[3:])

            return {
                'query': query,
                'results': enriched_results,
                'total_results': len(enriched_results)
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                'query': query,
                'results': [],
                'error': str(e),
                'total_results': 0
            }

    async def _duckduckgo_search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str
    ) -> List[Dict[str, Any]]:
        """DuckDuckGo検索"""
        try:
            # 非同期実行のため、別スレッドで実行
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self.ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    max_results=max_results
                ))
            )

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    async def _fallback_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """フォールバック検索（簡単なスクレイピング）"""
        try:
            # Bing検索のスクレイピング（簡易版）
            search_url = f"https://www.bing.com/search?q={query}&count={max_results}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            async with self.session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_bing_results(html)
                else:
                    logger.warning(f"Bing search failed with status: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    def _parse_bing_results(self, html: str) -> List[Dict[str, Any]]:
        """Bing検索結果をパース"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []

            # Bingの検索結果セレクター
            result_elements = soup.select('li.b_algo')

            for element in result_elements:
                try:
                    title_element = element.select_one('h2 a')
                    if not title_element:
                        continue

                    title = title_element.get_text(strip=True)
                    href = title_element.get('href', '')

                    # 説明文取得
                    body_element = element.select_one('.b_caption p')
                    body = body_element.get_text(strip=True) if body_element else ''

                    results.append({
                        'title': title,
                        'href': href,
                        'body': body
                    })

                except Exception as e:
                    logger.warning(f"Failed to parse result element: {e}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Failed to parse Bing results: {e}")
            return []

    async def _fetch_page_content(self, url: str) -> str:
        """ページコンテンツ取得"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # 不要な要素を削除
                    for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
                        element.decompose()

                    # メインコンテンツを抽出
                    main_content = soup.select_one('main') or soup.select_one('article') or soup.select_one('body')

                    if main_content:
                        text = main_content.get_text(separator='\n', strip=True)
                        # 空行を削除し、長すぎる行を制限
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        filtered_lines = [line[:200] for line in lines if len(line) > 10]

                        return '\n'.join(filtered_lines[:20])  # 最大20行
                    else:
                        return ""
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return ""

        except Exception as e:
            logger.warning(f"Failed to fetch content from {url}: {e}")
            return ""

    async def search_news(
        self,
        query: str,
        max_results: int = 5,
        region: str = "jp-jp"
    ) -> Dict[str, Any]:
        """ニュース検索"""
        try:
            logger.info(f"Searching news for: {query}")

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self.ddgs.news(
                    query,
                    region=region,
                    max_results=max_results
                ))
            )

            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'type': 'news'
            }

        except Exception as e:
            logger.error(f"News search failed: {e}")
            return {
                'query': query,
                'results': [],
                'error': str(e),
                'total_results': 0,
                'type': 'news'
            }

    async def search_images(
        self,
        query: str,
        max_results: int = 5,
        region: str = "jp-jp"
    ) -> Dict[str, Any]:
        """画像検索"""
        try:
            logger.info(f"Searching images for: {query}")

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self.ddgs.images(
                    query,
                    region=region,
                    max_results=max_results
                ))
            )

            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'type': 'images'
            }

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return {
                'query': query,
                'results': [],
                'error': str(e),
                'total_results': 0,
                'type': 'images'
            }

    async def get_status(self) -> str:
        """ツールステータス取得"""
        try:
            # 簡単なテスト検索
            test_result = await self.search("test", max_results=1)
            if test_result.get('total_results', 0) > 0:
                return "active"
            else:
                return "limited"
        except Exception:
            return "error"
