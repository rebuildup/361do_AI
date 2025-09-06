"""
Web Search Tool

ネット検索機能を提供するツール
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import urllib.parse

from langchain.tools import BaseTool


class WebSearchTool(BaseTool):
    """Web検索ツール"""
    
    name: str = "web_search"
    description: str = "インターネットで情報を検索します。検索クエリを指定してください。"
    
    def __init__(self, 
                 search_engine: str = "duckduckgo",
                 max_results: int = 5,
                 timeout: int = 30):
        super().__init__()
        self._search_engine = search_engine
        self._max_results = max_results
        self._timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    @property
    def search_engine(self) -> str:
        return self._search_engine
    
    @property
    def max_results(self) -> int:
        return self._max_results
    
    @property
    def timeout(self) -> int:
        return self._timeout
    
    def _run(self, query: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(query, **kwargs))
    
    async def _arun(self, query: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not query.strip():
                return "検索クエリが空です。"
            
            self.logger.info(f"Web search: {query}")
            
            if self.search_engine == "duckduckgo":
                results = await self._search_duckduckgo(query)
            elif self.search_engine == "google":
                results = await self._search_google(query)
            else:
                return f"サポートされていない検索エンジン: {self.search_engine}"
            
            if not results:
                return "検索結果が見つかりませんでした。"
            
            # 結果を整形
            formatted_results = []
            for i, result in enumerate(results[:self.max_results], 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('url', 'No URL')}\n"
                    f"   説明: {result.get('snippet', 'No description')}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            self.logger.error(f"Web search error: {e}")
            return f"検索エラー: {str(e)}"
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """DuckDuckGo検索"""
        
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        
                        # Abstract (要約)
                        if data.get("Abstract"):
                            results.append({
                                "title": data.get("Heading", "DuckDuckGo Instant Answer"),
                                "url": data.get("AbstractURL", ""),
                                "snippet": data.get("Abstract", "")
                            })
                        
                        # Related Topics
                        for topic in data.get("RelatedTopics", []):
                            if isinstance(topic, dict) and topic.get("Text"):
                                results.append({
                                    "title": topic.get("FirstURL", "").split("/")[-1] if topic.get("FirstURL") else "Related Topic",
                                    "url": topic.get("FirstURL", ""),
                                    "snippet": topic.get("Text", "")
                                })
                        
                        return results
                    else:
                        self.logger.error(f"DuckDuckGo API error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    async def _search_google(self, query: str) -> List[Dict[str, Any]]:
        """Google検索（簡易版）"""
        
        try:
            # Google Custom Search API の代替として、HTMLパースを使用
            # 注意: 実際のプロダクションでは適切なAPIキーを使用してください
            
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # 簡易的なHTMLパース（実際の実装ではBeautifulSoupなどを使用）
                        results = []
                        
                        # 基本的な結果抽出（実際の実装ではより詳細なパースが必要）
                        if "検索結果" in html or "Search results" in html:
                            results.append({
                                "title": f"Google検索結果: {query}",
                                "url": search_url,
                                "snippet": "Google検索が実行されました。詳細な結果を取得するには、適切なAPIキーが必要です。"
                            })
                        
                        return results
                    else:
                        self.logger.error(f"Google search error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Google search error: {e}")
            return []


class WebScrapingTool(BaseTool):
    """Webスクレイピングツール"""
    
    name: str = "web_scraping"
    description: str = "指定されたURLからWebページの内容を取得します。"
    
    def __init__(self, timeout: int = 30, max_content_length: int = 10000):
        super().__init__()
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.logger = logging.getLogger(__name__)
    
    def _run(self, url: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(url, **kwargs))
    
    async def _arun(self, url: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not url.strip():
                return "URLが指定されていません。"
            
            self.logger.info(f"Web scraping: {url}")
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # コンテンツの長さ制限
                        if len(content) > self.max_content_length:
                            content = content[:self.max_content_length] + "..."
                        
                        # 基本的なテキスト抽出（実際の実装ではBeautifulSoupなどを使用）
                        # HTMLタグを除去する簡易的な処理
                        import re
                        text_content = re.sub(r'<[^>]+>', '', content)
                        text_content = re.sub(r'\s+', ' ', text_content).strip()
                        
                        return f"URL: {url}\n\n内容:\n{text_content}"
                    else:
                        return f"HTTP エラー: {response.status} - {url}"
                        
        except Exception as e:
            self.logger.error(f"Web scraping error: {e}")
            return f"スクレイピングエラー: {str(e)}"


class NewsSearchTool(BaseTool):
    """ニュース検索ツール"""
    
    name: str = "news_search"
    description: str = "最新のニュースを検索します。"
    
    def __init__(self, max_results: int = 5, timeout: int = 30):
        super().__init__()
        self.max_results = max_results
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def _run(self, query: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(query, **kwargs))
    
    async def _arun(self, query: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not query.strip():
                return "検索クエリが空です。"
            
            self.logger.info(f"News search: {query}")
            
            # NewsAPIの代替として、RSSフィードを使用
            results = await self._search_news_rss(query)
            
            if not results:
                return "ニュースが見つかりませんでした。"
            
            # 結果を整形
            formatted_results = []
            for i, result in enumerate(results[:self.max_results], 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   日時: {result.get('date', 'No date')}\n"
                    f"   説明: {result.get('description', 'No description')}\n"
                    f"   URL: {result.get('url', 'No URL')}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            self.logger.error(f"News search error: {e}")
            return f"ニュース検索エラー: {str(e)}"
    
    async def _search_news_rss(self, query: str) -> List[Dict[str, Any]]:
        """RSSフィードからニュースを検索"""
        
        try:
            # 主要なニュースサイトのRSSフィード
            rss_feeds = [
                "https://feeds.bbci.co.uk/news/rss.xml",
                "https://rss.cnn.com/rss/edition.rss",
                "https://feeds.reuters.com/reuters/topNews"
            ]
            
            results = []
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                for feed_url in rss_feeds:
                    try:
                        async with session.get(feed_url) as response:
                            if response.status == 200:
                                content = await response.text()
                                
                                # 簡易的なRSSパース（実際の実装ではxml.etree.ElementTreeなどを使用）
                                import re
                                
                                # タイトルを抽出
                                titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', content)
                                descriptions = re.findall(r'<description><!\[CDATA\[(.*?)\]\]></description>', content)
                                links = re.findall(r'<link>(.*?)</link>', content)
                                
                                for i, title in enumerate(titles[:3]):  # 各フィードから最大3件
                                    if query.lower() in title.lower() or query.lower() in (descriptions[i] if i < len(descriptions) else "").lower():
                                        results.append({
                                            "title": title,
                                            "description": descriptions[i] if i < len(descriptions) else "",
                                            "url": links[i] if i < len(links) else "",
                                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        })
                                        
                    except Exception as e:
                        self.logger.error(f"RSS feed error {feed_url}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"RSS news search error: {e}")
            return []