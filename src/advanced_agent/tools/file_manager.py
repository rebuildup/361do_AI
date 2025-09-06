"""
File Manager Tool

ファイル操作機能を提供するツール
"""

import asyncio
import os
import shutil
import logging
import json
import tempfile
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import mimetypes
import hashlib

from langchain.tools import BaseTool


class FileManagerTool(BaseTool):
    """ファイル管理ツール"""
    
    name: str = "file_manager"
    description: str = "ファイルとディレクトリの操作を行います。"
    
    def __init__(self, 
                 base_directory: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 allowed_extensions: Optional[List[str]] = None):
        super().__init__()
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or [
            '.txt', '.md', '.json', '.yaml', '.yml', '.csv', '.xml', '.html', '.css', '.js',
            '.py', '.java', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.php', '.rb',
            '.sql', '.sh', '.bat', '.ps1', '.dockerfile', '.gitignore', '.env'
        ]
        self.logger = logging.getLogger(__name__)
    
    def _run(self, action: str, path: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(action, path, **kwargs))
    
    async def _arun(self, action: str, path: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not action or not path:
                return "アクションとパスを指定してください。"
            
            # パスを正規化
            normalized_path = self._normalize_path(path)
            if not normalized_path:
                return "無効なパスです。"
            
            self.logger.info(f"File operation: {action} on {normalized_path}")
            
            # アクション別の処理
            if action == "read":
                return await self._read_file(normalized_path)
            elif action == "write":
                content = kwargs.get("content", "")
                return await self._write_file(normalized_path, content)
            elif action == "append":
                content = kwargs.get("content", "")
                return await self._append_file(normalized_path, content)
            elif action == "delete":
                return await self._delete_file(normalized_path)
            elif action == "copy":
                dest_path = kwargs.get("dest_path", "")
                return await self._copy_file(normalized_path, dest_path)
            elif action == "move":
                dest_path = kwargs.get("dest_path", "")
                return await self._move_file(normalized_path, dest_path)
            elif action == "list":
                return await self._list_directory(normalized_path)
            elif action == "create_dir":
                return await self._create_directory(normalized_path)
            elif action == "delete_dir":
                return await self._delete_directory(normalized_path)
            elif action == "info":
                return await self._get_file_info(normalized_path)
            elif action == "search":
                pattern = kwargs.get("pattern", "")
                return await self._search_files(normalized_path, pattern)
            else:
                return f"サポートされていないアクション: {action}"
            
        except Exception as e:
            self.logger.error(f"File operation error: {e}")
            return f"ファイル操作エラー: {str(e)}"
    
    def _normalize_path(self, path: str) -> Optional[Path]:
        """パスを正規化"""
        
        try:
            # パスを解決
            resolved_path = Path(path).resolve()
            
            # ベースディレクトリ内かチェック
            try:
                resolved_path.relative_to(self.base_directory)
            except ValueError:
                # ベースディレクトリ外の場合は、ベースディレクトリからの相対パスとして扱う
                resolved_path = self.base_directory / path
            
            return resolved_path
            
        except Exception as e:
            self.logger.error(f"Path normalization error: {e}")
            return None
    
    async def _read_file(self, path: Path) -> str:
        """ファイルを読み取り"""
        
        try:
            if not path.exists():
                return f"ファイルが存在しません: {path}"
            
            if not path.is_file():
                return f"ファイルではありません: {path}"
            
            # ファイルサイズチェック
            if path.stat().st_size > self.max_file_size:
                return f"ファイルサイズが大きすぎます（最大: {self.max_file_size} bytes）"
            
            # 拡張子チェック
            if not self._is_allowed_extension(path):
                return f"許可されていないファイル形式: {path.suffix}"
            
            # ファイルを読み取り
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return f"ファイル内容 ({path}):\n\n{content}"
            
        except Exception as e:
            return f"ファイル読み取りエラー: {str(e)}"
    
    async def _write_file(self, path: Path, content: str) -> str:
        """ファイルに書き込み"""
        
        try:
            # ディレクトリが存在しない場合は作成
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 拡張子チェック
            if not self._is_allowed_extension(path):
                return f"許可されていないファイル形式: {path.suffix}"
            
            # ファイルに書き込み
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"ファイルに書き込みました: {path}"
            
        except Exception as e:
            return f"ファイル書き込みエラー: {str(e)}"
    
    async def _append_file(self, path: Path, content: str) -> str:
        """ファイルに追記"""
        
        try:
            # ディレクトリが存在しない場合は作成
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 拡張子チェック
            if not self._is_allowed_extension(path):
                return f"許可されていないファイル形式: {path.suffix}"
            
            # ファイルに追記
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            
            return f"ファイルに追記しました: {path}"
            
        except Exception as e:
            return f"ファイル追記エラー: {str(e)}"
    
    async def _delete_file(self, path: Path) -> str:
        """ファイルを削除"""
        
        try:
            if not path.exists():
                return f"ファイルが存在しません: {path}"
            
            if not path.is_file():
                return f"ファイルではありません: {path}"
            
            path.unlink()
            return f"ファイルを削除しました: {path}"
            
        except Exception as e:
            return f"ファイル削除エラー: {str(e)}"
    
    async def _copy_file(self, src_path: Path, dest_path: str) -> str:
        """ファイルをコピー"""
        
        try:
            if not src_path.exists():
                return f"ソースファイルが存在しません: {src_path}"
            
            if not src_path.is_file():
                return f"ソースがファイルではありません: {src_path}"
            
            dest_path = self._normalize_path(dest_path)
            if not dest_path:
                return "無効なコピー先パスです。"
            
            # ディレクトリが存在しない場合は作成
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_path, dest_path)
            return f"ファイルをコピーしました: {src_path} -> {dest_path}"
            
        except Exception as e:
            return f"ファイルコピーエラー: {str(e)}"
    
    async def _move_file(self, src_path: Path, dest_path: str) -> str:
        """ファイルを移動"""
        
        try:
            if not src_path.exists():
                return f"ソースファイルが存在しません: {src_path}"
            
            if not src_path.is_file():
                return f"ソースがファイルではありません: {src_path}"
            
            dest_path = self._normalize_path(dest_path)
            if not dest_path:
                return "無効な移動先パスです。"
            
            # ディレクトリが存在しない場合は作成
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src_path), str(dest_path))
            return f"ファイルを移動しました: {src_path} -> {dest_path}"
            
        except Exception as e:
            return f"ファイル移動エラー: {str(e)}"
    
    async def _list_directory(self, path: Path) -> str:
        """ディレクトリの内容を一覧表示"""
        
        try:
            if not path.exists():
                return f"ディレクトリが存在しません: {path}"
            
            if not path.is_dir():
                return f"ディレクトリではありません: {path}"
            
            items = []
            for item in sorted(path.iterdir()):
                if item.is_file():
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")
                elif item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    items.append(f"❓ {item.name}")
            
            if not items:
                return f"ディレクトリは空です: {path}"
            
            return f"ディレクトリ内容 ({path}):\n\n" + "\n".join(items)
            
        except Exception as e:
            return f"ディレクトリ一覧取得エラー: {str(e)}"
    
    async def _create_directory(self, path: Path) -> str:
        """ディレクトリを作成"""
        
        try:
            if path.exists():
                return f"ディレクトリは既に存在します: {path}"
            
            path.mkdir(parents=True, exist_ok=True)
            return f"ディレクトリを作成しました: {path}"
            
        except Exception as e:
            return f"ディレクトリ作成エラー: {str(e)}"
    
    async def _delete_directory(self, path: Path) -> str:
        """ディレクトリを削除"""
        
        try:
            if not path.exists():
                return f"ディレクトリが存在しません: {path}"
            
            if not path.is_dir():
                return f"ディレクトリではありません: {path}"
            
            shutil.rmtree(path)
            return f"ディレクトリを削除しました: {path}"
            
        except Exception as e:
            return f"ディレクトリ削除エラー: {str(e)}"
    
    async def _get_file_info(self, path: Path) -> str:
        """ファイル情報を取得"""
        
        try:
            if not path.exists():
                return f"ファイルが存在しません: {path}"
            
            stat = path.stat()
            
            info = {
                "パス": str(path),
                "名前": path.name,
                "拡張子": path.suffix,
                "サイズ": f"{stat.st_size} bytes",
                "作成日時": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "更新日時": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "アクセス日時": datetime.fromtimestamp(stat.st_atime).strftime("%Y-%m-%d %H:%M:%S"),
                "権限": oct(stat.st_mode)[-3:],
                "タイプ": "ディレクトリ" if path.is_dir() else "ファイル"
            }
            
            # MIMEタイプ
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type:
                info["MIMEタイプ"] = mime_type
            
            # ファイルの場合はハッシュ値も計算
            if path.is_file() and stat.st_size <= self.max_file_size:
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                        info["MD5"] = hashlib.md5(content).hexdigest()
                        info["SHA256"] = hashlib.sha256(content).hexdigest()
                except Exception:
                    pass
            
            return "\n".join([f"{k}: {v}" for k, v in info.items()])
            
        except Exception as e:
            return f"ファイル情報取得エラー: {str(e)}"
    
    async def _search_files(self, path: Path, pattern: str) -> str:
        """ファイルを検索"""
        
        try:
            if not path.exists():
                return f"パスが存在しません: {path}"
            
            if not path.is_dir():
                return f"ディレクトリではありません: {path}"
            
            if not pattern:
                return "検索パターンを指定してください。"
            
            found_files = []
            
            # ファイル名で検索
            for item in path.rglob("*"):
                if item.is_file() and pattern.lower() in item.name.lower():
                    found_files.append(str(item.relative_to(path)))
            
            if not found_files:
                return f"パターン '{pattern}' に一致するファイルが見つかりませんでした。"
            
            return f"検索結果 ({pattern}):\n\n" + "\n".join(found_files)
            
        except Exception as e:
            return f"ファイル検索エラー: {str(e)}"
    
    def _is_allowed_extension(self, path: Path) -> bool:
        """拡張子が許可されているかチェック"""
        
        if not self.allowed_extensions:
            return True
        
        return path.suffix.lower() in self.allowed_extensions


class TextProcessorTool(BaseTool):
    """テキスト処理ツール"""
    
    name: str = "text_processor"
    description: str = "テキストファイルの処理を行います。"
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def _run(self, action: str, text: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(action, text, **kwargs))
    
    async def _arun(self, action: str, text: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not action or not text:
                return "アクションとテキストを指定してください。"
            
            self.logger.info(f"Text processing: {action}")
            
            if action == "count_words":
                return await self._count_words(text)
            elif action == "count_lines":
                return await self._count_lines(text)
            elif action == "count_chars":
                return await self._count_chars(text)
            elif action == "extract_lines":
                start = kwargs.get("start", 1)
                end = kwargs.get("end", None)
                return await self._extract_lines(text, start, end)
            elif action == "search_text":
                pattern = kwargs.get("pattern", "")
                return await self._search_text(text, pattern)
            elif action == "replace_text":
                old_text = kwargs.get("old_text", "")
                new_text = kwargs.get("new_text", "")
                return await self._replace_text(text, old_text, new_text)
            elif action == "format_json":
                return await self._format_json(text)
            elif action == "validate_json":
                return await self._validate_json(text)
            else:
                return f"サポートされていないアクション: {action}"
            
        except Exception as e:
            self.logger.error(f"Text processing error: {e}")
            return f"テキスト処理エラー: {str(e)}"
    
    async def _count_words(self, text: str) -> str:
        """単語数をカウント"""
        
        words = text.split()
        return f"単語数: {len(words)}"
    
    async def _count_lines(self, text: str) -> str:
        """行数をカウント"""
        
        lines = text.split('\n')
        return f"行数: {len(lines)}"
    
    async def _count_chars(self, text: str) -> str:
        """文字数をカウント"""
        
        return f"文字数: {len(text)}"
    
    async def _extract_lines(self, text: str, start: int, end: Optional[int]) -> str:
        """指定行を抽出"""
        
        lines = text.split('\n')
        
        if start < 1:
            start = 1
        if end is None:
            end = len(lines)
        if end > len(lines):
            end = len(lines)
        
        extracted_lines = lines[start-1:end]
        return f"行 {start}-{end}:\n\n" + "\n".join(extracted_lines)
    
    async def _search_text(self, text: str, pattern: str) -> str:
        """テキストを検索"""
        
        if not pattern:
            return "検索パターンを指定してください。"
        
        lines = text.split('\n')
        matches = []
        
        for i, line in enumerate(lines, 1):
            if pattern.lower() in line.lower():
                matches.append(f"行 {i}: {line.strip()}")
        
        if not matches:
            return f"パターン '{pattern}' が見つかりませんでした。"
        
        return f"検索結果 ({pattern}):\n\n" + "\n".join(matches)
    
    async def _replace_text(self, text: str, old_text: str, new_text: str) -> str:
        """テキストを置換"""
        
        if not old_text:
            return "置換対象のテキストを指定してください。"
        
        replaced_text = text.replace(old_text, new_text)
        return f"置換結果:\n\n{replaced_text}"
    
    async def _format_json(self, text: str) -> str:
        """JSONを整形"""
        
        try:
            data = json.loads(text)
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            return f"整形されたJSON:\n\n{formatted}"
        except json.JSONDecodeError as e:
            return f"JSON解析エラー: {str(e)}"
    
    async def _validate_json(self, text: str) -> str:
        """JSONを検証"""
        
        try:
            json.loads(text)
            return "JSONは有効です。"
        except json.JSONDecodeError as e:
            return f"JSONは無効です: {str(e)}"