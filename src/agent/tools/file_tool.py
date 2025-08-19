"""
File Tool
ファイル操作機能を提供するツール
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class FileTool:
    """ファイル操作ツール"""
    
    def __init__(self):
        self.allowed_extensions = {
            'text': ['.txt', '.md', '.json', '.yaml', '.yml', '.csv'],
            'code': ['.py', '.js', '.html', '.css', '.xml', '.sql'],
            'data': ['.json', '.csv', '.xml', '.yaml', '.yml'],
            'web': ['.html', '.css', '.js', '.json']
        }
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.safe_directory = Path("/app/data/files")  # Docker環境用
        
    async def initialize(self):
        """ツール初期化"""
        logger.info("Initializing File Tool...")
        
        # 安全なディレクトリを作成
        self.safe_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("File Tool initialized")
    
    async def read_file(
        self,
        file_path: Union[str, Path],
        encoding: str = 'utf-8'
    ) -> Dict[str, Any]:
        """ファイル読み込み"""
        try:
            file_path = Path(file_path)
            
            # セキュリティチェック
            if not self._is_safe_path(file_path):
                return {
                    'success': False,
                    'error': 'Access to this path is not allowed',
                    'path': str(file_path)
                }
            
            # ファイル存在確認
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File not found',
                    'path': str(file_path)
                }
            
            # ファイルサイズチェック
            if file_path.stat().st_size > self.max_file_size:
                return {
                    'success': False,
                    'error': f'File too large (max {self.max_file_size} bytes)',
                    'path': str(file_path)
                }
            
            # ファイル読み込み
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return {
                'success': True,
                'content': content,
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'encoding': encoding
            }
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {file_path}: {e}")
            return {
                'success': False,
                'error': f'Encoding error: {str(e)}',
                'path': str(file_path)
            }
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': str(file_path)
            }
    
    async def write_file(
        self,
        file_path: Union[str, Path],
        content: str,
        encoding: str = 'utf-8',
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """ファイル書き込み"""
        try:
            file_path = Path(file_path)
            
            # セキュリティチェック
            if not self._is_safe_path(file_path):
                return {
                    'success': False,
                    'error': 'Access to this path is not allowed',
                    'path': str(file_path)
                }
            
            # ディレクトリ作成
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # コンテンツサイズチェック
            if len(content.encode(encoding)) > self.max_file_size:
                return {
                    'success': False,
                    'error': f'Content too large (max {self.max_file_size} bytes)',
                    'path': str(file_path)
                }
            
            # ファイル書き込み
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return {
                'success': True,
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'encoding': encoding
            }
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': str(file_path)
            }
    
    async def create_directory(
        self,
        dir_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """ディレクトリ作成"""
        try:
            dir_path = Path(dir_path)
            
            # セキュリティチェック
            if not self._is_safe_path(dir_path):
                return {
                    'success': False,
                    'error': 'Access to this path is not allowed',
                    'path': str(dir_path)
                }
            
            # ディレクトリ作成
            dir_path.mkdir(parents=True, exist_ok=True)
            
            return {
                'success': True,
                'path': str(dir_path),
                'created': True
            }
            
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': str(dir_path)
            }
    
    async def list_directory(
        self,
        dir_path: Union[str, Path],
        include_hidden: bool = False
    ) -> Dict[str, Any]:
        """ディレクトリ一覧取得"""
        try:
            dir_path = Path(dir_path)
            
            # セキュリティチェック
            if not self._is_safe_path(dir_path):
                return {
                    'success': False,
                    'error': 'Access to this path is not allowed',
                    'path': str(dir_path)
                }
            
            # ディレクトリ存在確認
            if not dir_path.exists():
                return {
                    'success': False,
                    'error': 'Directory not found',
                    'path': str(dir_path)
                }
            
            if not dir_path.is_dir():
                return {
                    'success': False,
                    'error': 'Path is not a directory',
                    'path': str(dir_path)
                }
            
            # ファイル・ディレクトリ一覧取得
            items = []
            for item in dir_path.iterdir():
                # 隠しファイル・ディレクトリのフィルタリング
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                try:
                    stat_info = item.stat()
                    items.append({
                        'name': item.name,
                        'path': str(item),
                        'type': 'directory' if item.is_dir() else 'file',
                        'size': stat_info.st_size if item.is_file() else None,
                        'modified': stat_info.st_mtime,
                        'extension': item.suffix if item.is_file() else None
                    })
                except Exception as e:
                    logger.warning(f"Failed to get info for {item}: {e}")
                    continue
            
            return {
                'success': True,
                'path': str(dir_path),
                'items': sorted(items, key=lambda x: (x['type'], x['name'])),
                'total_items': len(items)
            }
            
        except Exception as e:
            logger.error(f"Failed to list directory {dir_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': str(dir_path)
            }
    
    async def delete_file(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """ファイル削除"""
        try:
            file_path = Path(file_path)
            
            # セキュリティチェック
            if not self._is_safe_path(file_path):
                return {
                    'success': False,
                    'error': 'Access to this path is not allowed',
                    'path': str(file_path)
                }
            
            # ファイル存在確認
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File not found',
                    'path': str(file_path)
                }
            
            # ファイル削除
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
            
            return {
                'success': True,
                'path': str(file_path),
                'deleted': True
            }
            
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': str(file_path)
            }
    
    async def copy_file(
        self,
        source_path: Union[str, Path],
        dest_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """ファイルコピー"""
        try:
            source_path = Path(source_path)
            dest_path = Path(dest_path)
            
            # セキュリティチェック
            if not self._is_safe_path(source_path) or not self._is_safe_path(dest_path):
                return {
                    'success': False,
                    'error': 'Access to this path is not allowed',
                    'source': str(source_path),
                    'destination': str(dest_path)
                }
            
            # ソースファイル存在確認
            if not source_path.exists():
                return {
                    'success': False,
                    'error': 'Source file not found',
                    'source': str(source_path),
                    'destination': str(dest_path)
                }
            
            # 宛先ディレクトリ作成
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # ファイルコピー
            if source_path.is_file():
                shutil.copy2(source_path, dest_path)
            elif source_path.is_dir():
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            
            return {
                'success': True,
                'source': str(source_path),
                'destination': str(dest_path),
                'copied': True
            }
            
        except Exception as e:
            logger.error(f"Failed to copy {source_path} to {dest_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'source': str(source_path),
                'destination': str(dest_path)
            }
    
    async def create_temp_file(
        self,
        content: str,
        suffix: str = '.txt',
        encoding: str = 'utf-8'
    ) -> Dict[str, Any]:
        """一時ファイル作成"""
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=suffix,
                encoding=encoding,
                delete=False,
                dir=self.safe_directory
            ) as f:
                f.write(content)
                temp_path = f.name
            
            return {
                'success': True,
                'path': temp_path,
                'size': len(content.encode(encoding)),
                'temporary': True
            }
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            return {
                'success': False,
                'error': str(e),
                'temporary': True
            }
    
    def _is_safe_path(self, path: Path) -> bool:
        """パスの安全性チェック"""
        try:
            # 絶対パスに変換
            abs_path = path.resolve()
            safe_abs_path = self.safe_directory.resolve()
            
            # 安全なディレクトリ内かチェック
            return str(abs_path).startswith(str(safe_abs_path))
            
        except Exception:
            return False
    
    def _is_allowed_extension(self, file_path: Path, category: str = None) -> bool:
        """拡張子の許可チェック"""
        suffix = file_path.suffix.lower()
        
        if category:
            return suffix in self.allowed_extensions.get(category, [])
        else:
            # すべてのカテゴリをチェック
            all_extensions = []
            for exts in self.allowed_extensions.values():
                all_extensions.extend(exts)
            return suffix in all_extensions
    
    async def get_file_info(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """ファイル情報取得"""
        try:
            file_path = Path(file_path)
            
            # セキュリティチェック
            if not self._is_safe_path(file_path):
                return {
                    'success': False,
                    'error': 'Access to this path is not allowed',
                    'path': str(file_path)
                }
            
            # ファイル存在確認
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File not found',
                    'path': str(file_path)
                }
            
            stat_info = file_path.stat()
            
            return {
                'success': True,
                'path': str(file_path),
                'name': file_path.name,
                'type': 'directory' if file_path.is_dir() else 'file',
                'size': stat_info.st_size,
                'extension': file_path.suffix if file_path.is_file() else None,
                'created': stat_info.st_ctime,
                'modified': stat_info.st_mtime,
                'accessed': stat_info.st_atime,
                'is_readable': os.access(file_path, os.R_OK),
                'is_writable': os.access(file_path, os.W_OK)
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': str(file_path)
            }
    
    async def get_status(self) -> str:
        """ツールステータス取得"""
        try:
            # 安全ディレクトリの存在確認
            if self.safe_directory.exists() and self.safe_directory.is_dir():
                return "active"
            else:
                return "error"
        except Exception:
            return "error"
