import os
from typing import List, Dict, Any
from loguru import logger

class FileTool:
    """
    エージェントがワークスペース内のファイルを操作するためのツール。
    セキュリティのため、操作はプロジェクトルート内に制限されます。
    """

    def __init__(self, project_root: str):
        """
        FileToolを初期化します。

        Args:
            project_root (str): 操作を許可するプロジェクトのルートディレクトリ。
        """
        self.project_root = os.path.abspath(project_root)
        logger.info(f"FileTool initialized. Project root set to: {self.project_root}")

    def _is_safe_path(self, path: str) -> bool:
        """指定されたパスがプロジェクトルート内にあるかを確認します。"""
        abs_path = os.path.abspath(os.path.join(self.project_root, path))
        return os.path.commonpath([self.project_root, abs_path]) == self.project_root

    async def list_files(self, directory: str = ".") -> Dict[str, Any]:
        """
        指定されたディレクトリ内のファイルとディレクトリを一覧表示します。

        Args:
            directory (str): 一覧表示するディレクトリの相対パス。

        Returns:
            Dict[str, Any]: ファイルとディレクトリのリスト、またはエラー情報。
        """
        if not self._is_safe_path(directory):
            return {"error": f"アクセスが拒否されました: {directory}"}

        try:
            full_path = os.path.join(self.project_root, directory)
            items = os.listdir(full_path)
            return {"files": items}
        except FileNotFoundError:
            return {"error": f"ディレクトリが見つかりません: {directory}"}
        except Exception as e:
            logger.error(f"Error listing files in '{directory}': {e}")
            return {"error": str(e)}

    async def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        指定されたファイルの内容を読み込みます。

        Args:
            file_path (str): 読み込むファイルの相対パス。

        Returns:
            Dict[str, Any]: ファイルの内容、またはエラー情報。
        """
        if not self._is_safe_path(file_path):
            return {"error": f"アクセスが拒否されました: {file_path}"}

        try:
            full_path = os.path.join(self.project_root, file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content}
        except FileNotFoundError:
            return {"error": f"ファイルが見つかりません: {file_path}"}
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {e}")
            return {"error": str(e)}

    async def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        指定されたファイルに内容を書き込みます。ファイルが存在しない場合は作成されます。

        Args:
            file_path (str): 書き込むファイルの相対パス。
            content (str): 書き込む内容。

        Returns:
            Dict[str, Any]: 成功メッセージ、またはエラー情報。
        """
        if not self._is_safe_path(file_path):
            return {"error": f"アクセスが拒否されました: {file_path}"}

        try:
            full_path = os.path.join(self.project_root, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"File '{file_path}' has been written successfully.")
            return {"status": "success", "message": f"ファイル '{file_path}' が正常に書き込まれました。"}
        except Exception as e:
            logger.error(f"Error writing to file '{file_path}': {e}")
            return {"error": str(e)}

    async def initialize(self):
        """ツールの非同期初期化（現バージョンでは何もしない）"""
        logger.info("FileTool initialized (async).")

    async def close(self):
        """ツールのクローズ処理（現バージョンでは何もしない）"""
        logger.info("FileTool closed.")
