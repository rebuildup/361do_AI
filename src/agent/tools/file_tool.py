import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from loguru import logger


class FileTool:
    """Tool for file operations within the project. Operations are limited to allowed paths for safety."""

    def __init__(self, project_root: str, allowed_dirs: Optional[List[str]] = None, auto_apply: bool = True, proposals_dir: Optional[str] = None):
        """Initialize FileTool.

        Args:
            project_root: repository/project root to which file operations are relative.
            allowed_dirs: list of directories (absolute or relative to project_root) that are allowed for edits.
            auto_apply: if True, writes are applied directly; otherwise writes create proposals under proposals_dir.
            proposals_dir: directory where proposals are saved (absolute or relative to project_root).
        """
        self.project_root = os.path.abspath(project_root)
        self.auto_apply = bool(auto_apply)
        # Determine proposals directory
        if proposals_dir:
            self.proposals_dir = os.path.abspath(proposals_dir) if os.path.isabs(proposals_dir) else os.path.abspath(os.path.join(self.project_root, proposals_dir))
        else:
            self.proposals_dir = os.path.abspath(os.path.join(self.project_root, 'src', 'data', 'proposals'))

        os.makedirs(self.proposals_dir, exist_ok=True)

        if allowed_dirs:
            self.allowed_paths = [os.path.abspath(p) if os.path.isabs(p) else os.path.abspath(os.path.join(self.project_root, p)) for p in allowed_dirs]
        else:
            self.allowed_paths = [
                os.path.abspath(os.path.join(self.project_root, 'src', 'data', 'prompts')),
                os.path.abspath(os.path.join(self.project_root, 'src', 'data', 'learning_data')),
            ]

        logger.info(f"FileTool initialized. Project root: {self.project_root}")
        logger.debug(f"Allowed paths: {self.allowed_paths}")

    def _is_safe_path(self, rel_path: str) -> bool:
        """
        Return True if rel_path (relative to project_root) is under one of the allowed paths.
        This is a security check to prevent directory traversal attacks.
        """
        # 1. Normalize the path to resolve '..' and '.'
        normalized_path = os.path.normpath(os.path.join(self.project_root, rel_path))

        # 2. Get the real path to resolve any symbolic links
        try:
            real_full_path = os.path.realpath(normalized_path)
        except OSError:
            # If the path doesn't exist, realpath can fail. Use the normalized path.
            real_full_path = normalized_path

        # 3. Check if the real path is within any of the allowed directories
        for allowed_path in self.allowed_paths:
            real_allowed_path = os.path.realpath(allowed_path)
            if real_full_path.startswith(real_allowed_path):
                return True
        
        logger.warning(f"Path traversal attempt detected or unsafe path access: {rel_path}")
        return False

    async def list_files(self, directory: str = '.') -> Dict[str, Any]:
        if not self._is_safe_path(directory):
            return {"error": f"アクセスが拒否されました: {directory}"}

        try:
            full = os.path.join(self.project_root, directory)
            items = os.listdir(full)
            return {"files": items}
        except FileNotFoundError:
            return {"error": f"ディレクトリが見つかりません: {directory}"}
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return {"error": str(e)}

    async def read_file(self, file_path: str) -> Dict[str, Any]:
        if not self._is_safe_path(file_path):
            return {"error": f"アクセスが拒否されました: {file_path}"}

        try:
            full = os.path.join(self.project_root, file_path)
            with open(full, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content}
        except FileNotFoundError:
            return {"error": f"ファイルが見つかりません: {file_path}"}
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {e}")
            return {"error": str(e)}

    async def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to file_path. If auto_apply is False and path is under allowed paths, create a proposal instead of applying."""
        if not self._is_safe_path(file_path):
            return {"error": f"アクセスが拒否されました: {file_path}"}

        full = os.path.join(self.project_root, file_path)

        # If target is in allowed paths and auto_apply is False, create a proposal
        abs_full = os.path.abspath(full)
        is_allowed = any(os.path.commonpath([p, abs_full]) == p for p in self.allowed_paths)

        if is_allowed and not self.auto_apply:
            # create proposal metadata
            proposal_id = f"proposal_{int(datetime.utcnow().timestamp())}_{uuid4().hex[:6]}"
            proposal_path = os.path.join(self.proposals_dir, f"{proposal_id}.json")
            meta = {
                "target": file_path,
                "content": content,
                "created_at": datetime.utcnow().isoformat(),
                "author": "agent"
            }
            try:
                with open(proposal_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                logger.info(f"Proposal created: {proposal_path}")
                return {"status": "proposal_created", "proposal_file": os.path.basename(proposal_path), "message": f"提案が作成されました: {os.path.basename(proposal_path)}"}
            except Exception as e:
                logger.error(f"Failed to create proposal: {e}")
                return {"error": str(e)}

        try:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"File written: {file_path}")
            return {"status": "success", "message": f"ファイル '{file_path}' が正常に書き込まれました。"}
        except Exception as e:
            logger.error(f"Error writing file '{file_path}': {e}")
            return {"error": str(e)}

    async def initialize(self):
        logger.info("FileTool async initialized")

    async def close(self):
        logger.info("FileTool closed")

