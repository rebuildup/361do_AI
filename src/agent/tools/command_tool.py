"""
Command Tool
システムコマンド実行機能を提供するツール
"""

import asyncio
import os
import platform
import subprocess
from typing import Any, Dict, List, Optional

from loguru import logger


class CommandTool:
    """システムコマンド実行ツール"""

    def __init__(self, allowed_commands: Optional[List[str]] = None):
        """
        Args:
            allowed_commands: 許可されたコマンドのリスト。Noneの場合は基本的なシステム情報コマンドのみ許可
        """
        self.allowed_commands = allowed_commands or [
            'systeminfo', 'dir', 'ls', 'pwd', 'whoami', 'hostname', 
            'ipconfig', 'ifconfig', 'ping', 'netstat', 'tasklist', 'ps',
            'echo', 'date', 'time', 'uname', 'df', 'free', 'top', 'htop'
        ]
        self.system = platform.system().lower()

    async def initialize(self):
        """ツール初期化"""
        logger.info("Initializing Command Tool...")
        logger.info(f"System: {self.system}")
        logger.info(f"Allowed commands: {', '.join(self.allowed_commands)}")

    async def close(self):
        """ツール終了"""
        pass

    async def execute_command(
        self,
        command: str,
        timeout: int = 30,
        capture_output: bool = True
    ) -> Dict[str, Any]:
        """
        コマンド実行
        
        Args:
            command: 実行するコマンド
            timeout: タイムアウト秒数
            capture_output: 出力をキャプチャするかどうか
            
        Returns:
            実行結果の辞書
        """
        try:
            # コマンドの安全性チェック
            if not self._is_command_allowed(command):
                return {
                    'success': False,
                    'error': f'Command not allowed: {command.split()[0]}',
                    'stdout': '',
                    'stderr': '',
                    'return_code': -1
                }

            logger.info(f"Executing command: {command}")

            # Windows環境での調整
            if self.system == 'windows':
                # PowerShellまたはcmdでの実行
                if command.startswith('systeminfo') or command.startswith('dir'):
                    shell_command = ['cmd', '/c', command]
                else:
                    shell_command = ['powershell', '-Command', command]
            else:
                # Unix系での実行
                shell_command = ['bash', '-c', command]

            # 非同期でコマンド実行
            process = await asyncio.create_subprocess_exec(
                *shell_command,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                cwd=os.getcwd()
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ''
                stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ''
                
                return {
                    'success': process.returncode == 0,
                    'stdout': stdout_text,
                    'stderr': stderr_text,
                    'return_code': process.returncode,
                    'command': command
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    'success': False,
                    'error': f'Command timed out after {timeout} seconds',
                    'stdout': '',
                    'stderr': '',
                    'return_code': -1,
                    'command': command
                }

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': '',
                'return_code': -1,
                'command': command
            }

    def _is_command_allowed(self, command: str) -> bool:
        """コマンドが許可されているかチェック"""
        if not command.strip():
            return False
            
        # コマンドの最初の単語（実際のコマンド名）を取得
        cmd_name = command.strip().split()[0].lower()
        
        # パスが含まれている場合はベース名のみ取得
        if '/' in cmd_name or '\\' in cmd_name:
            cmd_name = os.path.basename(cmd_name)
        
        # 拡張子を除去（Windows用）
        if cmd_name.endswith('.exe'):
            cmd_name = cmd_name[:-4]
            
        return cmd_name in [cmd.lower() for cmd in self.allowed_commands]

    async def get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            info = {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version()
            }

            # Windows固有の情報
            if self.system == 'windows':
                systeminfo_result = await self.execute_command('systeminfo')
                if systeminfo_result['success']:
                    info['systeminfo'] = systeminfo_result['stdout']
                    
                # IPアドレス情報
                ipconfig_result = await self.execute_command('ipconfig')
                if ipconfig_result['success']:
                    info['network'] = ipconfig_result['stdout']

            # Unix系固有の情報
            else:
                uname_result = await self.execute_command('uname -a')
                if uname_result['success']:
                    info['uname'] = uname_result['stdout']
                    
                # IPアドレス情報
                ifconfig_result = await self.execute_command('ifconfig')
                if ifconfig_result['success']:
                    info['network'] = ifconfig_result['stdout']

            return info

        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}

    async def get_status(self) -> str:
        """ツールステータス取得"""
        try:
            # 簡単なテストコマンド実行
            if self.system == 'windows':
                test_result = await self.execute_command('echo test')
            else:
                test_result = await self.execute_command('echo test')
                
            if test_result['success']:
                return "active"
            else:
                return "error"
        except Exception:
            return "error"

    def get_allowed_commands(self) -> List[str]:
        """許可されたコマンド一覧取得"""
        return self.allowed_commands.copy()

    def add_allowed_command(self, command: str) -> bool:
        """許可コマンドを追加"""
        try:
            if command not in self.allowed_commands:
                self.allowed_commands.append(command)
                logger.info(f"Added allowed command: {command}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add command: {e}")
            return False

    def remove_allowed_command(self, command: str) -> bool:
        """許可コマンドを削除"""
        try:
            if command in self.allowed_commands:
                self.allowed_commands.remove(command)
                logger.info(f"Removed allowed command: {command}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove command: {e}")
            return False