"""
Command Executor Tool

コマンド実行機能を提供するツール
"""

import asyncio
import subprocess
import logging
import os
import shlex
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import tempfile
import json

from langchain.tools import BaseTool


class CommandExecutorTool(BaseTool):
    """コマンド実行ツール"""
    
    name: str = "command_executor"
    description: str = "システムコマンドを実行します。安全なコマンドのみ実行可能です。"
    
    def __init__(self, 
                 allowed_commands: Optional[List[str]] = None,
                 timeout: int = 30,
                 working_directory: Optional[str] = None,
                 max_output_length: int = 10000):
        super().__init__()
        self.allowed_commands = allowed_commands or [
            "ls", "dir", "pwd", "cd", "cat", "type", "echo", "date", "time",
            "python", "pip", "git", "npm", "node", "curl", "wget", "find",
            "grep", "head", "tail", "wc", "sort", "uniq", "ps", "top", "df",
            "du", "free", "uname", "whoami", "id", "env", "which", "where"
        ]
        self.timeout = timeout
        self.working_directory = working_directory or os.getcwd()
        self.max_output_length = max_output_length
        self.logger = logging.getLogger(__name__)
    
    def _run(self, command: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(command, **kwargs))
    
    async def _arun(self, command: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not command.strip():
                return "コマンドが指定されていません。"
            
            # コマンドの安全性チェック
            if not self._is_command_allowed(command):
                return f"許可されていないコマンドです: {command}"
            
            self.logger.info(f"Executing command: {command}")
            
            # コマンドを解析
            parsed_command = self._parse_command(command)
            if not parsed_command:
                return "無効なコマンドです。"
            
            # コマンド実行
            result = await self._execute_command(parsed_command)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command execution error: {e}")
            return f"コマンド実行エラー: {str(e)}"
    
    def _is_command_allowed(self, command: str) -> bool:
        """コマンドが許可されているかチェック"""
        
        try:
            # コマンドを解析
            parts = shlex.split(command)
            if not parts:
                return False
            
            base_command = parts[0].lower()
            
            # 危険なコマンドをブロック
            dangerous_commands = [
                "rm", "del", "format", "fdisk", "mkfs", "dd", "shutdown",
                "reboot", "halt", "poweroff", "init", "kill", "killall",
                "pkill", "xkill", "sudo", "su", "passwd", "chmod", "chown",
                "chgrp", "umount", "mount", "iptables", "ufw", "firewall"
            ]
            
            if base_command in dangerous_commands:
                return False
            
            # 許可されたコマンドかチェック
            if self.allowed_commands and base_command not in self.allowed_commands:
                return False
            
            # パイプやリダイレクトのチェック
            if any(char in command for char in ['|', '>', '<', '&', ';', '&&', '||']):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Command validation error: {e}")
            return False
    
    def _parse_command(self, command: str) -> Optional[List[str]]:
        """コマンドを解析"""
        
        try:
            return shlex.split(command)
        except Exception as e:
            self.logger.error(f"Command parsing error: {e}")
            return None
    
    async def _execute_command(self, command_parts: List[str]) -> str:
        """コマンドを実行"""
        
        try:
            # プロセスを開始
            process = await asyncio.create_subprocess_exec(
                *command_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory
            )
            
            # タイムアウト付きで実行
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"コマンドがタイムアウトしました（{self.timeout}秒）"
            
            # 結果を整形
            output = ""
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='ignore')
                if len(stdout_text) > self.max_output_length:
                    stdout_text = stdout_text[:self.max_output_length] + "\n... (出力が切り詰められました)"
                output += f"標準出力:\n{stdout_text}"
            
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='ignore')
                if len(stderr_text) > self.max_output_length:
                    stderr_text = stderr_text[:self.max_output_length] + "\n... (エラー出力が切り詰められました)"
                output += f"\n\n標準エラー出力:\n{stderr_text}"
            
            if process.returncode != 0:
                output += f"\n\n終了コード: {process.returncode}"
            
            return output if output else "コマンドが正常に実行されました（出力なし）"
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return f"コマンド実行に失敗しました: {str(e)}"


class PythonExecutorTool(BaseTool):
    """Pythonコード実行ツール"""
    
    name: str = "python_executor"
    description: str = "Pythonコードを安全に実行します。"
    
    def __init__(self, 
                 timeout: int = 30,
                 max_output_length: int = 10000,
                 allowed_modules: Optional[List[str]] = None):
        super().__init__()
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.allowed_modules = allowed_modules or [
            "math", "random", "datetime", "json", "csv", "os", "sys",
            "pathlib", "collections", "itertools", "functools", "operator",
            "string", "re", "urllib", "http", "socket", "threading",
            "multiprocessing", "asyncio", "concurrent", "logging"
        ]
        self.logger = logging.getLogger(__name__)
    
    def _run(self, code: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(code, **kwargs))
    
    async def _arun(self, code: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not code.strip():
                return "Pythonコードが指定されていません。"
            
            # コードの安全性チェック
            if not self._is_code_safe(code):
                return "安全でないコードが検出されました。"
            
            self.logger.info("Executing Python code")
            
            # 一時ファイルにコードを書き込み
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Pythonコードを実行
                result = await self._execute_python_code(temp_file)
                return result
            finally:
                # 一時ファイルを削除
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.error(f"Python execution error: {e}")
            return f"Python実行エラー: {str(e)}"
    
    def _is_code_safe(self, code: str) -> bool:
        """コードが安全かチェック"""
        
        try:
            # 危険な操作をチェック
            dangerous_patterns = [
                "import os", "import sys", "import subprocess", "import shutil",
                "import socket", "import urllib", "import http", "import ftplib",
                "import smtplib", "import telnetlib", "import poplib", "import imaplib",
                "import ssl", "import ssl", "import hashlib", "import hmac",
                "import base64", "import binascii", "import pickle", "import marshal",
                "import shelve", "import dbm", "import sqlite3", "import pymongo",
                "import psycopg2", "import mysql", "import sqlalchemy",
                "__import__", "exec", "eval", "compile", "open", "file",
                "input", "raw_input", "exit", "quit", "help", "dir", "vars",
                "globals", "locals", "callable", "hasattr", "getattr", "setattr",
                "delattr", "isinstance", "issubclass", "type", "super"
            ]
            
            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if pattern in code_lower:
                    return False
            
            # ファイル操作のチェック
            file_operations = ["open(", "file(", "with open", "read(", "write("]
            for operation in file_operations:
                if operation in code_lower:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Code safety check error: {e}")
            return False
    
    async def _execute_python_code(self, file_path: str) -> str:
        """Pythonコードを実行"""
        
        try:
            # Pythonプロセスを開始
            process = await asyncio.create_subprocess_exec(
                "python", file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # タイムアウト付きで実行
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"Pythonコードがタイムアウトしました（{self.timeout}秒）"
            
            # 結果を整形
            output = ""
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='ignore')
                if len(stdout_text) > self.max_output_length:
                    stdout_text = stdout_text[:self.max_output_length] + "\n... (出力が切り詰められました)"
                output += f"実行結果:\n{stdout_text}"
            
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='ignore')
                if len(stderr_text) > self.max_output_length:
                    stderr_text = stderr_text[:self.max_output_length] + "\n... (エラー出力が切り詰められました)"
                output += f"\n\nエラー:\n{stderr_text}"
            
            if process.returncode != 0:
                output += f"\n\n終了コード: {process.returncode}"
            
            return output if output else "Pythonコードが正常に実行されました（出力なし）"
            
        except Exception as e:
            self.logger.error(f"Python execution failed: {e}")
            return f"Python実行に失敗しました: {str(e)}"


class SystemInfoTool(BaseTool):
    """システム情報取得ツール"""
    
    name: str = "system_info"
    description: str = "システムの情報を取得します。"
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def _run(self, info_type: str = "all", **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(info_type, **kwargs))
    
    async def _arun(self, info_type: str = "all", **kwargs) -> str:
        """非同期実行"""
        
        try:
            self.logger.info(f"Getting system info: {info_type}")
            
            if info_type == "all":
                return await self._get_all_info()
            elif info_type == "cpu":
                return await self._get_cpu_info()
            elif info_type == "memory":
                return await self._get_memory_info()
            elif info_type == "disk":
                return await self._get_disk_info()
            elif info_type == "network":
                return await self._get_network_info()
            elif info_type == "processes":
                return await self._get_process_info()
            else:
                return f"サポートされていない情報タイプ: {info_type}"
            
        except Exception as e:
            self.logger.error(f"System info error: {e}")
            return f"システム情報取得エラー: {str(e)}"
    
    async def _get_all_info(self) -> str:
        """全システム情報を取得"""
        
        info_parts = []
        
        # CPU情報
        cpu_info = await self._get_cpu_info()
        info_parts.append(f"=== CPU情報 ===\n{cpu_info}")
        
        # メモリ情報
        memory_info = await self._get_memory_info()
        info_parts.append(f"\n=== メモリ情報 ===\n{memory_info}")
        
        # ディスク情報
        disk_info = await self._get_disk_info()
        info_parts.append(f"\n=== ディスク情報 ===\n{disk_info}")
        
        # ネットワーク情報
        network_info = await self._get_network_info()
        info_parts.append(f"\n=== ネットワーク情報 ===\n{network_info}")
        
        return "\n".join(info_parts)
    
    async def _get_cpu_info(self) -> str:
        """CPU情報を取得"""
        
        try:
            import platform
            import psutil
            
            cpu_info = {
                "プロセッサ": platform.processor(),
                "アーキテクチャ": platform.architecture()[0],
                "物理コア数": psutil.cpu_count(logical=False),
                "論理コア数": psutil.cpu_count(logical=True),
                "CPU使用率": f"{psutil.cpu_percent(interval=1):.1f}%"
            }
            
            return "\n".join([f"{k}: {v}" for k, v in cpu_info.items()])
            
        except ImportError:
            return "psutilがインストールされていません。"
        except Exception as e:
            return f"CPU情報取得エラー: {str(e)}"
    
    async def _get_memory_info(self) -> str:
        """メモリ情報を取得"""
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_info = {
                "総メモリ": f"{memory.total / (1024**3):.2f} GB",
                "使用メモリ": f"{memory.used / (1024**3):.2f} GB",
                "空きメモリ": f"{memory.available / (1024**3):.2f} GB",
                "メモリ使用率": f"{memory.percent:.1f}%",
                "総スワップ": f"{swap.total / (1024**3):.2f} GB",
                "使用スワップ": f"{swap.used / (1024**3):.2f} GB",
                "スワップ使用率": f"{swap.percent:.1f}%"
            }
            
            return "\n".join([f"{k}: {v}" for k, v in memory_info.items()])
            
        except ImportError:
            return "psutilがインストールされていません。"
        except Exception as e:
            return f"メモリ情報取得エラー: {str(e)}"
    
    async def _get_disk_info(self) -> str:
        """ディスク情報を取得"""
        
        try:
            import psutil
            
            disk_info = []
            for partition in psutil.disk_partitions():
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        "デバイス": partition.device,
                        "マウントポイント": partition.mountpoint,
                        "ファイルシステム": partition.fstype,
                        "総容量": f"{partition_usage.total / (1024**3):.2f} GB",
                        "使用量": f"{partition_usage.used / (1024**3):.2f} GB",
                        "空き容量": f"{partition_usage.free / (1024**3):.2f} GB",
                        "使用率": f"{(partition_usage.used / partition_usage.total) * 100:.1f}%"
                    })
                except PermissionError:
                    continue
            
            if not disk_info:
                return "ディスク情報を取得できませんでした。"
            
            result = []
            for i, info in enumerate(disk_info, 1):
                result.append(f"ディスク {i}:")
                result.extend([f"  {k}: {v}" for k, v in info.items()])
                result.append("")
            
            return "\n".join(result)
            
        except ImportError:
            return "psutilがインストールされていません。"
        except Exception as e:
            return f"ディスク情報取得エラー: {str(e)}"
    
    async def _get_network_info(self) -> str:
        """ネットワーク情報を取得"""
        
        try:
            import psutil
            import socket
            
            network_info = []
            
            # ネットワークインターフェース
            for interface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        network_info.append(f"インターフェース: {interface}")
                        network_info.append(f"  IPアドレス: {addr.address}")
                        network_info.append(f"  ネットマスク: {addr.netmask}")
                        network_info.append("")
            
            # ネットワーク統計
            net_io = psutil.net_io_counters()
            network_info.extend([
                "ネットワーク統計:",
                f"  送信バイト数: {net_io.bytes_sent / (1024**2):.2f} MB",
                f"  受信バイト数: {net_io.bytes_recv / (1024**2):.2f} MB",
                f"  送信パケット数: {net_io.packets_sent}",
                f"  受信パケット数: {net_io.packets_recv}"
            ])
            
            return "\n".join(network_info)
            
        except ImportError:
            return "psutilがインストールされていません。"
        except Exception as e:
            return f"ネットワーク情報取得エラー: {str(e)}"
    
    async def _get_process_info(self) -> str:
        """プロセス情報を取得"""
        
        try:
            import psutil
            
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # CPU使用率でソート
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            
            result = ["上位10プロセス（CPU使用率順）:"]
            for i, proc in enumerate(processes[:10], 1):
                result.append(
                    f"{i:2d}. {proc['name']} (PID: {proc['pid']}) - "
                    f"CPU: {proc['cpu_percent']:.1f}%, "
                    f"メモリ: {proc['memory_percent']:.1f}%"
                )
            
            return "\n".join(result)
            
        except ImportError:
            return "psutilがインストールされていません。"
        except Exception as e:
            return f"プロセス情報取得エラー: {str(e)}"