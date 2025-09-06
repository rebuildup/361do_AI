"""
Configuration loader for self-learning AI agent
自己学習AIエージェント用設定読み込み
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import ValidationError

from .settings import AgentConfig, get_agent_config, set_agent_config

logger = logging.getLogger(__name__)


class ConfigLoader:
    """設定読み込みクラス"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 設定ファイルの優先順位
        self.config_files = [
            "agent_config.yaml",
            "agent_config.yml", 
            "agent_config.json",
            "config.yaml",
            "config.yml",
            "config.json"
        ]
    
    def load_config(self, 
                   config_file: Optional[str] = None,
                   env_prefix: str = "AGENT_",
                   validate: bool = True) -> AgentConfig:
        """
        設定読み込み
        
        Args:
            config_file: 設定ファイルパス（指定しない場合は自動検出）
            env_prefix: 環境変数プレフィックス
            validate: 設定検証フラグ
            
        Returns:
            AgentConfig: エージェント設定
        """
        try:
            # 1. 環境変数から設定読み込み
            config_dict = self._load_from_env(env_prefix)
            
            # 2. 設定ファイルから読み込み（環境変数を上書き）
            if config_file:
                file_config = self._load_from_file(config_file)
                config_dict.update(file_config)
            else:
                # 自動検出
                for config_filename in self.config_files:
                    config_path = self.config_dir / config_filename
                    if config_path.exists():
                        file_config = self._load_from_file(str(config_path))
                        config_dict.update(file_config)
                        logger.info(f"設定ファイル読み込み: {config_path}")
                        break
            
            # 3. 設定オブジェクト作成
            if validate:
                config = AgentConfig(**config_dict)
            else:
                config = AgentConfig.construct(**config_dict)
            
            # 4. グローバル設定に設定
            set_agent_config(config)
            
            logger.info("設定読み込み完了")
            return config
            
        except ValidationError as e:
            logger.error(f"設定検証エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            raise
    
    def _load_from_env(self, prefix: str) -> Dict[str, Any]:
        """環境変数から設定読み込み"""
        config_dict = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # プレフィックスを除去
                config_key = key[len(prefix):].lower()
                
                # ネストした設定のキーを処理
                keys = config_key.split('_')
                current_dict = config_dict
                
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = {}
                    current_dict = current_dict[k]
                
                # 値を設定
                final_key = keys[-1]
                current_dict[final_key] = self._parse_env_value(value)
        
        return config_dict
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool, list, dict]:
        """環境変数値をパース"""
        # ブール値
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 数値
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON配列・オブジェクト
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # カンマ区切りリスト
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # 文字列
        return value
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """ファイルから設定読み込み"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif file_path.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ValueError(f"サポートされていないファイル形式: {file_path.suffix}")
    
    def save_config(self, config: AgentConfig, file_path: str, format: str = "yaml"):
        """
        設定保存
        
        Args:
            config: エージェント設定
            file_path: 保存ファイルパス
            format: 保存形式 ("yaml" or "json")
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == "json":
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"サポートされていない形式: {format}")
        
        logger.info(f"設定保存完了: {file_path}")
    
    def create_default_config(self, file_path: str = "config/agent_config.yaml"):
        """デフォルト設定ファイル作成"""
        config = AgentConfig()
        self.save_config(config, file_path)
        logger.info(f"デフォルト設定ファイル作成: {file_path}")
        return config
    
    def validate_config(self, config: AgentConfig) -> Dict[str, Any]:
        """設定検証"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # 基本検証
            config.validate()
            
            # カスタム検証
            self._validate_paths(config, validation_result)
            self._validate_ports(config, validation_result)
            self._validate_resources(config, validation_result)
            
        except ValidationError as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(str(e))
        
        return validation_result
    
    def _validate_paths(self, config: AgentConfig, result: Dict[str, Any]):
        """パス検証"""
        paths_to_check = [
            config.data_dir,
            config.logs_dir,
            config.models_dir,
            config.cache_dir,
            config.temp_dir
        ]
        
        for path in paths_to_check:
            if not Path(path).exists():
                result["warnings"].append(f"ディレクトリが存在しません: {path}")
                result["suggestions"].append(f"mkdir -p {path}")
    
    def _validate_ports(self, config: AgentConfig, result: Dict[str, Any]):
        """ポート検証"""
        ports = [
            ("Prometheus", config.monitoring.prometheus_port),
            ("Grafana", config.monitoring.grafana_port),
            ("Streamlit", config.ui.streamlit_port),
            ("FastAPI", config.ui.fastapi_port),
            ("WebSocket", config.ui.websocket_port)
        ]
        
        used_ports = set()
        for name, port in ports:
            if port in used_ports:
                result["errors"].append(f"ポート重複: {name} ({port})")
            used_ports.add(port)
    
    def _validate_resources(self, config: AgentConfig, result: Dict[str, Any]):
        """リソース検証"""
        # メモリ設定検証
        if config.memory.max_memory_size > 100000:
            result["warnings"].append("最大メモリサイズが大きすぎます")
        
        # 学習設定検証
        if config.learning.learning_rate > 0.1:
            result["warnings"].append("学習率が高すぎる可能性があります")
        
        # 進化設定検証
        if config.evolution.population_size < 10:
            result["warnings"].append("個体群サイズが小さい可能性があります")
    
    def get_config_summary(self, config: AgentConfig) -> Dict[str, Any]:
        """設定サマリー取得"""
        return {
            "agent": {
                "name": config.name,
                "version": config.version,
                "description": config.description
            },
            "database": {
                "type": config.database.type,
                "path": config.database.path
            },
            "ollama": {
                "base_url": config.ollama.base_url,
                "model": config.ollama.model
            },
            "memory": {
                "backend": config.memory.backend,
                "max_size": config.memory.max_memory_size
            },
            "learning": {
                "mutation_rate": config.learning.prompt_mutation_rate,
                "generation_size": config.learning.evolution_generation_size
            },
            "monitoring": {
                "enabled": config.monitoring.enabled,
                "prometheus_port": config.monitoring.prometheus_port
            },
            "ui": {
                "streamlit_port": config.ui.streamlit_port,
                "fastapi_port": config.ui.fastapi_port
            }
        }


def load_config_from_file(file_path: str) -> AgentConfig:
    """設定ファイルから読み込み（便利関数）"""
    loader = ConfigLoader()
    return loader.load_config(config_file=file_path)


def create_default_config_file(file_path: str = "config/agent_config.yaml") -> AgentConfig:
    """デフォルト設定ファイル作成（便利関数）"""
    loader = ConfigLoader()
    return loader.create_default_config(file_path)


def validate_current_config() -> Dict[str, Any]:
    """現在の設定検証（便利関数）"""
    config = get_agent_config()
    loader = ConfigLoader()
    return loader.validate_config(config)


def get_config_summary() -> Dict[str, Any]:
    """現在の設定サマリー取得（便利関数）"""
    config = get_agent_config()
    loader = ConfigLoader()
    return loader.get_config_summary(config)
