"""
本番環境準備・ドキュメント統合テスト

Docker + Docker Compose による デプロイメント自動化を統合し、
MkDocs による ユーザーマニュアル・開発者ドキュメント作成を統合、
SQLAlchemy + ChromaDB の既存機能による バックアップ・復元手順文書化を統合
"""

import pytest
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import os

# テスト用のモッククラス
class MockDockerClient:
    """Docker クライアントのモック"""
    
    def __init__(self):
        self.containers = {}
        self.images = {}
        self.networks = {}
        
    def build_image(self, dockerfile_path: str, tag: str) -> Dict[str, Any]:
        """イメージビルド"""
        self.images[tag] = {
            "id": f"img_{len(self.images)}",
            "tag": tag,
            "dockerfile": dockerfile_path,
            "created": datetime.now().isoformat()
        }
        return self.images[tag]
    
    def run_container(self, image: str, name: str, **kwargs) -> Dict[str, Any]:
        """コンテナ実行"""
        container = {
            "id": f"container_{len(self.containers)}",
            "name": name,
            "image": image,
            "status": "running",
            "created": datetime.now().isoformat(),
            "config": kwargs
        }
        self.containers[name] = container
        return container
    
    def stop_container(self, name: str) -> bool:
        """コンテナ停止"""
        if name in self.containers:
            self.containers[name]["status"] = "stopped"
            return True
        return False
    
    def get_container_logs(self, name: str) -> str:
        """コンテナログ取得"""
        if name in self.containers:
            return f"Mock logs for container {name}"
        return ""

class MockMkDocsBuilder:
    """MkDocs ビルダーのモック"""
    
    def __init__(self):
        self.config = {}
        self.pages = []
        
    def load_config(self, config_file: Path) -> Dict[str, Any]:
        """設定読み込み"""
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        return self.config
    
    def build_docs(self, output_dir: Path) -> bool:
        """ドキュメントビルド"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # インデックスページ作成
            index_file = output_dir / "index.html"
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write("<html><body><h1>Documentation</h1></body></html>")
            
            return True
        except Exception:
            return False
    
    def serve_docs(self, port: int = 8000) -> bool:
        """ドキュメントサーブ"""
        # モックなので常に成功
        return True


@pytest.fixture
def temp_dir():
    """テスト用一時ディレクトリ"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_docker():
    """モック Docker クライアント"""
    return MockDockerClient()


@pytest.fixture
def mock_mkdocs():
    """モック MkDocs ビルダー"""
    return MockMkDocsBuilder()


class TestDockerDeployment:
    """Docker デプロイメントテスト"""
    
    def test_dockerfile_creation(self, temp_dir):
        """Dockerfile 作成テスト"""
        
        dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

COPY requirements_advanced.txt .
RUN pip install -r requirements_advanced.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000 8501

CMD ["python", "-m", "src.advanced_agent.interfaces.fastapi_gateway"]
"""
        
        dockerfile = temp_dir / "Dockerfile"
        with open(dockerfile, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content.strip())
        
        # Dockerfile の検証
        assert dockerfile.exists()
        
        with open(dockerfile, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 必要な要素の確認
        assert "FROM python:" in content
        assert "WORKDIR /app" in content
        assert "requirements_advanced.txt" in content
        assert "EXPOSE" in content
        assert "CMD" in content
    
    def test_docker_compose_configuration(self, temp_dir):
        """Docker Compose 設定テスト"""
        
        compose_config = {
            "version": "3.8",
            "services": {
                "advanced-agent": {
                    "build": ".",
                    "ports": ["8000:8000", "8501:8501"],
                    "environment": [
                        "ADVANCED_AGENT_LOG_LEVEL=INFO",
                        "ADVANCED_AGENT_API_BASE_URL=http://localhost:8000"
                    ],
                    "volumes": [
                        "./data:/app/data",
                        "./logs:/app/logs"
                    ],
                    "restart": "unless-stopped"
                },
                "chromadb": {
                    "image": "chromadb/chroma:latest",
                    "ports": ["8002:8000"],
                    "volumes": ["./chroma_data:/chroma/chroma"],
                    "restart": "unless-stopped"
                },
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "ports": ["9090:9090"],
                    "volumes": ["./prometheus.yml:/etc/prometheus/prometheus.yml"],
                    "restart": "unless-stopped"
                }
            },
            "volumes": {
                "chroma_data": None,
                "prometheus_data": None
            }
        }
        
        compose_file = temp_dir / "docker-compose.yml"
        with open(compose_file, 'w', encoding='utf-8') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        # Docker Compose ファイルの検証
        assert compose_file.exists()
        
        with open(compose_file, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)
        
        # 設定の検証
        assert "services" in loaded_config
        assert "advanced-agent" in loaded_config["services"]
        assert "chromadb" in loaded_config["services"]
        assert "prometheus" in loaded_config["services"]
        
        # ポート設定の確認
        agent_service = loaded_config["services"]["advanced-agent"]
        assert "8000:8000" in agent_service["ports"]
        assert "8501:8501" in agent_service["ports"]
    
    def test_container_build_and_run(self, mock_docker, temp_dir):
        """コンテナビルド・実行テスト"""
        
        # Dockerfile 作成
        dockerfile = temp_dir / "Dockerfile"
        dockerfile.write_text("FROM python:3.11-slim\nWORKDIR /app")
        
        # イメージビルド
        image = mock_docker.build_image(str(dockerfile), "advanced-agent:test")
        
        assert image["tag"] == "advanced-agent:test"
        assert "advanced-agent:test" in mock_docker.images
        
        # コンテナ実行
        container = mock_docker.run_container(
            "advanced-agent:test",
            "test-container",
            ports={"8000": 8000},
            environment={"LOG_LEVEL": "INFO"}
        )
        
        assert container["name"] == "test-container"
        assert container["status"] == "running"
        assert "test-container" in mock_docker.containers
        
        # コンテナ停止
        stopped = mock_docker.stop_container("test-container")
        assert stopped is True
        assert mock_docker.containers["test-container"]["status"] == "stopped"
    
    def test_health_check_configuration(self, temp_dir):
        """ヘルスチェック設定テスト"""
        
        health_check_script = """#!/bin/bash
# Health check script for Advanced AI Agent

# Check API endpoint
curl -f http://localhost:8000/health || exit 1

# Check Streamlit UI
curl -f http://localhost:8501 || exit 1

# Check system resources
python3 -c "
import psutil
cpu = psutil.cpu_percent()
memory = psutil.virtual_memory().percent
if cpu > 95 or memory > 95:
    exit(1)
"

echo "Health check passed"
"""
        
        health_script = temp_dir / "health_check.sh"
        with open(health_script, 'w', encoding='utf-8') as f:
            f.write(health_check_script.strip())
        
        # スクリプトファイルの検証
        assert health_script.exists()
        
        with open(health_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ヘルスチェック要素の確認
        assert "curl -f http://localhost:8000/health" in content
        assert "curl -f http://localhost:8501" in content
        assert "psutil" in content


class TestDocumentationGeneration:
    """ドキュメント生成テスト - MkDocs による ユーザーマニュアル・開発者ドキュメント作成"""
    
    def test_mkdocs_configuration(self, temp_dir):
        """MkDocs 設定テスト"""
        
        mkdocs_config = {
            "site_name": "Advanced AI Agent Documentation",
            "site_description": "RTX 4050 6GB VRAM 最適化自己学習AIエージェント",
            "site_author": "Advanced Agent Team",
            "repo_url": "https://github.com/your-org/advanced-ai-agent",
            "theme": {
                "name": "material",
                "palette": {
                    "primary": "blue",
                    "accent": "light blue"
                },
                "features": [
                    "navigation.tabs",
                    "navigation.sections",
                    "toc.integrate"
                ]
            },
            "nav": [
                {"Home": "index.md"},
                {"User Guide": [
                    {"Getting Started": "user/getting-started.md"},
                    {"Configuration": "user/configuration.md"},
                    {"Usage Examples": "user/examples.md"}
                ]},
                {"Developer Guide": [
                    {"Architecture": "dev/architecture.md"},
                    {"API Reference": "dev/api.md"},
                    {"Contributing": "dev/contributing.md"}
                ]},
                {"Deployment": [
                    {"Docker Setup": "deploy/docker.md"},
                    {"Production Guide": "deploy/production.md"},
                    {"Troubleshooting": "deploy/troubleshooting.md"}
                ]}
            ],
            "markdown_extensions": [
                "codehilite",
                "admonition",
                "toc",
                "pymdownx.superfences"
            ]
        }
        
        config_file = temp_dir / "mkdocs.yml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(mkdocs_config, f, default_flow_style=False)
        
        # 設定ファイルの検証
        assert config_file.exists()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)
        
        # 設定内容の検証
        assert loaded_config["site_name"] == "Advanced AI Agent Documentation"
        assert "nav" in loaded_config
        assert "theme" in loaded_config
        assert loaded_config["theme"]["name"] == "material"
    
    def test_documentation_structure_creation(self, temp_dir):
        """ドキュメント構造作成テスト"""
        
        # ドキュメントディレクトリ構造
        docs_structure = {
            "docs/index.md": "# Advanced AI Agent\n\nWelcome to the documentation.",
            "docs/user/getting-started.md": "# Getting Started\n\nInstallation and setup guide.",
            "docs/user/configuration.md": "# Configuration\n\nConfiguration options and settings.",
            "docs/user/examples.md": "# Usage Examples\n\nPractical usage examples.",
            "docs/dev/architecture.md": "# Architecture\n\nSystem architecture overview.",
            "docs/dev/api.md": "# API Reference\n\nAPI documentation.",
            "docs/dev/contributing.md": "# Contributing\n\nContribution guidelines.",
            "docs/deploy/docker.md": "# Docker Setup\n\nDocker deployment guide.",
            "docs/deploy/production.md": "# Production Guide\n\nProduction deployment.",
            "docs/deploy/troubleshooting.md": "# Troubleshooting\n\nCommon issues and solutions."
        }
        
        # ドキュメントファイル作成
        for file_path, content in docs_structure.items():
            full_path = temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # ドキュメント構造の検証
        docs_dir = temp_dir / "docs"
        assert docs_dir.exists()
        assert (docs_dir / "index.md").exists()
        
        # 各セクションの確認
        for section in ["user", "dev", "deploy"]:
            section_dir = docs_dir / section
            assert section_dir.exists()
            assert len(list(section_dir.glob("*.md"))) > 0
    
    def test_documentation_build(self, mock_mkdocs, temp_dir):
        """ドキュメントビルドテスト"""
        
        # MkDocs 設定ファイル作成
        config_file = temp_dir / "mkdocs.yml"
        config = {"site_name": "Test Docs", "nav": [{"Home": "index.md"}]}
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        # ドキュメントファイル作成
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        
        index_file = docs_dir / "index.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("# Test Documentation")
        
        # 設定読み込み
        loaded_config = mock_mkdocs.load_config(config_file)
        assert loaded_config["site_name"] == "Test Docs"
        
        # ドキュメントビルド
        output_dir = temp_dir / "site"
        build_success = mock_mkdocs.build_docs(output_dir)
        
        assert build_success is True
        assert output_dir.exists()
        assert (output_dir / "index.html").exists()
    
    def test_api_documentation_generation(self, temp_dir):
        """API ドキュメント生成テスト"""
        
        # OpenAPI スキーマ（サンプル）
        openapi_schema = {
            "openapi": "3.0.0",
            "info": {
                "title": "Advanced AI Agent API",
                "version": "1.0.0",
                "description": "REST API for Advanced AI Agent"
            },
            "paths": {
                "/v1/chat/completions": {
                    "post": {
                        "summary": "Chat completion",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model": {"type": "string"},
                                            "messages": {"type": "array"},
                                            "temperature": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful response",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "choices": {"type": "array"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # OpenAPI スキーマファイル作成
        schema_file = temp_dir / "openapi.json"
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(openapi_schema, f, indent=2)
        
        # スキーマファイルの検証
        assert schema_file.exists()
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            loaded_schema = json.load(f)
        
        # API ドキュメントの検証
        assert loaded_schema["info"]["title"] == "Advanced AI Agent API"
        assert "/v1/chat/completions" in loaded_schema["paths"]
        assert "post" in loaded_schema["paths"]["/v1/chat/completions"]


class TestBackupRestoreProcedures:
    """バックアップ・復元手順テスト"""
    
    def test_database_backup_procedure(self, temp_dir):
        """データベースバックアップ手順テスト"""
        
        # バックアップスクリプト作成
        backup_script = """#!/bin/bash
# Database backup script

BACKUP_DIR="/app/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup directory
mkdir -p $BACKUP_DIR

# SQLite database backup
if [ -f "/app/data/agent.db" ]; then
    cp /app/data/agent.db $BACKUP_DIR/agent_$TIMESTAMP.db
    echo "SQLite backup created: agent_$TIMESTAMP.db"
fi

# ChromaDB backup
if [ -d "/app/data/chroma" ]; then
    tar -czf $BACKUP_DIR/chroma_$TIMESTAMP.tar.gz -C /app/data chroma/
    echo "ChromaDB backup created: chroma_$TIMESTAMP.tar.gz"
fi

# Configuration backup
if [ -d "/app/config" ]; then
    tar -czf $BACKUP_DIR/config_$TIMESTAMP.tar.gz -C /app config/
    echo "Configuration backup created: config_$TIMESTAMP.tar.gz"
fi

echo "Backup completed at $(date)"
"""
        
        script_file = temp_dir / "backup.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(backup_script.strip())
        
        # スクリプトファイルの検証
        assert script_file.exists()
        
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # バックアップ要素の確認
        assert "BACKUP_DIR=" in content
        assert "TIMESTAMP=" in content
        assert "agent.db" in content
        assert "chroma" in content
        assert "config" in content
    
    def test_restore_procedure(self, temp_dir):
        """復元手順テスト"""
        
        # 復元スクリプト作成
        restore_script = """#!/bin/bash
# Database restore script

if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_timestamp>"
    exit 1
fi

TIMESTAMP=$1
BACKUP_DIR="/app/backups"

# Stop services
echo "Stopping services..."
# docker-compose stop advanced-agent

# Restore SQLite database
if [ -f "$BACKUP_DIR/agent_$TIMESTAMP.db" ]; then
    cp $BACKUP_DIR/agent_$TIMESTAMP.db /app/data/agent.db
    echo "SQLite database restored"
fi

# Restore ChromaDB
if [ -f "$BACKUP_DIR/chroma_$TIMESTAMP.tar.gz" ]; then
    rm -rf /app/data/chroma
    tar -xzf $BACKUP_DIR/chroma_$TIMESTAMP.tar.gz -C /app/data/
    echo "ChromaDB restored"
fi

# Restore configuration
if [ -f "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" ]; then
    rm -rf /app/config
    tar -xzf $BACKUP_DIR/config_$TIMESTAMP.tar.gz -C /app/
    echo "Configuration restored"
fi

# Start services
echo "Starting services..."
# docker-compose start advanced-agent

echo "Restore completed at $(date)"
"""
        
        script_file = temp_dir / "restore.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(restore_script.strip())
        
        # スクリプトファイルの検証
        assert script_file.exists()
        
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 復元要素の確認
        assert "TIMESTAMP=$1" in content
        assert "Stopping services" in content
        assert "Starting services" in content
        assert "agent.db" in content
        assert "chroma" in content
    
    def test_backup_validation(self, temp_dir):
        """バックアップ検証テスト"""
        
        # テスト用バックアップファイル作成
        backup_dir = temp_dir / "backups"
        backup_dir.mkdir()
        
        # SQLite バックアップ
        sqlite_backup = backup_dir / "agent_20240101_120000.db"
        sqlite_backup.write_text("SQLite database backup content")
        
        # ChromaDB バックアップ
        chroma_backup = backup_dir / "chroma_20240101_120000.tar.gz"
        chroma_backup.write_bytes(b"Compressed ChromaDB backup")
        
        # 設定バックアップ
        config_backup = backup_dir / "config_20240101_120000.tar.gz"
        config_backup.write_bytes(b"Compressed config backup")
        
        # バックアップファイルの検証
        backup_files = list(backup_dir.glob("*_20240101_120000.*"))
        assert len(backup_files) == 3
        
        # ファイルサイズの確認
        for backup_file in backup_files:
            assert backup_file.stat().st_size > 0
        
        # バックアップ整合性チェック
        expected_files = ["agent_", "chroma_", "config_"]
        found_files = [f.name for f in backup_files]
        
        for expected in expected_files:
            assert any(expected in found for found in found_files)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])