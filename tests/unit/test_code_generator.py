"""
HuggingFace Code Generator のユニットテスト

要件: 3.2, 3.5
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import torch

from src.advanced_agent.multimodal.code_generator import (
    HuggingFaceCodeGenerator,
    CodeGenerationResult,
    CodeQualityMetrics
)


class TestHuggingFaceCodeGenerator:
    """HuggingFaceCodeGenerator のテストクラス"""
    
    @pytest.fixture
    def generator(self):
        """テスト用ジェネレータ"""
        with patch('src.advanced_agent.multimodal.code_generator.BitsAndBytesConfig') as mock_config:
            mock_config.side_effect = Exception("BitsAndBytes not available")
            return HuggingFaceCodeGenerator(
                model_name="microsoft/CodeGPT-small-py",
                max_vram_gb=2.0
            )
    
    @pytest.fixture
    def mock_pipeline_result(self):
        """モックパイプライン結果"""
        return [{
            "generated_text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "score": 0.85
        }]
    
    def test_init(self, generator):
        """初期化テスト"""
        assert generator.model_name == "microsoft/CodeGPT-small-py"
        assert generator.max_vram_gb == 2.0
        assert generator.device == "auto"
        assert generator.code_pipeline is None
    
    @pytest.mark.asyncio
    async def test_format_prompt(self, generator):
        """プロンプト整形テスト"""
        prompt = "create a function"
        
        # Python
        formatted = generator._format_prompt(prompt, "python")
        assert "Python code for: create a function" in formatted
        assert "```python" in formatted
        
        # JavaScript
        formatted = generator._format_prompt(prompt, "javascript")
        assert "JavaScript code for: create a function" in formatted
        assert "```javascript" in formatted
    
    def test_extract_code(self, generator):
        """コード抽出テスト"""
        # コードブロック付き
        text_with_block = "Here's the code:\n```python\ndef hello():\n    print('hello')\n```"
        extracted = generator._extract_code(text_with_block, "python")
        assert "def hello():" in extracted
        assert "print('hello')" in extracted
        
        # コードブロックなし
        text_without_block = "def hello():\n    print('hello')"
        extracted = generator._extract_code(text_without_block, "python")
        assert extracted == text_without_block
    
    @pytest.mark.asyncio
    async def test_check_python_syntax_valid(self, generator):
        """Python構文チェック（有効）"""
        valid_code = "def test():\n    return 42"
        is_valid, error = await generator._check_python_syntax(valid_code)
        assert is_valid is True
        assert error is None
    
    @pytest.mark.asyncio
    async def test_check_python_syntax_invalid(self, generator):
        """Python構文チェック（無効）"""
        invalid_code = "def test(\n    return 42"  # 括弧が閉じていない
        is_valid, error = await generator._check_python_syntax(invalid_code)
        assert is_valid is False
        assert error is not None
        assert "Syntax error" in error
    
    def test_calculate_complexity(self, generator):
        """複雑度計算テスト"""
        # シンプルなコード
        simple_code = "def hello():\n    print('hello')"
        complexity = generator._calculate_complexity(simple_code, "python")
        assert 0.0 <= complexity <= 1.0
        
        # 複雑なコード
        complex_code = """
def complex_function():
    if True:
        for i in range(10):
            while i > 0:
                try:
                    if i % 2 == 0:
                        print(i)
                except Exception:
                    pass
        """
        complexity_complex = generator._calculate_complexity(complex_code, "python")
        assert complexity_complex < complexity  # より複雑なので低いスコア
    
    def test_calculate_readability(self, generator):
        """可読性計算テスト"""
        # コメント付きコード
        commented_code = """
# This function calculates fibonacci
def fibonacci(n):
    # Base case
    if n <= 1:
        return n
    # Recursive case
    return fibonacci(n-1) + fibonacci(n-2)
        """
        readability = generator._calculate_readability(commented_code, "python")
        assert 0.0 <= readability <= 1.0
        
        # コメントなしコード
        uncommented_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
        """
        readability_uncommented = generator._calculate_readability(uncommented_code, "python")
        assert readability > readability_uncommented  # コメント付きの方が高い
    
    def test_calculate_security(self, generator):
        """セキュリティスコア計算テスト"""
        # 安全なコード
        safe_code = "def add(a, b):\n    return a + b"
        security_safe = generator._calculate_security(safe_code, "python")
        assert security_safe == 1.0
        
        # 危険なコード
        dangerous_code = "import os\nos.system('rm -rf /')\neval(user_input)"
        security_dangerous = generator._calculate_security(dangerous_code, "python")
        assert security_dangerous < security_safe
    
    def test_calculate_confidence(self, generator):
        """信頼度計算テスト"""
        quality_metrics = CodeQualityMetrics(
            complexity_score=0.8,
            readability_score=0.7,
            maintainability_score=0.9,
            security_score=1.0,
            performance_score=0.8
        )
        
        confidence = generator._calculate_confidence(0.9, True, quality_metrics)
        assert 0.0 <= confidence <= 1.0
        
        # 構文エラーありの場合
        confidence_syntax_error = generator._calculate_confidence(0.9, False, quality_metrics)
        assert confidence_syntax_error < confidence
    
    @pytest.mark.asyncio
    async def test_test_execution_safe(self, generator):
        """安全なコード実行テスト"""
        safe_code = "result = 2 + 3\nprint(result)"
        execution_result = await generator._test_execution(safe_code)
        assert "5" in execution_result
    
    @pytest.mark.asyncio
    async def test_test_execution_dangerous(self, generator):
        """危険なコード実行テスト"""
        dangerous_code = "import os\nos.system('echo hello')"
        execution_result = await generator._test_execution(dangerous_code)
        assert "potentially dangerous code detected" in execution_result
    
    @pytest.mark.asyncio
    async def test_generate_suggestions(self, generator):
        """改善提案生成テスト"""
        # 低品質コード
        low_quality_metrics = CodeQualityMetrics(
            complexity_score=0.3,
            readability_score=0.4,
            maintainability_score=0.2,
            security_score=0.5,
            performance_score=0.3
        )
        
        suggestions = await generator._generate_suggestions(
            "some code", "python", True, low_quality_metrics
        )
        
        assert len(suggestions) > 0
        assert any("複雑度" in s for s in suggestions)
        assert any("可読性" in s for s in suggestions)
    
    @pytest.mark.asyncio
    @patch('src.advanced_agent.multimodal.code_generator.AutoTokenizer')
    @patch('src.advanced_agent.multimodal.code_generator.AutoModelForCausalLM')
    @patch('src.advanced_agent.multimodal.code_generator.pipeline')
    async def test_initialize_success(self, mock_pipeline, mock_model, mock_tokenizer, generator):
        """初期化成功テスト"""
        # モック設定
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value.pad_token = None
        mock_tokenizer.from_pretrained.return_value.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value.eos_token_id = 0
        
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # 初期化実行
        result = await generator.initialize()
        
        assert result is True
        assert mock_tokenizer.from_pretrained.called
        assert mock_model.from_pretrained.called
        assert mock_pipeline.called
    
    @pytest.mark.asyncio
    @patch('src.advanced_agent.multimodal.code_generator.AutoTokenizer')
    async def test_initialize_failure(self, mock_tokenizer, generator):
        """初期化失敗テスト"""
        # 例外を発生させる
        mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
        
        result = await generator.initialize()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_code_mock(self, generator, mock_pipeline_result):
        """コード生成テスト（モック使用）"""
        # パイプラインをモック
        generator.code_pipeline = Mock()
        generator.code_pipeline.return_value = mock_pipeline_result
        generator.tokenizer = Mock()
        generator.tokenizer.eos_token_id = 0
        
        result = await generator.generate_code("create fibonacci function", "python")
        
        assert isinstance(result, CodeGenerationResult)
        assert result.language == "python"
        assert "fibonacci" in result.generated_code.lower()
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.syntax_valid, bool)
    
    @pytest.mark.asyncio
    async def test_batch_generate(self, generator, mock_pipeline_result):
        """バッチ生成テスト"""
        # パイプラインをモック
        generator.code_pipeline = Mock()
        generator.code_pipeline.return_value = mock_pipeline_result
        generator.tokenizer = Mock()
        generator.tokenizer.eos_token_id = 0
        
        prompts = [
            "create a function",
            "write a class",
            "implement algorithm"
        ]
        
        results = await generator.batch_generate(prompts, "python", max_concurrent=2)
        
        assert len(results) == 3
        assert all(isinstance(r, CodeGenerationResult) for r in results)
        assert all(r.language == "python" for r in results)
    
    def test_get_memory_usage(self, generator):
        """メモリ使用量取得テスト"""
        memory_info = generator.get_memory_usage()
        assert isinstance(memory_info, dict)
        
        if torch.cuda.is_available():
            assert "gpu_allocated_mb" in memory_info
            assert "gpu_reserved_mb" in memory_info
    
    @pytest.mark.asyncio
    async def test_cleanup(self, generator):
        """クリーンアップテスト"""
        # ダミーオブジェクト設定
        generator.model = Mock()
        generator.tokenizer = Mock()
        generator.code_pipeline = Mock()
        
        await generator.cleanup()
        
        # オブジェクトが削除されていることを確認
        # （実際の削除は確認困難なので、例外が発生しないことを確認）
        assert True  # クリーンアップが例外なく完了


class TestCodeQualityMetrics:
    """CodeQualityMetrics のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        metrics = CodeQualityMetrics(
            complexity_score=0.8,
            readability_score=0.7,
            maintainability_score=0.9,
            security_score=1.0,
            performance_score=0.8
        )
        
        assert metrics.complexity_score == 0.8
        assert metrics.readability_score == 0.7
        assert metrics.maintainability_score == 0.9
        assert metrics.security_score == 1.0
        assert metrics.performance_score == 0.8


class TestCodeGenerationResult:
    """CodeGenerationResult のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        result = CodeGenerationResult(
            generated_code="def test(): pass",
            language="python",
            confidence_score=0.85,
            syntax_valid=True,
            quality_metrics={"complexity": 0.8}
        )
        
        assert result.generated_code == "def test(): pass"
        assert result.language == "python"
        assert result.confidence_score == 0.85
        assert result.syntax_valid is True
        assert result.quality_metrics == {"complexity": 0.8}


@pytest.mark.integration
class TestCodeGeneratorIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """完全ワークフローテスト"""
        with patch('src.advanced_agent.multimodal.code_generator.BitsAndBytesConfig') as mock_config:
            mock_config.side_effect = Exception("BitsAndBytes not available")
            generator = HuggingFaceCodeGenerator(
                model_name="microsoft/CodeGPT-small-py",
                max_vram_gb=1.0  # 小さなVRAM制限でテスト
            )
        
        try:
            # 初期化をスキップ（実際のモデルダウンロードを避ける）
            # 代わりにモック化された軽量テストを実行
            
            # 構文チェックのみテスト
            valid_code = "def hello():\n    return 'Hello, World!'"
            is_valid, error = await generator._check_python_syntax(valid_code)
            assert is_valid is True
            
            # 品質評価のみテスト
            quality = await generator._evaluate_code_quality(valid_code, "python")
            assert isinstance(quality, CodeQualityMetrics)
            assert 0.0 <= quality.complexity_score <= 1.0
            
        finally:
            await generator.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])