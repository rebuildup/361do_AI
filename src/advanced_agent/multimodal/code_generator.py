"""
HuggingFace Code Generation Pipeline Integration

RTX 4050 6GB VRAM環境でのコード生成機能を提供します。
HuggingFace Transformersの既存パイプラインを活用し、
構文チェックとコード品質評価を統合します。

要件: 3.2, 3.5
"""

import ast
import asyncio
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from accelerate import Accelerator
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationResult:
    """コード生成結果"""
    generated_code: str
    language: str
    confidence_score: float
    syntax_valid: bool
    quality_metrics: Dict[str, float]
    execution_result: Optional[str] = None
    error_message: Optional[str] = None
    suggestions: List[str] = None


@dataclass
class CodeQualityMetrics:
    """コード品質指標"""
    complexity_score: float
    readability_score: float
    maintainability_score: float
    security_score: float
    performance_score: float


class HuggingFaceCodeGenerator:
    """HuggingFace Transformers による統合コード生成システム"""
    
    def __init__(self,
                 model_name: str = "microsoft/CodeGPT-small-py",
                 max_vram_gb: float = 4.0,
                 device: str = "auto"):
        """
        初期化
        
        Args:
            model_name: 使用するコード生成モデル名
            max_vram_gb: 最大VRAM使用量（GB）
            device: 使用デバイス
        """
        self.model_name = model_name
        self.max_vram_gb = max_vram_gb
        self.device = device
        
        # HuggingFace Accelerate初期化
        self.accelerator = Accelerator()
        
        # 量子化設定（VRAM節約）
        try:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        except Exception as e:
            logger.warning(f"BitsAndBytes not available, using default config: {e}")
            self.quantization_config = None
        
        # パイプライン初期化
        self.code_pipeline = None
        self.tokenizer = None
        self.model = None
        
        # 言語別構文チェッカー
        self.syntax_checkers = {
            "python": self._check_python_syntax,
            "javascript": self._check_javascript_syntax,
            "java": self._check_java_syntax,
            "cpp": self._check_cpp_syntax
        }
        
    async def initialize(self) -> bool:
        """パイプライン初期化"""
        try:
            logger.info(f"Initializing code generation pipeline with model: {self.model_name}")
            
            # GPU メモリチェック
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < self.max_vram_gb:
                    logger.warning(f"GPU memory ({gpu_memory:.1f}GB) is less than required ({self.max_vram_gb}GB)")
            
            # トークナイザー読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # パディングトークン設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデル読み込み（量子化適用）
            model_kwargs = {
                "device_map": self.device,
                "torch_dtype": torch.float16,
                "trust_remote_code": True
            }
            
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # HuggingFace Pipeline作成
            self.code_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device,
                torch_dtype=torch.float16,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Accelerate最適化
            self.model = self.accelerator.prepare(self.model)
            
            logger.info("Code generation pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize code generation pipeline: {e}")
            return False
    
    async def generate_code(self,
                          prompt: str,
                          language: str = "python",
                          max_length: int = 512,
                          temperature: float = 0.1) -> CodeGenerationResult:
        """
        コード生成実行
        
        Args:
            prompt: コード生成プロンプト
            language: 対象言語
            max_length: 最大生成長
            temperature: 生成温度
            
        Returns:
            CodeGenerationResult: 生成結果
        """
        try:
            if not self.code_pipeline:
                await self.initialize()
            
            # 言語別プロンプト調整
            formatted_prompt = self._format_prompt(prompt, language)
            
            logger.info(f"Generating {language} code for prompt: {prompt[:100]}...")
            
            # HuggingFace Pipeline でコード生成
            generation_result = self.code_pipeline(
                formatted_prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # 生成されたコード抽出
            generated_text = generation_result[0]["generated_text"]
            generated_code = self._extract_code(generated_text, language)
            
            # 構文チェック
            syntax_valid, syntax_error = await self._check_syntax(generated_code, language)
            
            # 品質評価
            quality_metrics = await self._evaluate_code_quality(generated_code, language)
            
            # 信頼度スコア計算
            confidence_score = self._calculate_confidence(
                generation_result[0].get("score", 0.5),
                syntax_valid,
                quality_metrics
            )
            
            # 実行テスト（Pythonのみ）
            execution_result = None
            if language == "python" and syntax_valid:
                execution_result = await self._test_execution(generated_code)
            
            # 改善提案生成
            suggestions = await self._generate_suggestions(
                generated_code, language, syntax_valid, quality_metrics
            )
            
            result = CodeGenerationResult(
                generated_code=generated_code,
                language=language,
                confidence_score=confidence_score,
                syntax_valid=syntax_valid,
                quality_metrics=quality_metrics.__dict__,
                execution_result=execution_result,
                error_message=syntax_error if not syntax_valid else None,
                suggestions=suggestions
            )
            
            logger.info(f"Code generation completed. Confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return CodeGenerationResult(
                generated_code="",
                language=language,
                confidence_score=0.0,
                syntax_valid=False,
                quality_metrics={},
                error_message=str(e)
            )
    
    def _format_prompt(self, prompt: str, language: str) -> str:
        """言語別プロンプト整形"""
        language_templates = {
            "python": f"# Python code for: {prompt}\n```python\n",
            "javascript": f"// JavaScript code for: {prompt}\n```javascript\n",
            "java": f"// Java code for: {prompt}\n```java\n",
            "cpp": f"// C++ code for: {prompt}\n```cpp\n"
        }
        
        return language_templates.get(language, f"# Code for: {prompt}\n```\n")
    
    def _extract_code(self, generated_text: str, language: str) -> str:
        """生成テキストからコード部分を抽出"""
        # コードブロック抽出
        if "```" in generated_text:
            parts = generated_text.split("```")
            if len(parts) >= 2:
                code_part = parts[1]
                # 言語指定を除去
                lines = code_part.split("\n")
                if lines[0].strip().lower() in ["python", "javascript", "java", "cpp", "c++"]:
                    return "\n".join(lines[1:])
                return code_part
        
        # コードブロックがない場合は全体を返す
        return generated_text.strip()
    
    async def _check_syntax(self, code: str, language: str) -> Tuple[bool, Optional[str]]:
        """構文チェック実行"""
        checker = self.syntax_checkers.get(language)
        if checker:
            return await checker(code)
        return True, None  # 未対応言語は通す
    
    async def _check_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Python構文チェック"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Parse error: {e}"
    
    async def _check_javascript_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """JavaScript構文チェック（Node.js使用）"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Node.js で構文チェック
            result = subprocess.run(
                ["node", "--check", temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            Path(temp_file).unlink()  # 一時ファイル削除
            
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Syntax check timeout"
        except FileNotFoundError:
            logger.warning("Node.js not found, skipping JavaScript syntax check")
            return True, None  # Node.jsがない場合はスキップ
        except Exception as e:
            return False, f"Syntax check error: {e}"
    
    async def _check_java_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Java構文チェック（javac使用）"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # javac でコンパイルチェック
            result = subprocess.run(
                ["javac", temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            Path(temp_file).unlink()  # 一時ファイル削除
            
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except FileNotFoundError:
            logger.warning("javac not found, skipping Java syntax check")
            return True, None
        except Exception as e:
            return False, f"Compilation error: {e}"
    
    async def _check_cpp_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """C++構文チェック（g++使用）"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # g++ でコンパイルチェック
            result = subprocess.run(
                ["g++", "-fsyntax-only", temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            Path(temp_file).unlink()  # 一時ファイル削除
            
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except FileNotFoundError:
            logger.warning("g++ not found, skipping C++ syntax check")
            return True, None
        except Exception as e:
            return False, f"Compilation error: {e}"    

    async def _evaluate_code_quality(self, code: str, language: str) -> CodeQualityMetrics:
        """コード品質評価"""
        try:
            # 基本的な品質指標計算
            complexity_score = self._calculate_complexity(code, language)
            readability_score = self._calculate_readability(code, language)
            maintainability_score = self._calculate_maintainability(code, language)
            security_score = self._calculate_security(code, language)
            performance_score = self._calculate_performance(code, language)
            
            return CodeQualityMetrics(
                complexity_score=complexity_score,
                readability_score=readability_score,
                maintainability_score=maintainability_score,
                security_score=security_score,
                performance_score=performance_score
            )
            
        except Exception as e:
            logger.warning(f"Code quality evaluation failed: {e}")
            return CodeQualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5)
    
    def _calculate_complexity(self, code: str, language: str) -> float:
        """複雑度計算（簡易版）"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 制御構造の数をカウント
        control_keywords = {
            "python": ["if", "elif", "else", "for", "while", "try", "except", "with"],
            "javascript": ["if", "else", "for", "while", "try", "catch", "switch"],
            "java": ["if", "else", "for", "while", "try", "catch", "switch"],
            "cpp": ["if", "else", "for", "while", "try", "catch", "switch"]
        }
        
        keywords = control_keywords.get(language, [])
        complexity_count = 0
        
        for line in non_empty_lines:
            for keyword in keywords:
                if f" {keyword} " in line or line.strip().startswith(keyword):
                    complexity_count += 1
        
        # 正規化（0-1スケール、低いほど良い）
        if len(non_empty_lines) == 0:
            return 1.0
        
        complexity_ratio = complexity_count / len(non_empty_lines)
        return max(0.0, min(1.0, 1.0 - complexity_ratio * 2))
    
    def _calculate_readability(self, code: str, language: str) -> float:
        """可読性計算"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        # 平均行長
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        
        # コメント率
        comment_chars = {"python": "#", "javascript": "//", "java": "//", "cpp": "//"}
        comment_char = comment_chars.get(language, "#")
        comment_lines = sum(1 for line in non_empty_lines if line.strip().startswith(comment_char))
        comment_ratio = comment_lines / len(non_empty_lines)
        
        # 可読性スコア（0-1、高いほど良い）
        length_score = max(0.0, min(1.0, 1.0 - (avg_line_length - 50) / 100))
        comment_score = min(1.0, comment_ratio * 3)  # コメント率を重視
        
        return (length_score + comment_score) / 2
    
    def _calculate_maintainability(self, code: str, language: str) -> float:
        """保守性計算"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        # 関数/メソッドの数
        function_keywords = {
            "python": ["def "],
            "javascript": ["function ", "=> "],
            "java": ["public ", "private ", "protected "],
            "cpp": ["void ", "int ", "double ", "float "]
        }
        
        keywords = function_keywords.get(language, [])
        function_count = 0
        
        for line in non_empty_lines:
            for keyword in keywords:
                if keyword in line:
                    function_count += 1
                    break
        
        # 関数あたりの行数（小さいほど良い）
        if function_count > 0:
            lines_per_function = len(non_empty_lines) / function_count
            maintainability = max(0.0, min(1.0, 1.0 - (lines_per_function - 10) / 50))
        else:
            maintainability = 0.5  # 関数がない場合は中程度
        
        return maintainability
    
    def _calculate_security(self, code: str, language: str) -> float:
        """セキュリティスコア計算"""
        # 危険なパターンを検出
        security_risks = {
            "python": ["eval(", "exec(", "os.system(", "subprocess.call(", "input("],
            "javascript": ["eval(", "innerHTML", "document.write(", "setTimeout("],
            "java": ["Runtime.exec(", "ProcessBuilder(", "System.exit("],
            "cpp": ["system(", "gets(", "strcpy(", "sprintf("]
        }
        
        risks = security_risks.get(language, [])
        risk_count = 0
        
        for risk in risks:
            if risk in code:
                risk_count += 1
        
        # セキュリティスコア（リスクが少ないほど高い）
        return max(0.0, 1.0 - risk_count * 0.2)
    
    def _calculate_performance(self, code: str, language: str) -> float:
        """パフォーマンススコア計算"""
        # パフォーマンスに影響する要素を検出
        performance_issues = {
            "python": ["for.*in.*range(len(", "while True:", "time.sleep("],
            "javascript": ["for.*in.*", "while(true)", "setTimeout("],
            "java": ["while(true)", "Thread.sleep(", "System.gc()"],
            "cpp": ["while(true)", "sleep(", "malloc("]
        }
        
        issues = performance_issues.get(language, [])
        issue_count = 0
        
        import re
        for issue in issues:
            if re.search(issue, code, re.IGNORECASE):
                issue_count += 1
        
        # パフォーマンススコア
        return max(0.0, 1.0 - issue_count * 0.15)
    
    def _calculate_confidence(self,
                            generation_score: float,
                            syntax_valid: bool,
                            quality_metrics: CodeQualityMetrics) -> float:
        """総合信頼度スコア計算"""
        # 各要素の重み
        weights = {
            "generation": 0.3,
            "syntax": 0.4,
            "quality": 0.3
        }
        
        # 構文有効性スコア
        syntax_score = 1.0 if syntax_valid else 0.0
        
        # 品質スコア平均
        quality_score = (
            quality_metrics.complexity_score +
            quality_metrics.readability_score +
            quality_metrics.maintainability_score +
            quality_metrics.security_score +
            quality_metrics.performance_score
        ) / 5
        
        # 重み付き平均
        confidence = (
            generation_score * weights["generation"] +
            syntax_score * weights["syntax"] +
            quality_score * weights["quality"]
        )
        
        return max(0.0, min(1.0, confidence))
    
    async def _test_execution(self, code: str) -> Optional[str]:
        """Pythonコード実行テスト（安全な環境で）"""
        try:
            # 危険なコードの実行を防ぐ
            dangerous_patterns = [
                "import os", "import sys", "import subprocess",
                "eval(", "exec(", "__import__", "open(",
                "file(", "input(", "raw_input("
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code:
                    return "Execution skipped: potentially dangerous code detected"
            
            # 一時的な名前空間で実行
            namespace = {"__builtins__": {}}
            
            # 基本的な組み込み関数のみ許可
            safe_builtins = {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round
            }
            namespace.update(safe_builtins)
            
            # 実行結果をキャプチャ
            import io
            import contextlib
            
            output = io.StringIO()
            
            with contextlib.redirect_stdout(output):
                exec(code, namespace)
            
            result = output.getvalue()
            return result if result else "Code executed successfully (no output)"
            
        except Exception as e:
            return f"Execution error: {e}"
    
    async def _generate_suggestions(self,
                                  code: str,
                                  language: str,
                                  syntax_valid: bool,
                                  quality_metrics: CodeQualityMetrics) -> List[str]:
        """改善提案生成"""
        suggestions = []
        
        # 構文エラーの提案
        if not syntax_valid:
            suggestions.append("構文エラーを修正してください")
        
        # 品質に基づく提案
        if quality_metrics.complexity_score < 0.6:
            suggestions.append("コードの複雑度を下げることを検討してください")
        
        if quality_metrics.readability_score < 0.6:
            suggestions.append("コメントを追加して可読性を向上させてください")
        
        if quality_metrics.maintainability_score < 0.6:
            suggestions.append("関数をより小さな単位に分割することを検討してください")
        
        if quality_metrics.security_score < 0.8:
            suggestions.append("セキュリティリスクのあるコードパターンを見直してください")
        
        if quality_metrics.performance_score < 0.6:
            suggestions.append("パフォーマンスを改善できる箇所があります")
        
        # 言語固有の提案
        if language == "python":
            if "for i in range(len(" in code:
                suggestions.append("enumerate()の使用を検討してください")
            if "while True:" in code and "break" not in code:
                suggestions.append("無限ループの終了条件を確認してください")
        
        return suggestions
    
    async def batch_generate(self,
                           prompts: List[str],
                           language: str = "python",
                           max_concurrent: int = 3) -> List[CodeGenerationResult]:
        """バッチコード生成"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt: str) -> CodeGenerationResult:
            async with semaphore:
                return await self.generate_code(prompt, language)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 例外を結果に変換
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(CodeGenerationResult(
                    generated_code="",
                    language=language,
                    confidence_score=0.0,
                    syntax_valid=False,
                    quality_metrics={},
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """メモリ使用量取得"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
            memory_info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return memory_info
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.code_pipeline:
                del self.code_pipeline
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Code generator cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# 使用例とテスト用のヘルパー関数
async def demo_code_generation():
    """デモ用コード生成実行"""
    generator = HuggingFaceCodeGenerator()
    
    try:
        # 初期化
        if not await generator.initialize():
            print("Failed to initialize code generator")
            return
        
        # テストプロンプト
        test_prompts = [
            "Create a function to calculate fibonacci numbers",
            "Write a class for managing a simple todo list",
            "Implement binary search algorithm"
        ]
        
        print("=== Code Generation Demo ===")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}: {prompt} ---")
            
            result = await generator.generate_code(prompt, "python")
            
            print(f"Generated Code:\n{result.generated_code}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Syntax Valid: {result.syntax_valid}")
            print(f"Quality Metrics: {result.quality_metrics}")
            
            if result.suggestions:
                print(f"Suggestions: {', '.join(result.suggestions)}")
            
            if result.execution_result:
                print(f"Execution Result: {result.execution_result}")
        
        # メモリ使用量表示
        memory_usage = generator.get_memory_usage()
        print(f"\nMemory Usage: {memory_usage}")
        
    finally:
        await generator.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_code_generation())