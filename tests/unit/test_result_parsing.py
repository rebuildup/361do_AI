"""
Tests for result parsing system
推論結果解析システムのテスト
"""

import pytest
import json
import time
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from src.advanced_agent.reasoning.result_parser import (
    ResultParser, ParsedElement, ParseResultType, StructuredResult,
    parse_reasoning_response, extract_key_information, format_structured_result
)
from src.advanced_agent.reasoning.cot_engine import (
    CoTResponse, ReasoningStep, CoTStep, ReasoningState
)


class TestResultParser:
    """結果解析器のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.parser = ResultParser()
        
        # テスト用のCoTResponse作成
        self.test_steps = [
            ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
            ReasoningStep(2, CoTStep.ACTION, "計算を実行します"),
            ReasoningStep(3, CoTStep.OBSERVATION, "結果を確認します"),
            ReasoningStep(4, CoTStep.CONCLUSION, "最終回答を導きます")
        ]
        
        self.test_response = CoTResponse(
            request_id="test_123",
            response_text="テストレスポンス",
            processing_time=5.0,
            reasoning_steps=self.test_steps,
            final_confidence=0.8,
            step_count=4,
            total_thinking_time=4.5,
            quality_score=0.7,
            model_used="qwen2:7b-instruct",
            state=ReasoningState.COMPLETED
        )
    
    def test_parser_initialization(self):
        """解析器の初期化テスト"""
        assert self.parser is not None
        assert hasattr(self.parser, 'parsing_patterns')
        assert hasattr(self.parser, 'extraction_rules')
        assert hasattr(self.parser, 'parsing_stats')
        
        # パターンの確認
        assert ParseResultType.JSON in self.parser.parsing_patterns
        assert ParseResultType.CODE in self.parser.parsing_patterns
        assert ParseResultType.LIST in self.parser.parsing_patterns
    
    def test_parse_simple_text(self):
        """シンプルテキストの解析テスト"""
        simple_text = "これはシンプルなテキストです。"
        
        response = CoTResponse(
            request_id="test_simple",
            response_text=simple_text,
            processing_time=1.0,
            reasoning_steps=[],
            final_confidence=0.5,
            step_count=0,
            total_thinking_time=0.0,
            quality_score=0.5,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        result = self.parser.parse_response(response)
        
        assert result is not None
        assert result.original_text == simple_text
        assert result.structure_type == ParseResultType.TEXT
        assert result.main_content == simple_text
        assert len(result.parsed_elements) == 0
        assert result.confidence > 0.0
    
    def test_parse_json_content(self):
        """JSONコンテンツの解析テスト"""
        json_text = """
        分析結果は以下の通りです：
        
        ```json
        {
            "result": 42,
            "confidence": 0.95,
            "method": "calculation"
        }
        ```
        
        この結果に基づいて結論を導きます。
        """
        
        response = CoTResponse(
            request_id="test_json",
            response_text=json_text,
            processing_time=2.0,
            reasoning_steps=self.test_steps,
            final_confidence=0.8,
            step_count=4,
            total_thinking_time=1.5,
            quality_score=0.8,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        result = self.parser.parse_response(response)
        
        assert result is not None
        assert result.structure_type == ParseResultType.JSON
        assert len(result.parsed_elements) > 0
        
        # JSON要素の確認
        json_elements = [elem for elem in result.parsed_elements 
                        if elem.element_type == ParseResultType.JSON]
        assert len(json_elements) > 0
        
        json_element = json_elements[0]
        assert isinstance(json_element.content, dict)
        assert json_element.content["result"] == 42
        assert json_element.confidence > 0.8
    
    def test_parse_code_content(self):
        """コードコンテンツの解析テスト"""
        code_text = """
        計算を実行するためのPythonコード：
        
        ```python
        def calculate(x, y):
            return x + y
        
        result = calculate(20, 22)
        print(result)
        ```
        
        このコードの結果は42になります。
        """
        
        response = CoTResponse(
            request_id="test_code",
            response_text=code_text,
            processing_time=3.0,
            reasoning_steps=self.test_steps,
            final_confidence=0.9,
            step_count=4,
            total_thinking_time=2.0,
            quality_score=0.9,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        result = self.parser.parse_response(response)
        
        assert result is not None
        assert len(result.parsed_elements) > 0
        
        # コード要素の確認
        code_elements = [elem for elem in result.parsed_elements 
                        if elem.element_type == ParseResultType.CODE]
        assert len(code_elements) > 0
        
        code_element = code_elements[0]
        assert "def calculate" in code_element.content
        assert code_element.metadata["language"] == "python"
        assert code_element.confidence > 0.7
    
    def test_parse_list_content(self):
        """リストコンテンツの解析テスト"""
        list_text = """
        解決手順は以下の通りです：
        
        1. 問題を理解する
        2. 解決方法を検討する
        3. 計算を実行する
        4. 結果を検証する
        5. 結論を導く
        
        これらの手順に従って進めます。
        """
        
        response = CoTResponse(
            request_id="test_list",
            response_text=list_text,
            processing_time=2.5,
            reasoning_steps=self.test_steps,
            final_confidence=0.8,
            step_count=4,
            total_thinking_time=1.8,
            quality_score=0.8,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        result = self.parser.parse_response(response)
        
        assert result is not None
        assert len(result.parsed_elements) > 0
        
        # リスト要素の確認
        list_elements = [elem for elem in result.parsed_elements 
                        if elem.element_type == ParseResultType.LIST]
        assert len(list_elements) > 0
        
        list_element = list_elements[0]
        assert isinstance(list_element.content, list)
        assert len(list_element.content) == 5
        assert "問題を理解する" in list_element.content[0]
    
    def test_parse_table_content(self):
        """テーブルコンテンツの解析テスト"""
        table_text = """
        結果の詳細：
        
        | 項目 | 値 | 説明 |
        |------|-----|------|
        | 結果 | 42 | 計算結果 |
        | 信頼度 | 0.95 | 高い信頼度 |
        | 方法 | 加算 | 使用した計算方法 |
        
        この表に基づいて分析します。
        """
        
        response = CoTResponse(
            request_id="test_table",
            response_text=table_text,
            processing_time=2.0,
            reasoning_steps=self.test_steps,
            final_confidence=0.8,
            step_count=4,
            total_thinking_time=1.5,
            quality_score=0.8,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        result = self.parser.parse_response(response)
        
        assert result is not None
        assert len(result.parsed_elements) > 0
        
        # テーブル要素の確認
        table_elements = [elem for elem in result.parsed_elements 
                         if elem.element_type == ParseResultType.TABLE]
        assert len(table_elements) > 0
        
        table_element = table_elements[0]
        assert isinstance(table_element.content, dict)
        assert "headers" in table_element.content
        assert "rows" in table_element.content
        assert len(table_element.content["headers"]) == 3
        assert len(table_element.content["rows"]) == 3
    
    def test_extract_data(self):
        """データ抽出テスト"""
        text_with_data = """
        分析結果：
        - 数値: 42, 3.14, 100
        - 日付: 2024-01-15
        - URL: https://example.com
        - メール: test@example.com
        - 重要: この結果は信頼できます
        """
        
        response = CoTResponse(
            request_id="test_extract",
            response_text=text_with_data,
            processing_time=1.5,
            reasoning_steps=self.test_steps,
            final_confidence=0.8,
            step_count=4,
            total_thinking_time=1.0,
            quality_score=0.8,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        result = self.parser.parse_response(response)
        
        assert result is not None
        assert len(result.extracted_data) > 0
        
        # 抽出データの確認
        assert "numbers" in result.extracted_data
        assert "dates" in result.extracted_data
        assert "urls" in result.extracted_data
        assert "emails" in result.extracted_data
        assert "keywords" in result.extracted_data
        
        # 数値の確認
        numbers = result.extracted_data["numbers"]["values"]
        assert "42" in numbers
        assert "3.14" in numbers
        assert "100" in numbers
    
    def test_confidence_calculation(self):
        """信頼度計算テスト"""
        # 高品質なレスポンス
        high_quality_text = """
        ```json
        {"result": 42, "confidence": 0.95}
        ```
        
        | 項目 | 値 |
        |------|-----|
        | 結果 | 42 |
        
        1. 計算を実行
        2. 結果を確認
        3. 結論を導く
        """
        
        high_quality_response = CoTResponse(
            request_id="test_high_quality",
            response_text=high_quality_text,
            processing_time=3.0,
            reasoning_steps=self.test_steps,
            final_confidence=0.9,
            step_count=4,
            total_thinking_time=2.0,
            quality_score=0.9,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        high_quality_result = self.parser.parse_response(high_quality_response)
        
        # 低品質なレスポンス
        low_quality_text = "これは単純なテキストです。"
        
        low_quality_response = CoTResponse(
            request_id="test_low_quality",
            response_text=low_quality_text,
            processing_time=1.0,
            reasoning_steps=[],
            final_confidence=0.3,
            step_count=0,
            total_thinking_time=0.0,
            quality_score=0.3,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        low_quality_result = self.parser.parse_response(low_quality_response)
        
        # 信頼度の比較
        assert high_quality_result.confidence > low_quality_result.confidence
        assert high_quality_result.confidence > 0.7
        assert low_quality_result.confidence < 0.5
    
    def test_structure_type_detection(self):
        """構造タイプ検出テスト"""
        # JSON構造
        json_response = CoTResponse(
            request_id="test_json_structure",
            response_text='{"result": 42}',
            processing_time=1.0,
            reasoning_steps=[],
            final_confidence=0.8,
            step_count=0,
            total_thinking_time=0.0,
            quality_score=0.8,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        json_result = self.parser.parse_response(json_response)
        assert json_result.structure_type == ParseResultType.JSON
        
        # リスト構造
        list_response = CoTResponse(
            request_id="test_list_structure",
            response_text="1. 項目1\n2. 項目2\n3. 項目3",
            processing_time=1.0,
            reasoning_steps=[],
            final_confidence=0.8,
            step_count=0,
            total_thinking_time=0.0,
            quality_score=0.8,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        list_result = self.parser.parse_response(list_response)
        assert list_result.structure_type == ParseResultType.LIST
    
    def test_parsing_statistics(self):
        """解析統計テスト"""
        # 初期統計
        initial_stats = self.parser.get_parsing_statistics()
        assert initial_stats["total_parses"] == 0
        
        # 複数回の解析実行
        for i in range(5):
            response = CoTResponse(
                request_id=f"test_stats_{i}",
                response_text=f"テストテキスト {i}",
                processing_time=1.0,
                reasoning_steps=[],
                final_confidence=0.8,
                step_count=0,
                total_thinking_time=0.0,
                quality_score=0.8,
                model_used="test",
                state=ReasoningState.COMPLETED
            )
            self.parser.parse_response(response)
        
        # 統計の確認
        final_stats = self.parser.get_parsing_statistics()
        assert final_stats["total_parses"] == 5
        assert final_stats["successful_parses"] >= 0
        assert final_stats["failed_parses"] >= 0
        assert final_stats["average_confidence"] > 0.0
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 無効なレスポンス
        invalid_response = None
        
        with pytest.raises(AttributeError):
            self.parser.parse_response(invalid_response)
        
        # 空のテキスト
        empty_response = CoTResponse(
            request_id="test_empty",
            response_text="",
            processing_time=0.0,
            reasoning_steps=[],
            final_confidence=0.0,
            step_count=0,
            total_thinking_time=0.0,
            quality_score=0.0,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        result = self.parser.parse_response(empty_response)
        assert result is not None
        assert result.confidence >= 0.0
    
    def test_reset_statistics(self):
        """統計リセットテスト"""
        # 統計を蓄積
        response = CoTResponse(
            request_id="test_reset",
            response_text="テスト",
            processing_time=1.0,
            reasoning_steps=[],
            final_confidence=0.8,
            step_count=0,
            total_thinking_time=0.0,
            quality_score=0.8,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        self.parser.parse_response(response)
        
        # 統計の確認
        stats_before = self.parser.get_parsing_statistics()
        assert stats_before["total_parses"] > 0
        
        # リセット
        self.parser.reset_statistics()
        
        # 統計の確認
        stats_after = self.parser.get_parsing_statistics()
        assert stats_after["total_parses"] == 0
        assert stats_after["successful_parses"] == 0
        assert stats_after["failed_parses"] == 0


class TestConvenienceFunctions:
    """便利関数のテスト"""
    
    def test_parse_reasoning_response(self):
        """推論レスポンス解析便利関数テスト"""
        test_steps = [
            ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
            ReasoningStep(2, CoTStep.ACTION, "計算を実行します")
        ]
        
        response = CoTResponse(
            request_id="test_convenience",
            response_text="テストレスポンス",
            processing_time=2.0,
            reasoning_steps=test_steps,
            final_confidence=0.8,
            step_count=2,
            total_thinking_time=1.5,
            quality_score=0.8,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        result = parse_reasoning_response(response)
        
        assert result is not None
        assert isinstance(result, StructuredResult)
        assert result.original_text == "テストレスポンス"
    
    def test_extract_key_information(self):
        """重要情報抽出テスト"""
        # テスト用のStructuredResult作成
        test_elements = [
            ParsedElement(
                element_type=ParseResultType.JSON,
                content={"result": 42},
                confidence=0.9,
                position=(0, 20),
                metadata={"json_type": "dict"}
            )
        ]
        
        test_result = StructuredResult(
            original_text="テストテキスト",
            parsed_elements=test_elements,
            main_content="メインコンテンツ",
            extracted_data={"numbers": {"values": ["42"], "count": 1, "type": "float"}},
            structure_type=ParseResultType.JSON,
            confidence=0.8,
            parsing_time=1.0,
            metadata={}
        )
        
        key_info = extract_key_information(test_result)
        
        assert key_info is not None
        assert "main_content" in key_info
        assert "structure_type" in key_info
        assert "confidence" in key_info
        assert "element_count" in key_info
        assert "extracted_data" in key_info
        
        assert key_info["structure_type"] == "json"
        assert key_info["confidence"] == 0.8
        assert key_info["element_count"] == 1
    
    def test_format_structured_result(self):
        """構造化結果フォーマットテスト"""
        test_elements = [
            ParsedElement(
                element_type=ParseResultType.JSON,
                content={"result": 42},
                confidence=0.9,
                position=(0, 20),
                metadata={"json_type": "dict"}
            )
        ]
        
        test_result = StructuredResult(
            original_text="テストテキスト",
            parsed_elements=test_elements,
            main_content="メインコンテンツ",
            extracted_data={"numbers": {"values": ["42"], "count": 1, "type": "float"}},
            structure_type=ParseResultType.JSON,
            confidence=0.8,
            parsing_time=1.0,
            metadata={}
        )
        
        formatted = format_structured_result(test_result)
        
        assert formatted is not None
        assert isinstance(formatted, str)
        assert "Structured Result" in formatted
        assert "json" in formatted
        assert "0.800" in formatted
        assert "1" in formatted  # element count


class TestIntegration:
    """統合テスト"""
    
    def test_complete_parsing_workflow(self):
        """完全な解析ワークフローテスト"""
        # 複雑なレスポンスの作成
        complex_text = """
        ## 分析結果
        
        この問題を段階的に解決します。
        
        ### ステップ1: データ準備
        ```json
        {
            "input": [1, 2, 3, 4, 5],
            "operation": "sum"
        }
        ```
        
        ### ステップ2: 計算実行
        ```python
        def calculate_sum(numbers):
            return sum(numbers)
        
        result = calculate_sum([1, 2, 3, 4, 5])
        print(f"結果: {result}")
        ```
        
        ### ステップ3: 結果検証
        | 項目 | 値 | 説明 |
        |------|-----|------|
        | 入力 | [1,2,3,4,5] | 元のデータ |
        | 計算 | sum() | 使用した関数 |
        | 結果 | 15 | 計算結果 |
        
        ### 結論
        1. データを準備しました
        2. 計算を実行しました
        3. 結果を検証しました
        4. 答えは15です
        
        数値: 15, 1, 2, 3, 4, 5
        日付: 2024-01-15
        URL: https://example.com
        """
        
        test_steps = [
            ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
            ReasoningStep(2, CoTStep.ACTION, "データを準備します"),
            ReasoningStep(3, CoTStep.ACTION, "計算を実行します"),
            ReasoningStep(4, CoTStep.OBSERVATION, "結果を確認します"),
            ReasoningStep(5, CoTStep.CONCLUSION, "結論を導きます")
        ]
        
        response = CoTResponse(
            request_id="test_complete",
            response_text=complex_text,
            processing_time=5.0,
            reasoning_steps=test_steps,
            final_confidence=0.9,
            step_count=5,
            total_thinking_time=4.0,
            quality_score=0.9,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        # 解析実行
        parser = ResultParser()
        result = parser.parse_response(response)
        
        # 結果の検証
        assert result is not None
        assert result.confidence > 0.7
        assert len(result.parsed_elements) > 0
        assert len(result.extracted_data) > 0
        
        # 要素タイプの確認
        element_types = [elem.element_type for elem in result.parsed_elements]
        assert ParseResultType.JSON in element_types
        assert ParseResultType.CODE in element_types
        assert ParseResultType.TABLE in element_types
        assert ParseResultType.LIST in element_types
        assert ParseResultType.STRUCTURED in element_types
        
        # 抽出データの確認
        assert "numbers" in result.extracted_data
        assert "dates" in result.extracted_data
        assert "urls" in result.extracted_data
        
        # 重要情報の抽出
        key_info = extract_key_information(result)
        assert key_info["element_count"] > 0
        assert key_info["confidence"] > 0.7
        
        # フォーマット
        formatted = format_structured_result(result)
        assert "Structured Result" in formatted
        assert "Parsed Elements" in formatted
        assert "Extracted Data" in formatted


if __name__ == "__main__":
    pytest.main([__file__])
