"""
Result parser for reasoning responses
推論レスポンス結果解析システム
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import ast
import yaml

from .cot_engine import CoTResponse, ReasoningStep, CoTStep, ReasoningState

logger = logging.getLogger(__name__)


class ParseResultType(Enum):
    """解析結果タイプ"""
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    STRUCTURED = "structured"


@dataclass
class ParsedElement:
    """解析要素"""
    element_type: ParseResultType
    content: Any
    confidence: float
    position: Tuple[int, int]  # (start, end)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredResult:
    """構造化結果"""
    original_text: str
    parsed_elements: List[ParsedElement]
    main_content: str
    extracted_data: Dict[str, Any]
    structure_type: ParseResultType
    confidence: float
    parsing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultParser:
    """推論結果解析器"""
    
    def __init__(self):
        self.parsing_patterns = self._initialize_parsing_patterns()
        self.extraction_rules = self._initialize_extraction_rules()
        
        # 解析統計
        self.parsing_stats = {
            "total_parses": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "type_distribution": {},
            "average_confidence": 0.0,
            "parsing_times": []
        }
        
        logger.info("Result parser initialized")
    
    def _initialize_parsing_patterns(self) -> Dict[ParseResultType, List[str]]:
        """解析パターン初期化"""
        return {
            ParseResultType.JSON: [
                r'```json\s*(.*?)\s*```',
                r'\{.*?\}',
                r'\[.*?\]'
            ],
            ParseResultType.YAML: [
                r'```yaml\s*(.*?)\s*```',
                r'```yml\s*(.*?)\s*```',
                r'^[a-zA-Z_][a-zA-Z0-9_]*:\s*.*$'
            ],
            ParseResultType.CODE: [
                r'```python\s*(.*?)\s*```',
                r'```javascript\s*(.*?)\s*```',
                r'```java\s*(.*?)\s*```',
                r'```cpp\s*(.*?)\s*```',
                r'```sql\s*(.*?)\s*```',
                r'```bash\s*(.*?)\s*```'
            ],
            ParseResultType.LIST: [
                r'^\s*[-*+]\s+.*$',
                r'^\s*\d+\.\s+.*$',
                r'^\s*[a-zA-Z]\.\s+.*$'
            ],
            ParseResultType.TABLE: [
                r'\|.*\|',
                r'^\s*[-\s|]+\s*$'
            ],
            ParseResultType.STRUCTURED: [
                r'##\s+.*$',
                r'###\s+.*$',
                r'**.*?:**',
                r'__.*?__:'
            ]
        }
    
    def _initialize_extraction_rules(self) -> Dict[str, Dict[str, Any]]:
        """抽出ルール初期化"""
        return {
            "numbers": {
                "pattern": r'\b\d+(?:\.\d+)?\b',
                "type": "float",
                "description": "数値の抽出"
            },
            "dates": {
                "pattern": r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
                "type": "date",
                "description": "日付の抽出"
            },
            "urls": {
                "pattern": r'https?://[^\s]+',
                "type": "url",
                "description": "URLの抽出"
            },
            "emails": {
                "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "type": "email",
                "description": "メールアドレスの抽出"
            },
            "keywords": {
                "pattern": r'\b(?:重要|注意|警告|推奨|必須|オプション|結論|結果|方法|手順|ステップ)\b',
                "type": "keyword",
                "description": "キーワードの抽出"
            },
            "questions": {
                "pattern": r'[^。！？]*[？?]',
                "type": "question",
                "description": "質問の抽出"
            },
            "answers": {
                "pattern": r'(?:答え|回答|結果|結論)[:：]\s*(.+)',
                "type": "answer",
                "description": "回答の抽出"
            }
        }
    
    def parse_response(self, response: CoTResponse) -> StructuredResult:
        """推論レスポンスの解析"""
        import time
        start_time = time.time()
        
        try:
            original_text = response.response_text
            
            # 各要素の解析
            parsed_elements = []
            
            # JSON解析
            json_elements = self._parse_json(original_text)
            parsed_elements.extend(json_elements)
            
            # YAML解析
            yaml_elements = self._parse_yaml(original_text)
            parsed_elements.extend(yaml_elements)
            
            # コード解析
            code_elements = self._parse_code(original_text)
            parsed_elements.extend(code_elements)
            
            # リスト解析
            list_elements = self._parse_lists(original_text)
            parsed_elements.extend(list_elements)
            
            # テーブル解析
            table_elements = self._parse_tables(original_text)
            parsed_elements.extend(table_elements)
            
            # 構造化テキスト解析
            structured_elements = self._parse_structured_text(original_text)
            parsed_elements.extend(structured_elements)
            
            # データ抽出
            extracted_data = self._extract_data(original_text)
            
            # メインコンテンツの特定
            main_content = self._extract_main_content(original_text, parsed_elements)
            
            # 構造タイプの決定
            structure_type = self._determine_structure_type(parsed_elements)
            
            # 信頼度計算
            confidence = self._calculate_confidence(parsed_elements, extracted_data)
            
            parsing_time = time.time() - start_time
            
            result = StructuredResult(
                original_text=original_text,
                parsed_elements=parsed_elements,
                main_content=main_content,
                extracted_data=extracted_data,
                structure_type=structure_type,
                confidence=confidence,
                parsing_time=parsing_time,
                metadata={
                    "response_id": response.request_id,
                    "step_count": response.step_count,
                    "processing_time": response.processing_time,
                    "model_used": response.model_used
                }
            )
            
            # 統計更新
            self._update_parsing_stats(result)
            
            logger.info(f"Response parsed successfully: {structure_type.value} in {parsing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return self._create_error_result(response.response_text, str(e), time.time() - start_time)
    
    def _parse_json(self, text: str) -> List[ParsedElement]:
        """JSON解析"""
        elements = []
        patterns = self.parsing_patterns[ParseResultType.JSON]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                try:
                    json_text = match.group(1) if match.groups() else match.group(0)
                    
                    # JSON解析試行
                    parsed_json = json.loads(json_text)
                    
                    element = ParsedElement(
                        element_type=ParseResultType.JSON,
                        content=parsed_json,
                        confidence=0.9,
                        position=(match.start(), match.end()),
                        metadata={
                            "json_type": type(parsed_json).__name__,
                            "json_size": len(str(parsed_json))
                        }
                    )
                    
                    elements.append(element)
                    
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return elements
    
    def _parse_yaml(self, text: str) -> List[ParsedElement]:
        """YAML解析"""
        elements = []
        patterns = self.parsing_patterns[ParseResultType.YAML]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                try:
                    yaml_text = match.group(1) if match.groups() else match.group(0)
                    
                    # YAML解析試行
                    parsed_yaml = yaml.safe_load(yaml_text)
                    
                    element = ParsedElement(
                        element_type=ParseResultType.YAML,
                        content=parsed_yaml,
                        confidence=0.8,
                        position=(match.start(), match.end()),
                        metadata={
                            "yaml_type": type(parsed_yaml).__name__,
                            "yaml_size": len(str(parsed_yaml))
                        }
                    )
                    
                    elements.append(element)
                    
                except (yaml.YAMLError, ValueError):
                    continue
        
        return elements
    
    def _parse_code(self, text: str) -> List[ParsedElement]:
        """コード解析"""
        elements = []
        patterns = self.parsing_patterns[ParseResultType.CODE]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                try:
                    code_text = match.group(1) if match.groups() else match.group(0)
                    
                    # 言語の特定
                    language = self._detect_language(code_text, pattern)
                    
                    # コードの構文チェック
                    syntax_valid = self._validate_syntax(code_text, language)
                    
                    element = ParsedElement(
                        element_type=ParseResultType.CODE,
                        content=code_text,
                        confidence=0.8 if syntax_valid else 0.5,
                        position=(match.start(), match.end()),
                        metadata={
                            "language": language,
                            "syntax_valid": syntax_valid,
                            "code_length": len(code_text),
                            "line_count": len(code_text.split('\n'))
                        }
                    )
                    
                    elements.append(element)
                    
                except Exception:
                    continue
        
        return elements
    
    def _parse_lists(self, text: str) -> List[ParsedElement]:
        """リスト解析"""
        elements = []
        patterns = self.parsing_patterns[ParseResultType.LIST]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            
            list_items = []
            for match in matches:
                list_items.append(match.group(0).strip())
            
            if list_items:
                element = ParsedElement(
                    element_type=ParseResultType.LIST,
                    content=list_items,
                    confidence=0.7,
                    position=(matches[0].start() if matches else (0, 0), 
                             matches[-1].end() if matches else (0, 0)),
                    metadata={
                        "item_count": len(list_items),
                        "list_type": "ordered" if re.match(r'^\s*\d+\.', list_items[0]) else "unordered"
                    }
                )
                
                elements.append(element)
        
        return elements
    
    def _parse_tables(self, text: str) -> List[ParsedElement]:
        """テーブル解析"""
        elements = []
        
        # テーブル行の検出
        lines = text.split('\n')
        table_lines = []
        current_table = []
        
        for i, line in enumerate(lines):
            if '|' in line:
                current_table.append(line)
            else:
                if len(current_table) >= 2:  # ヘッダー + データ行
                    table_lines.append((i - len(current_table), i - 1, current_table))
                current_table = []
        
        # 最後のテーブル処理
        if len(current_table) >= 2:
            table_lines.append((len(lines) - len(current_table), len(lines) - 1, current_table))
        
        for start_line, end_line, table_data in table_lines:
            try:
                parsed_table = self._parse_table_data(table_data)
                
                element = ParsedElement(
                    element_type=ParseResultType.TABLE,
                    content=parsed_table,
                    confidence=0.8,
                    position=(start_line, end_line),
                    metadata={
                        "row_count": len(parsed_table.get("rows", [])),
                        "column_count": len(parsed_table.get("headers", [])),
                        "table_type": "markdown"
                    }
                )
                
                elements.append(element)
                
            except Exception:
                continue
        
        return elements
    
    def _parse_structured_text(self, text: str) -> List[ParsedElement]:
        """構造化テキスト解析"""
        elements = []
        patterns = self.parsing_patterns[ParseResultType.STRUCTURED]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            
            structured_items = []
            for match in matches:
                structured_items.append({
                    "text": match.group(0).strip(),
                    "position": (match.start(), match.end())
                })
            
            if structured_items:
                element = ParsedElement(
                    element_type=ParseResultType.STRUCTURED,
                    content=structured_items,
                    confidence=0.6,
                    position=(structured_items[0]["position"][0], 
                             structured_items[-1]["position"][1]),
                    metadata={
                        "item_count": len(structured_items),
                        "structure_type": "headers" if "##" in pattern else "key_value"
                    }
                )
                
                elements.append(element)
        
        return elements
    
    def _extract_data(self, text: str) -> Dict[str, Any]:
        """データ抽出"""
        extracted = {}
        
        for rule_name, rule in self.extraction_rules.items():
            pattern = rule["pattern"]
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            if matches:
                extracted[rule_name] = {
                    "values": matches,
                    "count": len(matches),
                    "type": rule["type"],
                    "description": rule["description"]
                }
        
        return extracted
    
    def _extract_main_content(self, text: str, elements: List[ParsedElement]) -> str:
        """メインコンテンツの抽出"""
        if not elements:
            return text
        
        # 要素の位置を取得
        element_positions = [(elem.position[0], elem.position[1]) for elem in elements]
        element_positions.sort()
        
        # 要素間のテキストを抽出
        main_parts = []
        last_end = 0
        
        for start, end in element_positions:
            if start > last_end:
                part = text[last_end:start].strip()
                if part:
                    main_parts.append(part)
            last_end = end
        
        # 最後の要素以降のテキスト
        if last_end < len(text):
            part = text[last_end:].strip()
            if part:
                main_parts.append(part)
        
        return '\n\n'.join(main_parts)
    
    def _determine_structure_type(self, elements: List[ParsedElement]) -> ParseResultType:
        """構造タイプの決定"""
        if not elements:
            return ParseResultType.TEXT
        
        # 要素タイプの集計
        type_counts = {}
        for element in elements:
            type_counts[element.element_type] = type_counts.get(element.element_type, 0) + 1
        
        # 最も多いタイプを返す
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_confidence(self, elements: List[ParsedElement], extracted_data: Dict[str, Any]) -> float:
        """信頼度計算"""
        if not elements and not extracted_data:
            return 0.3  # プレーンテキスト
        
        total_confidence = 0.0
        total_weight = 0.0
        
        # 要素の信頼度
        for element in elements:
            weight = self._get_element_weight(element.element_type)
            total_confidence += element.confidence * weight
            total_weight += weight
        
        # 抽出データの信頼度
        if extracted_data:
            data_confidence = min(1.0, len(extracted_data) * 0.2)
            total_confidence += data_confidence * 0.5
            total_weight += 0.5
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _get_element_weight(self, element_type: ParseResultType) -> float:
        """要素タイプの重み"""
        weights = {
            ParseResultType.JSON: 1.0,
            ParseResultType.YAML: 0.9,
            ParseResultType.CODE: 0.8,
            ParseResultType.TABLE: 0.7,
            ParseResultType.LIST: 0.6,
            ParseResultType.STRUCTURED: 0.5,
            ParseResultType.TEXT: 0.3
        }
        return weights.get(element_type, 0.5)
    
    def _detect_language(self, code: str, pattern: str) -> str:
        """言語検出"""
        if 'python' in pattern:
            return 'python'
        elif 'javascript' in pattern:
            return 'javascript'
        elif 'java' in pattern:
            return 'java'
        elif 'cpp' in pattern:
            return 'cpp'
        elif 'sql' in pattern:
            return 'sql'
        elif 'bash' in pattern:
            return 'bash'
        else:
            # ヒューリスティック検出
            if 'def ' in code or 'import ' in code:
                return 'python'
            elif 'function ' in code or 'const ' in code:
                return 'javascript'
            elif 'SELECT ' in code or 'FROM ' in code:
                return 'sql'
            else:
                return 'unknown'
    
    def _validate_syntax(self, code: str, language: str) -> bool:
        """構文検証"""
        try:
            if language == 'python':
                ast.parse(code)
                return True
            elif language == 'javascript':
                # 簡単な構文チェック
                return code.count('{') == code.count('}') and code.count('(') == code.count(')')
            else:
                return True  # その他の言語は基本的な検証のみ
        except:
            return False
    
    def _parse_table_data(self, table_lines: List[str]) -> Dict[str, Any]:
        """テーブルデータ解析"""
        if len(table_lines) < 2:
            return {"headers": [], "rows": []}
        
        # ヘッダー行
        header_line = table_lines[0]
        headers = [col.strip() for col in header_line.split('|') if col.strip()]
        
        # 区切り行をスキップ
        data_lines = table_lines[2:] if len(table_lines) > 2 else []
        
        # データ行
        rows = []
        for line in data_lines:
            if '|' in line:
                row_data = [col.strip() for col in line.split('|') if col.strip()]
                if len(row_data) == len(headers):
                    rows.append(row_data)
        
        return {
            "headers": headers,
            "rows": rows,
            "column_count": len(headers),
            "row_count": len(rows)
        }
    
    def _update_parsing_stats(self, result: StructuredResult):
        """解析統計更新"""
        self.parsing_stats["total_parses"] += 1
        
        if result.confidence > 0.5:
            self.parsing_stats["successful_parses"] += 1
        else:
            self.parsing_stats["failed_parses"] += 1
        
        # タイプ分布更新
        structure_type = result.structure_type.value
        self.parsing_stats["type_distribution"][structure_type] = \
            self.parsing_stats["type_distribution"].get(structure_type, 0) + 1
        
        # 平均信頼度更新
        total_parses = self.parsing_stats["total_parses"]
        current_avg = self.parsing_stats["average_confidence"]
        self.parsing_stats["average_confidence"] = \
            (current_avg * (total_parses - 1) + result.confidence) / total_parses
        
        # 解析時間記録
        self.parsing_stats["parsing_times"].append(result.parsing_time)
    
    def _create_error_result(self, text: str, error_message: str, parsing_time: float) -> StructuredResult:
        """エラー結果の作成"""
        return StructuredResult(
            original_text=text,
            parsed_elements=[],
            main_content=text,
            extracted_data={},
            structure_type=ParseResultType.TEXT,
            confidence=0.0,
            parsing_time=parsing_time,
            metadata={"error": error_message}
        )
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """解析統計取得"""
        import statistics
        
        return {
            "total_parses": self.parsing_stats["total_parses"],
            "successful_parses": self.parsing_stats["successful_parses"],
            "failed_parses": self.parsing_stats["failed_parses"],
            "success_rate": (
                self.parsing_stats["successful_parses"] / self.parsing_stats["total_parses"]
                if self.parsing_stats["total_parses"] > 0 else 0.0
            ),
            "type_distribution": self.parsing_stats["type_distribution"],
            "average_confidence": self.parsing_stats["average_confidence"],
            "average_parsing_time": (
                statistics.mean(self.parsing_stats["parsing_times"])
                if self.parsing_stats["parsing_times"] else 0.0
            )
        }
    
    def reset_statistics(self):
        """統計リセット"""
        self.parsing_stats = {
            "total_parses": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "type_distribution": {},
            "average_confidence": 0.0,
            "parsing_times": []
        }


# 便利関数
def parse_reasoning_response(response: CoTResponse) -> StructuredResult:
    """推論レスポンス解析（便利関数）"""
    parser = ResultParser()
    return parser.parse_response(response)


def extract_key_information(result: StructuredResult) -> Dict[str, Any]:
    """重要情報の抽出"""
    key_info = {
        "main_content": result.main_content,
        "structure_type": result.structure_type.value,
        "confidence": result.confidence,
        "element_count": len(result.parsed_elements)
    }
    
    # 抽出データから重要情報を選択
    important_data = {}
    for key, data in result.extracted_data.items():
        if data["count"] > 0:
            important_data[key] = {
                "count": data["count"],
                "sample": data["values"][:3]  # 最初の3つをサンプルとして
            }
    
    key_info["extracted_data"] = important_data
    
    return key_info


def format_structured_result(result: StructuredResult) -> str:
    """構造化結果のフォーマット"""
    output = []
    
    output.append(f"=== Structured Result ===")
    output.append(f"Structure Type: {result.structure_type.value}")
    output.append(f"Confidence: {result.confidence:.3f}")
    output.append(f"Parsing Time: {result.parsing_time:.3f}s")
    output.append(f"Elements: {len(result.parsed_elements)}")
    
    if result.parsed_elements:
        output.append(f"\n=== Parsed Elements ===")
        for i, element in enumerate(result.parsed_elements):
            output.append(f"{i+1}. {element.element_type.value} (confidence: {element.confidence:.3f})")
            if element.metadata:
                output.append(f"   Metadata: {element.metadata}")
    
    if result.extracted_data:
        output.append(f"\n=== Extracted Data ===")
        for key, data in result.extracted_data.items():
            output.append(f"{key}: {data['count']} items")
            if data["values"]:
                output.append(f"  Sample: {data['values'][:2]}")
    
    output.append(f"\n=== Main Content ===")
    output.append(result.main_content[:500] + "..." if len(result.main_content) > 500 else result.main_content)
    
    return "\n".join(output)


# 使用例
if __name__ == "__main__":
    # テスト用のCoTResponse作成
    from .cot_engine import CoTResponse, ReasoningStep, CoTStep, ReasoningState
    
    test_steps = [
        ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
        ReasoningStep(2, CoTStep.ACTION, "計算を実行します"),
        ReasoningStep(3, CoTStep.OBSERVATION, "結果を確認します"),
        ReasoningStep(4, CoTStep.CONCLUSION, "最終回答を導きます")
    ]
    
    test_response = CoTResponse(
        request_id="test_123",
        response_text="""
        この問題を段階的に解決します。

        ## 分析結果
        - 数値: 42, 3.14, 100
        - 日付: 2024-01-15
        - URL: https://example.com

        ```json
        {
            "result": 42,
            "confidence": 0.95,
            "method": "calculation"
        }
        ```

        ```python
        def calculate(x, y):
            return x + y
        
        result = calculate(20, 22)
        print(result)
        ```

        | 項目 | 値 | 説明 |
        |------|-----|------|
        | 結果 | 42 | 計算結果 |
        | 信頼度 | 0.95 | 高い信頼度 |

        結論: 答えは42です。
        """,
        processing_time=5.0,
        reasoning_steps=test_steps,
        final_confidence=0.8,
        step_count=4,
        total_thinking_time=4.5,
        quality_score=0.7,
        model_used="qwen2:7b-instruct",
        state=ReasoningState.COMPLETED
    )
    
    # 結果解析実行
    parser = ResultParser()
    result = parser.parse_response(test_response)
    
    print(format_structured_result(result))
    
    # 重要情報抽出
    key_info = extract_key_information(result)
    print(f"\n=== Key Information ===")
    print(json.dumps(key_info, indent=2, ensure_ascii=False))
    
    # 統計表示
    stats = parser.get_parsing_statistics()
    print(f"\n=== Parsing Statistics ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
