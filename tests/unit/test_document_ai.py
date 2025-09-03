"""
HuggingFace Document AI のユニットテスト

要件: 3.3, 3.5
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import torch

from src.advanced_agent.multimodal.document_ai import (
    HuggingFaceDocumentAI,
    DocumentAnalysisResult,
    DocumentEntity,
    DocumentSection,
    MultimodalResult
)


class TestHuggingFaceDocumentAI:
    """HuggingFaceDocumentAI のテストクラス"""
    
    @pytest.fixture
    def doc_ai(self):
        """テスト用Document AI"""
        with patch('src.advanced_agent.multimodal.document_ai.BitsAndBytesConfig') as mock_config:
            mock_config.side_effect = Exception("BitsAndBytes not available")
            return HuggingFaceDocumentAI(
                ner_model="dbmdz/bert-large-cased-finetuned-conll03-english",
                classification_model="microsoft/DialoGPT-medium",
                max_vram_gb=2.0
            )
    
    @pytest.fixture
    def sample_text(self):
        """テスト用サンプルテキスト"""
        return """
        BUSINESS REPORT
        
        Executive Summary
        This report analyzes the quarterly performance of our company.
        
        Financial Results
        Revenue: $1,250,000
        Profit: $350,000
        Growth: 15%
        
        Key Personnel
        CEO: John Smith (john.smith@company.com)
        CFO: Jane Doe (jane.doe@company.com)
        
        Conclusion
        The company shows strong growth potential for the next quarter.
        """
    
    def test_init(self, doc_ai):
        """初期化テスト"""
        assert doc_ai.ner_model_name == "dbmdz/bert-large-cased-finetuned-conll03-english"
        assert doc_ai.classification_model_name == "microsoft/DialoGPT-medium"
        assert doc_ai.max_vram_gb == 2.0
        assert doc_ai.device == "auto"
        assert doc_ai.ner_pipeline is None
    
    @pytest.mark.asyncio
    async def test_detect_language(self, doc_ai):
        """言語検出テスト"""
        # 英語テキスト
        english_text = "This is an English document."
        language = await doc_ai._detect_language(english_text)
        assert language == "en"
        
        # 日本語テキスト
        japanese_text = "これは日本語の文書です。" * 5  # 十分な文字数
        language = await doc_ai._detect_language(japanese_text)
        assert language == "ja"
    
    @pytest.mark.asyncio
    async def test_classify_document_type(self, doc_ai):
        """ドキュメントタイプ分類テスト"""
        # 契約書
        contract_text = "This agreement between parties..."
        doc_type = await doc_ai._classify_document_type(contract_text)
        assert doc_type == "contract"
        
        # 請求書
        invoice_text = "Invoice #12345, Amount due: $500"
        doc_type = await doc_ai._classify_document_type(invoice_text)
        assert doc_type == "invoice"
        
        # レポート
        report_text = "This report analyzes the findings..."
        doc_type = await doc_ai._classify_document_type(report_text)
        assert doc_type == "report"
        
        # 一般文書
        general_text = "Some random text without specific keywords"
        doc_type = await doc_ai._classify_document_type(general_text)
        assert doc_type == "general"
    
    @pytest.mark.asyncio
    async def test_segment_document(self, doc_ai, sample_text):
        """ドキュメントセグメンテーションテスト"""
        sections = await doc_ai._segment_document(sample_text)
        
        assert len(sections) > 0
        assert all(isinstance(section, DocumentSection) for section in sections)
        assert all(section.confidence > 0 for section in sections)
        
        # セクションタイトルの確認
        section_titles = [section.title for section in sections]
        assert any("BUSINESS REPORT" in title for title in section_titles)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_txt(self, doc_ai, sample_text):
        """テキストファイル抽出テスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text)
            temp_file = Path(f.name)
        
        try:
            extracted = await doc_ai._extract_text_from_txt(temp_file)
            assert extracted == sample_text
        finally:
            temp_file.unlink()
    
    @pytest.mark.asyncio
    async def test_extract_key_information(self, doc_ai, sample_text):
        """キー情報抽出テスト"""
        entities = [
            DocumentEntity("John Smith", "PERSON", 0.9, 0, 10),
            DocumentEntity("$1,250,000", "MONEY", 0.8, 20, 30)
        ]
        
        key_info = await doc_ai._extract_key_information(sample_text, entities)
        
        assert isinstance(key_info, dict)
        assert "document_length" in key_info
        assert "word_count" in key_info
        assert "entity_count" in key_info
        assert key_info["entity_count"] == 2
        
        # 数値の抽出確認
        assert len(key_info["numbers"]) > 0
        
        # メールアドレスの抽出確認
        assert len(key_info["emails"]) > 0
        assert "john.smith@company.com" in key_info["emails"]
    
    def test_calculate_total_confidence(self, doc_ai):
        """総合信頼度計算テスト"""
        sections = [
            DocumentSection("Title", "Content", "section", 0.8),
            DocumentSection("Title2", "Content2", "section", 0.9)
        ]
        
        entities = [
            DocumentEntity("Entity1", "TYPE1", 0.7, 0, 7),
            DocumentEntity("Entity2", "TYPE2", 0.9, 10, 17)
        ]
        
        confidence = doc_ai._calculate_total_confidence(sections, entities)
        assert 0.0 <= confidence <= 1.0
        
        # 空のリストでのテスト
        empty_confidence = doc_ai._calculate_total_confidence([], [])
        assert empty_confidence == 0.0
    
    def test_merge_duplicate_entities(self, doc_ai):
        """重複エンティティマージテスト"""
        entities = [
            DocumentEntity("John Smith", "PERSON", 0.8, 0, 10),
            DocumentEntity("john smith", "PERSON", 0.9, 20, 30),  # 重複（大文字小文字）
            DocumentEntity("Jane Doe", "PERSON", 0.7, 40, 48)
        ]
        
        merged = doc_ai._merge_duplicate_entities(entities)
        
        # 重複が除去されていることを確認
        assert len(merged) == 2
        
        # 信頼度でソートされていることを確認
        assert merged[0].confidence >= merged[1].confidence
    
    def test_analyze_cross_modal_relationships(self, doc_ai):
        """クロスモーダル関係性分析テスト"""
        text_analysis = DocumentAnalysisResult(
            document_type="report",
            language="en",
            total_confidence=0.8,
            sections=[],
            entities=[
                DocumentEntity("John Smith", "PERSON", 0.9, 0, 10),
                DocumentEntity("Company", "ORG", 0.8, 20, 27)
            ],
            summary="Test summary",
            key_information={},
            processing_time=1.0
        )
        
        image_analysis = {
            "entities": [
                {"text": "john smith", "label": "PERSON", "confidence": 0.7},
                {"text": "Logo", "label": "ORG", "confidence": 0.6}
            ]
        }
        
        relationships = doc_ai._analyze_cross_modal_relationships(text_analysis, image_analysis)
        
        assert isinstance(relationships, dict)
        assert "text_image_overlap" in relationships
        assert "complementary_info" in relationships
        assert "contradictions" in relationships
        
        # 重複する情報があることを確認
        assert len(relationships["text_image_overlap"]) > 0
    
    def test_calculate_combined_confidence(self, doc_ai):
        """統合信頼度計算テスト"""
        text_analysis = DocumentAnalysisResult(
            document_type="report",
            language="en",
            total_confidence=0.8,
            sections=[],
            entities=[],
            summary="",
            key_information={},
            processing_time=1.0
        )
        
        image_analysis = {"confidence": 0.6}
        
        combined = doc_ai._calculate_combined_confidence(text_analysis, image_analysis)
        assert 0.0 <= combined <= 1.0
        assert combined == 0.7  # (0.8 + 0.6) / 2
        
        # 片方だけの場合
        text_only = doc_ai._calculate_combined_confidence(text_analysis, None)
        assert text_only == 0.8
        
        # 両方Noneの場合
        none_confidence = doc_ai._calculate_combined_confidence(None, None)
        assert none_confidence == 0.0
    
    def test_generate_integrated_summary(self, doc_ai):
        """統合要約生成テスト"""
        text_analysis = DocumentAnalysisResult(
            document_type="report",
            language="en",
            total_confidence=0.8,
            sections=[],
            entities=[],
            summary="This is a test summary",
            key_information={},
            processing_time=1.0
        )
        
        image_analysis = {
            "extracted_text": "Image contains important information about the report"
        }
        
        entities = [
            DocumentEntity("John Smith", "PERSON", 0.9, 0, 10),
            DocumentEntity("Company", "ORG", 0.8, 20, 27)
        ]
        
        summary = doc_ai._generate_integrated_summary(text_analysis, image_analysis, entities)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Text Analysis" in summary
        assert "Image Content" in summary
        assert "Key Entities" in summary
    
    @pytest.mark.asyncio
    @patch('src.advanced_agent.multimodal.document_ai.pipeline')
    async def test_initialize_success(self, mock_pipeline, doc_ai):
        """初期化成功テスト"""
        # モック設定
        mock_pipeline.return_value = Mock()
        
        result = await doc_ai.initialize()
        
        assert result is True
        assert mock_pipeline.call_count >= 3  # NER, classification, summarization
    
    @pytest.mark.asyncio
    @patch('src.advanced_agent.multimodal.document_ai.pipeline')
    async def test_initialize_failure(self, mock_pipeline, doc_ai):
        """初期化失敗テスト"""
        # 例外を発生させる
        mock_pipeline.side_effect = Exception("Model loading failed")
        
        result = await doc_ai.initialize()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_analyze_document_mock(self, doc_ai, sample_text):
        """ドキュメント解析テスト（モック使用）"""
        # パイプラインをモック
        doc_ai.ner_pipeline = Mock()
        doc_ai.ner_pipeline.return_value = [
            {"word": "John Smith", "entity_group": "PERSON", "score": 0.9, "start": 0, "end": 10}
        ]
        
        doc_ai.summarization_pipeline = Mock()
        doc_ai.summarization_pipeline.return_value = [{"summary_text": "Test summary"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text)
            temp_file = Path(f.name)
        
        try:
            result = await doc_ai.analyze_document(temp_file)
            
            assert isinstance(result, DocumentAnalysisResult)
            assert result.document_type in ["report", "general", "unknown"]
            assert result.language == "en"
            assert 0.0 <= result.total_confidence <= 1.0
            assert len(result.sections) > 0
            assert result.processing_time > 0
            
        finally:
            temp_file.unlink()
    
    @pytest.mark.asyncio
    async def test_batch_analyze(self, doc_ai, sample_text):
        """バッチ解析テスト"""
        # パイプラインをモック
        doc_ai.ner_pipeline = Mock()
        doc_ai.ner_pipeline.return_value = []
        doc_ai.summarization_pipeline = Mock()
        doc_ai.summarization_pipeline.return_value = [{"summary_text": "Test"}]
        
        # テストファイル作成
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"{sample_text} - File {i}")
                temp_files.append(Path(f.name))
        
        try:
            results = await doc_ai.batch_analyze(temp_files, max_concurrent=2)
            
            assert len(results) == 3
            assert all(isinstance(r, DocumentAnalysisResult) for r in results)
            
        finally:
            for temp_file in temp_files:
                temp_file.unlink()
    
    def test_get_memory_usage(self, doc_ai):
        """メモリ使用量取得テスト"""
        memory_info = doc_ai.get_memory_usage()
        assert isinstance(memory_info, dict)
        
        if torch.cuda.is_available():
            assert "gpu_allocated_mb" in memory_info
            assert "gpu_reserved_mb" in memory_info
    
    @pytest.mark.asyncio
    async def test_cleanup(self, doc_ai):
        """クリーンアップテスト"""
        # ダミーオブジェクト設定
        doc_ai.ner_pipeline = Mock()
        doc_ai.classification_pipeline = Mock()
        doc_ai.summarization_pipeline = Mock()
        
        await doc_ai.cleanup()
        
        # クリーンアップが例外なく完了することを確認
        assert True


class TestDocumentEntity:
    """DocumentEntity のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        entity = DocumentEntity(
            text="John Smith",
            label="PERSON",
            confidence=0.9,
            start_pos=0,
            end_pos=10,
            context="John Smith is the CEO"
        )
        
        assert entity.text == "John Smith"
        assert entity.label == "PERSON"
        assert entity.confidence == 0.9
        assert entity.start_pos == 0
        assert entity.end_pos == 10
        assert entity.context == "John Smith is the CEO"


class TestDocumentSection:
    """DocumentSection のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        section = DocumentSection(
            title="Executive Summary",
            content="This is the summary content",
            section_type="summary",
            confidence=0.8
        )
        
        assert section.title == "Executive Summary"
        assert section.content == "This is the summary content"
        assert section.section_type == "summary"
        assert section.confidence == 0.8
        assert section.entities == []
        assert section.metadata == {}


class TestDocumentAnalysisResult:
    """DocumentAnalysisResult のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        result = DocumentAnalysisResult(
            document_type="report",
            language="en",
            total_confidence=0.85,
            sections=[],
            entities=[],
            summary="Test summary",
            key_information={"test": "value"},
            processing_time=1.5
        )
        
        assert result.document_type == "report"
        assert result.language == "en"
        assert result.total_confidence == 0.85
        assert result.sections == []
        assert result.entities == []
        assert result.summary == "Test summary"
        assert result.key_information == {"test": "value"}
        assert result.processing_time == 1.5
        assert result.error_message is None


class TestMultimodalResult:
    """MultimodalResult のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        result = MultimodalResult(
            text_analysis=None,
            image_analysis=None,
            combined_confidence=0.7,
            integrated_entities=[],
            cross_modal_relationships={},
            final_summary="Integrated summary"
        )
        
        assert result.text_analysis is None
        assert result.image_analysis is None
        assert result.combined_confidence == 0.7
        assert result.integrated_entities == []
        assert result.cross_modal_relationships == {}
        assert result.final_summary == "Integrated summary"


@pytest.mark.integration
class TestDocumentAIIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """完全ワークフローテスト"""
        with patch('src.advanced_agent.multimodal.document_ai.BitsAndBytesConfig') as mock_config:
            mock_config.side_effect = Exception("BitsAndBytes not available")
            doc_ai = HuggingFaceDocumentAI(max_vram_gb=1.0)
        
        try:
            # 初期化をスキップ（実際のモデルダウンロードを避ける）
            # 代わりに軽量テストを実行
            
            # 言語検出のみテスト
            language = await doc_ai._detect_language("This is English text")
            assert language == "en"
            
            # ドキュメントタイプ分類のみテスト
            doc_type = await doc_ai._classify_document_type("This is a report about our findings")
            assert doc_type == "report"
            
            # キー情報抽出のみテスト
            key_info = await doc_ai._extract_key_information("Contact: john@example.com", [])
            assert isinstance(key_info, dict)
            assert len(key_info["emails"]) > 0
            
        finally:
            await doc_ai.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])