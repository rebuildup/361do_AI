"""
HuggingFace Document AI Pipeline Integration

RTX 4050 6GB VRAM環境でのドキュメント解析機能を提供します。
HuggingFace Transformersの既存パイプラインを活用し、
構造化情報抽出とマルチモーダル結果統合を実装します。

要件: 3.3, 3.5
"""

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
    BitsAndBytesConfig
)
from accelerate import Accelerator
import logging
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None
# オプショナル依存関係
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


@dataclass
class DocumentEntity:
    """抽出されたエンティティ"""
    text: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""


@dataclass
class DocumentSection:
    """ドキュメントセクション"""
    title: str
    content: str
    section_type: str
    confidence: float
    entities: List[DocumentEntity] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentAnalysisResult:
    """ドキュメント解析結果"""
    document_type: str
    language: str
    total_confidence: float
    sections: List[DocumentSection]
    entities: List[DocumentEntity]
    summary: str
    key_information: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalResult:
    """マルチモーダル統合結果"""
    text_analysis: Optional[DocumentAnalysisResult]
    image_analysis: Optional[Dict[str, Any]]
    combined_confidence: float
    integrated_entities: List[DocumentEntity]
    cross_modal_relationships: Dict[str, List[str]]
    final_summary: str


class HuggingFaceDocumentAI:
    """HuggingFace Transformers による統合ドキュメント AI システム"""
    
    def __init__(self,
                 ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
                 classification_model: str = "microsoft/DialoGPT-medium",
                 max_vram_gb: float = 4.0,
                 device: str = "auto"):
        """
        初期化
        
        Args:
            ner_model: 固有表現認識モデル名
            classification_model: 文書分類モデル名
            max_vram_gb: 最大VRAM使用量（GB）
            device: 使用デバイス
        """
        self.ner_model_name = ner_model
        self.classification_model_name = classification_model
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
        self.ner_pipeline = None
        self.classification_pipeline = None
        self.summarization_pipeline = None
        
        # サポートされるファイル形式
        self.supported_formats = {
            '.txt': self._extract_text_from_txt,
            '.pdf': self._extract_text_from_pdf,
            '.docx': self._extract_text_from_docx,
            '.doc': self._extract_text_from_docx,
            '.csv': self._extract_text_from_csv,
            '.xlsx': self._extract_text_from_excel,
            '.xls': self._extract_text_from_excel,
            '.png': self._extract_text_from_image,
            '.jpg': self._extract_text_from_image,
            '.jpeg': self._extract_text_from_image,
            '.bmp': self._extract_text_from_image,
            '.tiff': self._extract_text_from_image
        }
        
    async def initialize(self) -> bool:
        """パイプライン初期化"""
        try:
            logger.info("Initializing Document AI pipelines...")
            
            # GPU メモリチェック
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < self.max_vram_gb:
                    logger.warning(f"GPU memory ({gpu_memory:.1f}GB) is less than required ({self.max_vram_gb}GB)")
            
            # 固有表現認識パイプライン
            model_kwargs = {
                "device_map": self.device,
                "torch_dtype": torch.float16
            }
            
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                tokenizer=self.ner_model_name,
                aggregation_strategy="simple",
                device_map=self.device
            )
            
            # 文書分類パイプライン
            self.classification_pipeline = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device_map=self.device
            )
            
            # 要約パイプライン
            self.summarization_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device_map=self.device,
                max_length=150,
                min_length=30
            )
            
            logger.info("Document AI pipelines initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Document AI pipelines: {e}")
            return False
    
    async def analyze_document(self,
                             file_path: Union[str, Path],
                             extract_entities: bool = True,
                             classify_content: bool = True,
                             generate_summary: bool = True) -> DocumentAnalysisResult:
        """
        ドキュメント解析実行
        
        Args:
            file_path: ドキュメントファイルパス
            extract_entities: エンティティ抽出を実行するか
            classify_content: コンテンツ分類を実行するか
            generate_summary: 要約生成を実行するか
            
        Returns:
            DocumentAnalysisResult: 解析結果
        """
        import time
        start_time = time.time()
        
        try:
            if not self.ner_pipeline:
                await self.initialize()
            
            file_path = Path(file_path)
            logger.info(f"Analyzing document: {file_path}")
            
            # ファイル形式チェック
            if file_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # テキスト抽出
            extracted_text = await self._extract_text(file_path)
            if not extracted_text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # 言語検出
            language = await self._detect_language(extracted_text)
            
            # ドキュメントタイプ分類
            document_type = await self._classify_document_type(extracted_text)
            
            # セクション分割
            sections = await self._segment_document(extracted_text)
            
            # エンティティ抽出
            entities = []
            if extract_entities:
                entities = await self._extract_entities(extracted_text)
            
            # 要約生成
            summary = ""
            if generate_summary:
                summary = await self._generate_summary(extracted_text)
            
            # キー情報抽出
            key_information = await self._extract_key_information(extracted_text, entities)
            
            # 総合信頼度計算
            total_confidence = self._calculate_total_confidence(sections, entities)
            
            processing_time = time.time() - start_time
            
            result = DocumentAnalysisResult(
                document_type=document_type,
                language=language,
                total_confidence=total_confidence,
                sections=sections,
                entities=entities,
                summary=summary,
                key_information=key_information,
                processing_time=processing_time,
                metadata={
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "text_length": len(extracted_text)
                }
            )
            
            logger.info(f"Document analysis completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return DocumentAnalysisResult(
                document_type="unknown",
                language="unknown",
                total_confidence=0.0,
                sections=[],
                entities=[],
                summary="",
                key_information={},
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def analyze_multimodal(self,
                               text_file: Optional[Union[str, Path]] = None,
                               image_file: Optional[Union[str, Path]] = None,
                               text_content: Optional[str] = None) -> MultimodalResult:
        """
        マルチモーダル解析実行
        
        Args:
            text_file: テキストファイルパス
            image_file: 画像ファイルパス
            text_content: 直接テキストコンテンツ
            
        Returns:
            MultimodalResult: マルチモーダル解析結果
        """
        try:
            text_analysis = None
            image_analysis = None
            
            # テキスト解析
            if text_file:
                text_analysis = await self.analyze_document(text_file)
            elif text_content:
                # 一時ファイルに保存して解析
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(text_content)
                    temp_path = f.name
                
                text_analysis = await self.analyze_document(temp_path)
                Path(temp_path).unlink()  # 一時ファイル削除
            
            # 画像解析
            if image_file:
                image_analysis = await self._analyze_image(image_file)
            
            # 結果統合
            integrated_result = await self._integrate_multimodal_results(
                text_analysis, image_analysis
            )
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}")
            return MultimodalResult(
                text_analysis=text_analysis,
                image_analysis=image_analysis,
                combined_confidence=0.0,
                integrated_entities=[],
                cross_modal_relationships={},
                final_summary=f"Analysis failed: {e}"
            )
    
    async def _extract_text(self, file_path: Path) -> str:
        """ファイルからテキスト抽出"""
        extractor = self.supported_formats.get(file_path.suffix.lower())
        if extractor:
            return await extractor(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    async def _extract_text_from_txt(self, file_path: Path) -> str:
        """テキストファイルから抽出"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    async def _extract_text_from_pdf(self, file_path: Path) -> str:
        """PDFファイルから抽出"""
        if fitz is None:
            logger.warning("PyMuPDF not available, skipping PDF extraction")
            return ""
        
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    async def _extract_text_from_docx(self, file_path: Path) -> str:
        """Word文書から抽出"""
        if docx is None:
            logger.warning("python-docx not available, skipping DOCX extraction")
            return ""
        
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    async def _extract_text_from_csv(self, file_path: Path) -> str:
        """CSVファイルから抽出"""
        if pd is None:
            logger.warning("pandas not available, skipping CSV extraction")
            return ""
        
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"CSV extraction failed: {e}")
            return ""
    
    async def _extract_text_from_excel(self, file_path: Path) -> str:
        """Excelファイルから抽出"""
        if pd is None:
            logger.warning("pandas not available, skipping Excel extraction")
            return ""
        
        try:
            df = pd.read_excel(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            return ""
    
    async def _extract_text_from_image(self, file_path: Path) -> str:
        """画像ファイルからOCRでテキスト抽出"""
        if Image is None or pytesseract is None:
            logger.warning("PIL or pytesseract not available, skipping OCR extraction")
            return ""
        
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    async def _detect_language(self, text: str) -> str:
        """言語検出"""
        try:
            # 簡易言語検出（実際の実装では langdetect などを使用）
            if any(ord(char) > 127 for char in text[:1000]):
                # 非ASCII文字が含まれている場合の判定
                japanese_chars = sum(1 for char in text[:1000] if 0x3040 <= ord(char) <= 0x309F or 0x30A0 <= ord(char) <= 0x30FF)
                if japanese_chars > 10:
                    return "ja"
                chinese_chars = sum(1 for char in text[:1000] if 0x4E00 <= ord(char) <= 0x9FFF)
                if chinese_chars > 10:
                    return "zh"
            return "en"  # デフォルトは英語
        except Exception:
            return "unknown"
    
    async def _classify_document_type(self, text: str) -> str:
        """ドキュメントタイプ分類"""
        try:
            # キーワードベースの簡易分類
            text_lower = text.lower()
            
            if any(keyword in text_lower for keyword in ["contract", "agreement", "terms", "conditions"]):
                return "contract"
            elif any(keyword in text_lower for keyword in ["invoice", "bill", "payment", "amount due"]):
                return "invoice"
            elif any(keyword in text_lower for keyword in ["resume", "cv", "experience", "education"]):
                return "resume"
            elif any(keyword in text_lower for keyword in ["report", "analysis", "findings", "conclusion"]):
                return "report"
            elif any(keyword in text_lower for keyword in ["email", "subject:", "from:", "to:"]):
                return "email"
            elif any(keyword in text_lower for keyword in ["manual", "instructions", "guide", "how to"]):
                return "manual"
            else:
                return "general"
                
        except Exception:
            return "unknown"
    
    async def _segment_document(self, text: str) -> List[DocumentSection]:
        """ドキュメントセクション分割"""
        try:
            sections = []
            lines = text.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # セクションヘッダーの検出（簡易版）
                if (len(line) < 100 and 
                    (line.isupper() or 
                     line.startswith('#') or 
                     any(keyword in line.lower() for keyword in ['chapter', 'section', 'part']))):
                    
                    # 前のセクションを保存
                    if current_section and current_content:
                        sections.append(DocumentSection(
                            title=current_section,
                            content='\n'.join(current_content),
                            section_type="content",
                            confidence=0.8
                        ))
                    
                    # 新しいセクション開始
                    current_section = line
                    current_content = []
                else:
                    current_content.append(line)
            
            # 最後のセクションを保存
            if current_section and current_content:
                sections.append(DocumentSection(
                    title=current_section,
                    content='\n'.join(current_content),
                    section_type="content",
                    confidence=0.8
                ))
            
            # セクションが見つからない場合は全体を一つのセクションとする
            if not sections:
                sections.append(DocumentSection(
                    title="Document Content",
                    content=text,
                    section_type="full_document",
                    confidence=1.0
                ))
            
            return sections
            
        except Exception as e:
            logger.error(f"Document segmentation failed: {e}")
            return [DocumentSection(
                title="Document Content",
                content=text,
                section_type="full_document",
                confidence=0.5
            )]
    
    async def _extract_entities(self, text: str) -> List[DocumentEntity]:
        """固有表現抽出"""
        try:
            if not self.ner_pipeline:
                return []
            
            # テキストを適切な長さに分割（モデルの制限に対応）
            max_length = 512
            text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            all_entities = []
            offset = 0
            
            for chunk in text_chunks:
                if not chunk.strip():
                    offset += len(chunk)
                    continue
                
                # NER実行
                ner_results = self.ner_pipeline(chunk)
                
                for result in ner_results:
                    entity = DocumentEntity(
                        text=result['word'],
                        label=result['entity_group'],
                        confidence=result['score'],
                        start_pos=result['start'] + offset,
                        end_pos=result['end'] + offset,
                        context=chunk[max(0, result['start']-50):result['end']+50]
                    )
                    all_entities.append(entity)
                
                offset += len(chunk)
            
            # 重複除去と信頼度でソート
            unique_entities = []
            seen_texts = set()
            
            for entity in sorted(all_entities, key=lambda x: x.confidence, reverse=True):
                if entity.text not in seen_texts:
                    unique_entities.append(entity)
                    seen_texts.add(entity.text)
            
            return unique_entities[:50]  # 上位50個に制限
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _generate_summary(self, text: str) -> str:
        """要約生成"""
        try:
            if not self.summarization_pipeline:
                return "Summary generation not available"
            
            # テキストが長すぎる場合は分割
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            # 要約生成
            summary_result = self.summarization_pipeline(text)
            return summary_result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Summary generation failed: {e}"
    
    async def _extract_key_information(self, text: str, entities: List[DocumentEntity]) -> Dict[str, Any]:
        """キー情報抽出"""
        try:
            key_info = {
                "document_length": len(text),
                "word_count": len(text.split()),
                "entity_count": len(entities),
                "entity_types": {},
                "important_entities": [],
                "dates": [],
                "numbers": [],
                "emails": [],
                "urls": []
            }
            
            # エンティティタイプ別カウント
            for entity in entities:
                entity_type = entity.label
                if entity_type not in key_info["entity_types"]:
                    key_info["entity_types"][entity_type] = 0
                key_info["entity_types"][entity_type] += 1
                
                # 高信頼度エンティティ
                if entity.confidence > 0.9:
                    key_info["important_entities"].append({
                        "text": entity.text,
                        "type": entity.label,
                        "confidence": entity.confidence
                    })
            
            # パターンマッチングで特定情報抽出
            import re
            
            # 日付パターン
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
            ]
            
            for pattern in date_patterns:
                dates = re.findall(pattern, text, re.IGNORECASE)
                key_info["dates"].extend(dates)
            
            # 数値パターン
            numbers = re.findall(r'\$?[\d,]+\.?\d*', text)
            key_info["numbers"] = numbers[:10]  # 上位10個
            
            # メールアドレス
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            key_info["emails"] = emails
            
            # URL
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            key_info["urls"] = urls
            
            return key_info
            
        except Exception as e:
            logger.error(f"Key information extraction failed: {e}")
            return {"error": str(e)}
    
    def _calculate_total_confidence(self, sections: List[DocumentSection], entities: List[DocumentEntity]) -> float:
        """総合信頼度計算"""
        try:
            if not sections and not entities:
                return 0.0
            
            # セクション信頼度の平均
            section_confidence = 0.0
            if sections:
                section_confidence = sum(section.confidence for section in sections) / len(sections)
            
            # エンティティ信頼度の平均
            entity_confidence = 0.0
            if entities:
                entity_confidence = sum(entity.confidence for entity in entities) / len(entities)
            
            # 重み付き平均
            if sections and entities:
                total_confidence = (section_confidence * 0.4 + entity_confidence * 0.6)
            elif sections:
                total_confidence = section_confidence
            elif entities:
                total_confidence = entity_confidence
            else:
                total_confidence = 0.0
            
            return min(1.0, max(0.0, total_confidence))
            
        except Exception:
            return 0.5
    
    async def _analyze_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """画像解析"""
        try:
            image_path = Path(image_path)
            
            # OCRでテキスト抽出
            extracted_text = await self._extract_text_from_image(image_path)
            
            # 画像メタデータ
            metadata = {"file_size": image_path.stat().st_size}
            if Image is not None:
                try:
                    image = Image.open(image_path)
                    metadata.update({
                        "format": image.format,
                        "size": image.size,
                        "mode": image.mode
                    })
                except Exception as e:
                    logger.warning(f"Could not read image metadata: {e}")
            
            # 抽出されたテキストがある場合はエンティティ抽出
            entities = []
            if extracted_text.strip():
                entities = await self._extract_entities(extracted_text)
            
            return {
                "extracted_text": extracted_text,
                "entities": [
                    {
                        "text": entity.text,
                        "label": entity.label,
                        "confidence": entity.confidence
                    } for entity in entities
                ],
                "metadata": metadata,
                "confidence": 0.8 if extracted_text.strip() else 0.3
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "extracted_text": "",
                "entities": [],
                "metadata": {},
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _integrate_multimodal_results(self,
                                          text_analysis: Optional[DocumentAnalysisResult],
                                          image_analysis: Optional[Dict[str, Any]]) -> MultimodalResult:
        """マルチモーダル結果統合"""
        try:
            # 統合エンティティリスト
            integrated_entities = []
            
            # テキスト解析からのエンティティ
            if text_analysis and text_analysis.entities:
                integrated_entities.extend(text_analysis.entities)
            
            # 画像解析からのエンティティ
            if image_analysis and image_analysis.get("entities"):
                for img_entity in image_analysis["entities"]:
                    entity = DocumentEntity(
                        text=img_entity["text"],
                        label=img_entity["label"],
                        confidence=img_entity["confidence"],
                        start_pos=0,
                        end_pos=len(img_entity["text"]),
                        context="from_image"
                    )
                    integrated_entities.append(entity)
            
            # 重複エンティティの統合
            unique_entities = self._merge_duplicate_entities(integrated_entities)
            
            # クロスモーダル関係性分析
            cross_modal_relationships = self._analyze_cross_modal_relationships(
                text_analysis, image_analysis
            )
            
            # 統合信頼度計算
            combined_confidence = self._calculate_combined_confidence(
                text_analysis, image_analysis
            )
            
            # 最終要約生成
            final_summary = self._generate_integrated_summary(
                text_analysis, image_analysis, unique_entities
            )
            
            return MultimodalResult(
                text_analysis=text_analysis,
                image_analysis=image_analysis,
                combined_confidence=combined_confidence,
                integrated_entities=unique_entities,
                cross_modal_relationships=cross_modal_relationships,
                final_summary=final_summary
            )
            
        except Exception as e:
            logger.error(f"Multimodal integration failed: {e}")
            return MultimodalResult(
                text_analysis=text_analysis,
                image_analysis=image_analysis,
                combined_confidence=0.0,
                integrated_entities=[],
                cross_modal_relationships={},
                final_summary=f"Integration failed: {e}"
            )
    
    def _merge_duplicate_entities(self, entities: List[DocumentEntity]) -> List[DocumentEntity]:
        """重複エンティティのマージ"""
        entity_groups = {}
        
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        merged_entities = []
        for group in entity_groups.values():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # 最高信頼度のエンティティを選択し、信頼度を平均化
                best_entity = max(group, key=lambda x: x.confidence)
                avg_confidence = sum(e.confidence for e in group) / len(group)
                best_entity.confidence = avg_confidence
                merged_entities.append(best_entity)
        
        return sorted(merged_entities, key=lambda x: x.confidence, reverse=True)
    
    def _analyze_cross_modal_relationships(self,
                                         text_analysis: Optional[DocumentAnalysisResult],
                                         image_analysis: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        """クロスモーダル関係性分析"""
        relationships = {
            "text_image_overlap": [],
            "complementary_info": [],
            "contradictions": []
        }
        
        try:
            if not text_analysis or not image_analysis:
                return relationships
            
            text_entities = {e.text.lower() for e in text_analysis.entities}
            image_entities = {e["text"].lower() for e in image_analysis.get("entities", [])}
            
            # 重複する情報
            overlap = text_entities.intersection(image_entities)
            relationships["text_image_overlap"] = list(overlap)
            
            # 補完的な情報
            text_only = text_entities - image_entities
            image_only = image_entities - text_entities
            relationships["complementary_info"] = list(text_only.union(image_only))
            
            return relationships
            
        except Exception as e:
            logger.error(f"Cross-modal analysis failed: {e}")
            return relationships
    
    def _calculate_combined_confidence(self,
                                     text_analysis: Optional[DocumentAnalysisResult],
                                     image_analysis: Optional[Dict[str, Any]]) -> float:
        """統合信頼度計算"""
        try:
            confidences = []
            
            if text_analysis:
                confidences.append(text_analysis.total_confidence)
            
            if image_analysis:
                confidences.append(image_analysis.get("confidence", 0.0))
            
            if not confidences:
                return 0.0
            
            return sum(confidences) / len(confidences)
            
        except Exception:
            return 0.0
    
    def _generate_integrated_summary(self,
                                   text_analysis: Optional[DocumentAnalysisResult],
                                   image_analysis: Optional[Dict[str, Any]],
                                   entities: List[DocumentEntity]) -> str:
        """統合要約生成"""
        try:
            summary_parts = []
            
            if text_analysis:
                summary_parts.append(f"Text Analysis: {text_analysis.summary}")
            
            if image_analysis and image_analysis.get("extracted_text"):
                summary_parts.append(f"Image Content: {image_analysis['extracted_text'][:200]}...")
            
            if entities:
                top_entities = entities[:5]
                entity_summary = ", ".join([f"{e.text} ({e.label})" for e in top_entities])
                summary_parts.append(f"Key Entities: {entity_summary}")
            
            return " | ".join(summary_parts) if summary_parts else "No content analyzed"
            
        except Exception as e:
            return f"Summary generation failed: {e}"
    
    async def batch_analyze(self,
                          file_paths: List[Union[str, Path]],
                          max_concurrent: int = 3) -> List[DocumentAnalysisResult]:
        """バッチドキュメント解析"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single(file_path: Union[str, Path]) -> DocumentAnalysisResult:
            async with semaphore:
                return await self.analyze_document(file_path)
        
        tasks = [analyze_single(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 例外を結果に変換
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(DocumentAnalysisResult(
                    document_type="error",
                    language="unknown",
                    total_confidence=0.0,
                    sections=[],
                    entities=[],
                    summary="",
                    key_information={},
                    processing_time=0.0,
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
            if self.ner_pipeline:
                del self.ner_pipeline
            if self.classification_pipeline:
                del self.classification_pipeline
            if self.summarization_pipeline:
                del self.summarization_pipeline
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Document AI cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# 使用例とテスト用のヘルパー関数
async def demo_document_analysis():
    """デモ用ドキュメント解析実行"""
    doc_ai = HuggingFaceDocumentAI()
    
    try:
        # 初期化
        if not await doc_ai.initialize():
            print("Failed to initialize Document AI")
            return
        
        print("=== Document AI Demo ===")
        
        # テスト用テキストファイル作成
        test_content = """
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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # ドキュメント解析実行
            result = await doc_ai.analyze_document(temp_file)
            
            print(f"\nDocument Type: {result.document_type}")
            print(f"Language: {result.language}")
            print(f"Confidence: {result.total_confidence:.2f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            
            print(f"\nSections ({len(result.sections)}):")
            for section in result.sections:
                print(f"  - {section.title} (confidence: {section.confidence:.2f})")
            
            print(f"\nEntities ({len(result.entities)}):")
            for entity in result.entities[:5]:  # 上位5個
                print(f"  - {entity.text} ({entity.label}, confidence: {entity.confidence:.2f})")
            
            print(f"\nSummary: {result.summary}")
            
            print(f"\nKey Information:")
            for key, value in result.key_information.items():
                if isinstance(value, list) and value:
                    print(f"  {key}: {value[:3]}...")  # 最初の3個のみ表示
                elif not isinstance(value, list):
                    print(f"  {key}: {value}")
            
            # メモリ使用量表示
            memory_usage = doc_ai.get_memory_usage()
            print(f"\nMemory Usage: {memory_usage}")
            
        finally:
            Path(temp_file).unlink()  # 一時ファイル削除
        
    finally:
        await doc_ai.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_document_analysis())