"""
HuggingFace Document AI デモスクリプト

RTX 4050 6GB VRAM環境でのドキュメント解析機能をデモンストレーションします。

使用方法:
    python -m src.advanced_agent.multimodal.demo_document_ai

要件: 3.3, 3.5
"""

import asyncio
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import logging

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.advanced_agent.multimodal.document_ai import (
    HuggingFaceDocumentAI,
    DocumentAnalysisResult,
    MultimodalResult
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentAIDemo:
    """ドキュメント AI デモクラス"""
    
    def __init__(self):
        self.doc_ai = HuggingFaceDocumentAI(
            ner_model="dbmdz/bert-large-cased-finetuned-conll03-english",
            classification_model="microsoft/DialoGPT-medium",
            max_vram_gb=4.0  # RTX 4050用設定
        )
        
        # デモ用サンプルドキュメント
        self.sample_documents = {
            "business_report": """
QUARTERLY BUSINESS REPORT
Q4 2024 Performance Analysis

EXECUTIVE SUMMARY
This report presents the financial and operational performance of TechCorp Inc. 
for the fourth quarter of 2024.

FINANCIAL HIGHLIGHTS
• Total Revenue: $2,450,000 (up 18% from Q3)
• Net Profit: $485,000 (up 22% from Q3)
• Operating Expenses: $1,965,000
• EBITDA Margin: 28.5%

KEY PERSONNEL
CEO: Sarah Johnson (sarah.johnson@techcorp.com)
CFO: Michael Chen (michael.chen@techcorp.com)
CTO: Dr. Emily Rodriguez (emily.rodriguez@techcorp.com)

MARKET ANALYSIS
The technology sector showed strong growth in Q4 2024.
Our main competitors include:
- InnovateTech Solutions
- Digital Dynamics Corp
- Future Systems Ltd

STRATEGIC INITIATIVES
1. Cloud Migration Project (Budget: $500,000)
2. AI Research Division Launch
3. International Expansion to Europe

RISK FACTORS
• Market volatility in tech sector
• Regulatory changes in data privacy
• Supply chain disruptions

CONCLUSION
TechCorp is well-positioned for continued growth in 2025.
The board recommends increasing R&D investment by 15%.

Contact Information:
Headquarters: 123 Innovation Drive, Silicon Valley, CA 94025
Phone: +1 (555) 123-4567
Website: www.techcorp.com
            """,
            
            "contract": """
SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into on January 15, 2025,
between TechCorp Inc. ("Licensor") and Enterprise Solutions Ltd. ("Licensee").

PARTIES
Licensor: TechCorp Inc.
Address: 123 Innovation Drive, Silicon Valley, CA 94025
Contact: legal@techcorp.com

Licensee: Enterprise Solutions Ltd.
Address: 456 Business Park, New York, NY 10001
Contact: contracts@enterprisesolutions.com

TERMS AND CONDITIONS

1. GRANT OF LICENSE
Licensor hereby grants to Licensee a non-exclusive, non-transferable license
to use the Software for internal business purposes only.

2. LICENSE FEE
The total license fee is $50,000 USD, payable within 30 days of execution.

3. TERM
This Agreement shall commence on February 1, 2025, and continue for 12 months.

4. RESTRICTIONS
Licensee shall not:
- Reverse engineer the Software
- Distribute copies to third parties
- Use for commercial resale

5. SUPPORT AND MAINTENANCE
Licensor will provide technical support during business hours (9 AM - 5 PM PST).

6. TERMINATION
Either party may terminate with 30 days written notice.

7. GOVERNING LAW
This Agreement shall be governed by California state law.

SIGNATURES
Licensor: _________________________ Date: _________
Sarah Johnson, CEO, TechCorp Inc.

Licensee: _________________________ Date: _________
Robert Wilson, CTO, Enterprise Solutions Ltd.
            """,
            
            "invoice": """
INVOICE

TechCorp Inc.
123 Innovation Drive
Silicon Valley, CA 94025
Phone: (555) 123-4567
Email: billing@techcorp.com

BILL TO:
Enterprise Solutions Ltd.
456 Business Park
New York, NY 10001

Invoice Number: INV-2025-001
Invoice Date: January 20, 2025
Due Date: February 19, 2025
Payment Terms: Net 30

DESCRIPTION OF SERVICES:
1. Software License Fee (Annual)          $50,000.00
2. Implementation Services (40 hours)     $8,000.00
3. Training Sessions (2 days)             $3,000.00
4. Technical Support (Premium)            $2,400.00

                                Subtotal: $63,400.00
                                Tax (8.5%): $5,389.00
                                TOTAL DUE: $68,789.00

PAYMENT INSTRUCTIONS:
Please remit payment to:
Bank: Silicon Valley Bank
Account: 1234567890
Routing: 987654321

Or pay online at: www.techcorp.com/payments
Reference: INV-2025-001

Thank you for your business!

Questions? Contact: accounting@techcorp.com
            """,
            
            "resume": """
JOHN ALEXANDER SMITH
Senior Software Engineer

Contact Information:
Email: john.smith@email.com
Phone: (555) 987-6543
LinkedIn: linkedin.com/in/johnsmith
GitHub: github.com/johnsmith
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Experienced software engineer with 8+ years in full-stack development.
Expertise in Python, JavaScript, and cloud technologies.
Proven track record of leading teams and delivering scalable solutions.

TECHNICAL SKILLS
• Programming Languages: Python, JavaScript, Java, Go, TypeScript
• Frameworks: Django, React, Node.js, FastAPI, Flask
• Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
• Cloud Platforms: AWS, Google Cloud, Azure
• DevOps: Docker, Kubernetes, Jenkins, GitLab CI/CD
• Tools: Git, JIRA, Confluence, VS Code

WORK EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2021 - Present
• Led development of microservices architecture serving 1M+ users
• Reduced system latency by 40% through optimization initiatives
• Mentored 5 junior developers and conducted code reviews
• Implemented CI/CD pipelines reducing deployment time by 60%

Software Engineer | InnovateTech Solutions | 2018 - 2021
• Developed RESTful APIs using Python and Django framework
• Built responsive web applications with React and TypeScript
• Collaborated with product team to define technical requirements
• Maintained 99.9% uptime for critical business applications

Junior Developer | StartupXYZ | 2016 - 2018
• Created web applications using JavaScript and Node.js
• Participated in agile development processes and daily standups
• Wrote unit tests achieving 90%+ code coverage
• Assisted in database design and optimization

EDUCATION

Master of Science in Computer Science
Stanford University | 2014 - 2016
GPA: 3.8/4.0

Bachelor of Science in Software Engineering
UC Berkeley | 2010 - 2014
Magna Cum Laude, GPA: 3.7/4.0

CERTIFICATIONS
• AWS Certified Solutions Architect (2023)
• Google Cloud Professional Developer (2022)
• Certified Kubernetes Administrator (2021)

PROJECTS
• Open Source Contributor: Django REST Framework (500+ stars)
• Personal Project: AI-powered task management app
• Hackathon Winner: Best Technical Innovation (2020)

LANGUAGES
• English (Native)
• Spanish (Conversational)
• Mandarin (Basic)
            """
        }
    
    async def run_demo(self):
        """デモ実行"""
        print("=" * 60)
        print("HuggingFace Document AI Demo")
        print("RTX 4050 6GB VRAM Optimized")
        print("=" * 60)
        
        try:
            # 初期化
            print("\n🔧 Initializing Document AI...")
            start_time = time.time()
            
            if not await self.doc_ai.initialize():
                print("❌ Failed to initialize Document AI")
                return
            
            init_time = time.time() - start_time
            print(f"✅ Initialization completed in {init_time:.2f} seconds")
            
            # メモリ使用量表示
            memory_usage = self.doc_ai.get_memory_usage()
            if memory_usage:
                print(f"📊 GPU Memory: {memory_usage.get('gpu_allocated_mb', 0):.1f} MB allocated")
            
            # インタラクティブデモ
            await self._interactive_demo()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            logger.error(f"Demo error: {e}", exc_info=True)
        finally:
            print("\n🧹 Cleaning up...")
            await self.doc_ai.cleanup()
            print("✅ Cleanup completed")
    
    async def _interactive_demo(self):
        """インタラクティブデモ"""
        while True:
            print("\n" + "=" * 50)
            print("Choose demo mode:")
            print("1. Analyze Sample Documents")
            print("2. Upload Custom Document")
            print("3. Multimodal Analysis")
            print("4. Batch Processing")
            print("5. Performance Benchmark")
            print("6. Entity Extraction Demo")
            print("0. Exit")
            print("=" * 50)
            
            try:
                choice = input("\nEnter your choice (0-6): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    await self._sample_documents_demo()
                elif choice == "2":
                    await self._custom_document_demo()
                elif choice == "3":
                    await self._multimodal_demo()
                elif choice == "4":
                    await self._batch_processing_demo()
                elif choice == "5":
                    await self._performance_benchmark()
                elif choice == "6":
                    await self._entity_extraction_demo()
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    async def _sample_documents_demo(self):
        """サンプルドキュメントデモ"""
        print("\n📄 Sample Documents Analysis")
        print("-" * 40)
        
        for doc_type, content in self.sample_documents.items():
            print(f"\n📋 Analyzing: {doc_type.replace('_', ' ').title()}")
            
            if input("Analyze this document? (y/n): ").lower() != 'y':
                continue
            
            await self._analyze_and_display(content, doc_type)
    
    async def _custom_document_demo(self):
        """カスタムドキュメントデモ"""
        print("\n📁 Custom Document Analysis")
        print("-" * 40)
        
        file_path = input("Enter file path (or 'text' for direct input): ").strip()
        
        if file_path.lower() == 'text':
            print("Enter your text (press Ctrl+D or Ctrl+Z when finished):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            
            content = '\n'.join(lines)
            if content.strip():
                await self._analyze_and_display(content, "custom_text")
            else:
                print("❌ No text entered")
        else:
            path = Path(file_path)
            if path.exists():
                try:
                    result = await self.doc_ai.analyze_document(path)
                    self._display_analysis_result(result)
                except Exception as e:
                    print(f"❌ Analysis failed: {e}")
            else:
                print("❌ File not found")
    
    async def _multimodal_demo(self):
        """マルチモーダルデモ"""
        print("\n🖼️  Multimodal Analysis Demo")
        print("-" * 40)
        
        text_input = input("Enter text content (or press Enter to skip): ").strip()
        image_path = input("Enter image file path (or press Enter to skip): ").strip()
        
        if not text_input and not image_path:
            print("❌ No input provided")
            return
        
        try:
            print("\n🔄 Performing multimodal analysis...")
            start_time = time.time()
            
            result = await self.doc_ai.analyze_multimodal(
                text_content=text_input if text_input else None,
                image_file=Path(image_path) if image_path and Path(image_path).exists() else None
            )
            
            analysis_time = time.time() - start_time
            print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
            
            self._display_multimodal_result(result)
            
        except Exception as e:
            print(f"❌ Multimodal analysis failed: {e}")
    
    async def _batch_processing_demo(self):
        """バッチ処理デモ"""
        print("\n🔄 Batch Processing Demo")
        print("-" * 40)
        
        # サンプルドキュメントでバッチ処理
        print("Creating temporary files for batch processing...")
        
        temp_files = []
        try:
            for doc_type, content in list(self.sample_documents.items())[:3]:  # 最初の3つ
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(content)
                    temp_files.append(Path(f.name))
                    print(f"  Created: {doc_type}")
            
            print(f"\n🚀 Processing {len(temp_files)} documents...")
            start_time = time.time()
            
            results = await self.doc_ai.batch_analyze(temp_files, max_concurrent=2)
            
            batch_time = time.time() - start_time
            print(f"⏱️  Batch processing completed in {batch_time:.2f} seconds")
            
            # 結果サマリー表示
            print(f"\n📊 Batch Results Summary:")
            for i, result in enumerate(results, 1):
                print(f"  Document {i}:")
                print(f"    Type: {result.document_type}")
                print(f"    Confidence: {result.confidence:.2f}")
                print(f"    Entities: {len(result.entities)}")
                print(f"    Processing Time: {result.processing_time:.2f}s")
                if result.error_message:
                    print(f"    Error: {result.error_message}")
            
        finally:
            # 一時ファイル削除
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception:
                    pass
    
    async def _performance_benchmark(self):
        """パフォーマンスベンチマーク"""
        print("\n⚡ Performance Benchmark")
        print("-" * 40)
        
        # 異なるサイズのドキュメントでテスト
        test_docs = {
            "small": self.sample_documents["invoice"][:500],
            "medium": self.sample_documents["business_report"],
            "large": self.sample_documents["resume"] * 3  # 3倍に拡張
        }
        
        print("Testing performance with different document sizes...")
        
        results = {}
        for size, content in test_docs.items():
            print(f"\n📏 Testing {size} document ({len(content)} chars)...")
            
            start_time = time.time()
            result = await self._analyze_content(content)
            end_time = time.time()
            
            processing_time = end_time - start_time
            results[size] = {
                "chars": len(content),
                "words": len(content.split()),
                "processing_time": processing_time,
                "chars_per_second": len(content) / processing_time,
                "confidence": result.total_confidence if result else 0.0
            }
            
            print(f"  ⏱️  Time: {processing_time:.2f}s")
            print(f"  🚀 Speed: {results[size]['chars_per_second']:.0f} chars/sec")
        
        # 統計表示
        print(f"\n📊 Performance Statistics:")
        for size, stats in results.items():
            print(f"  {size.title()} Document:")
            print(f"    Characters: {stats['chars']:,}")
            print(f"    Processing Time: {stats['processing_time']:.2f}s")
            print(f"    Speed: {stats['chars_per_second']:.0f} chars/sec")
            print(f"    Confidence: {stats['confidence']:.2f}")
        
        # メモリ使用量
        memory_usage = self.doc_ai.get_memory_usage()
        if memory_usage:
            print(f"\n💾 Memory Usage:")
            for key, value in memory_usage.items():
                print(f"    {key}: {value:.1f} MB")
    
    async def _entity_extraction_demo(self):
        """エンティティ抽出デモ"""
        print("\n🏷️  Entity Extraction Demo")
        print("-" * 40)
        
        sample_text = """
        Meeting Notes - January 25, 2025
        
        Attendees:
        - Sarah Johnson (CEO, sarah.johnson@techcorp.com)
        - Michael Chen (CFO, michael.chen@techcorp.com)
        - Dr. Emily Rodriguez (CTO)
        
        Agenda:
        1. Q4 Financial Review ($2.45M revenue)
        2. New Product Launch (March 15, 2025)
        3. Partnership with InnovateTech Solutions
        
        Action Items:
        - Budget approval for $500K cloud migration
        - Schedule meeting with Google Cloud team
        - Review contract with Enterprise Solutions Ltd.
        
        Next Meeting: February 1, 2025 at 2:00 PM PST
        Location: Conference Room A, 123 Innovation Drive
        """
        
        print("Sample text for entity extraction:")
        print("-" * 30)
        print(sample_text[:300] + "...")
        
        if input("\nExtract entities from this text? (y/n): ").lower() == 'y':
            print("\n🔍 Extracting entities...")
            
            # エンティティ抽出（モック版）
            entities = await self.doc_ai._extract_entities(sample_text)
            
            if entities:
                print(f"\n📋 Found {len(entities)} entities:")
                
                # エンティティタイプ別にグループ化
                entity_groups = {}
                for entity in entities:
                    if entity.label not in entity_groups:
                        entity_groups[entity.label] = []
                    entity_groups[entity.label].append(entity)
                
                for entity_type, group in entity_groups.items():
                    print(f"\n  {entity_type}:")
                    for entity in sorted(group, key=lambda x: x.confidence, reverse=True):
                        print(f"    • {entity.text} (confidence: {entity.confidence:.2f})")
            else:
                print("❌ No entities extracted (NER pipeline not initialized)")
    
    async def _analyze_and_display(self, content: str, doc_type: str):
        """コンテンツ解析と結果表示"""
        result = await self._analyze_content(content)
        if result:
            print(f"\n📊 Analysis Results for {doc_type}:")
            self._display_analysis_result(result)
    
    async def _analyze_content(self, content: str) -> DocumentAnalysisResult:
        """コンテンツ解析"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = Path(f.name)
        
        try:
            return await self.doc_ai.analyze_document(temp_file)
        finally:
            temp_file.unlink()
    
    def _display_analysis_result(self, result: DocumentAnalysisResult):
        """解析結果表示"""
        print("\n" + "=" * 50)
        print("📄 Document Analysis Results")
        print("-" * 20)
        
        print(f"Document Type: {result.document_type}")
        print(f"Language: {result.language}")
        print(f"Overall Confidence: {result.total_confidence:.2f}")
        print(f"Processing Time: {result.processing_time:.2f} seconds")
        
        if result.error_message:
            print(f"❌ Error: {result.error_message}")
            return
        
        print(f"\n📑 Sections ({len(result.sections)}):")
        for i, section in enumerate(result.sections[:3], 1):  # 最初の3つのみ
            print(f"  {i}. {section.title}")
            print(f"     Type: {section.section_type}")
            print(f"     Confidence: {section.confidence:.2f}")
            print(f"     Content: {section.content[:100]}...")
        
        if len(result.sections) > 3:
            print(f"     ... and {len(result.sections) - 3} more sections")
        
        print(f"\n🏷️  Entities ({len(result.entities)}):")
        for entity in result.entities[:10]:  # 上位10個
            print(f"  • {entity.text} ({entity.label}) - confidence: {entity.confidence:.2f}")
        
        if len(result.entities) > 10:
            print(f"  ... and {len(result.entities) - 10} more entities")
        
        print(f"\n📝 Summary:")
        print(f"  {result.summary}")
        
        print(f"\n🔑 Key Information:")
        for key, value in result.key_information.items():
            if isinstance(value, list) and value:
                if len(value) <= 3:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value[:3]}... ({len(value)} total)")
            elif not isinstance(value, list):
                print(f"  {key}: {value}")
        
        print("=" * 50)
    
    def _display_multimodal_result(self, result: MultimodalResult):
        """マルチモーダル結果表示"""
        print("\n" + "=" * 50)
        print("🖼️  Multimodal Analysis Results")
        print("-" * 20)
        
        print(f"Combined Confidence: {result.combined_confidence:.2f}")
        
        if result.text_analysis:
            print(f"\n📄 Text Analysis:")
            print(f"  Type: {result.text_analysis.document_type}")
            print(f"  Confidence: {result.text_analysis.total_confidence:.2f}")
            print(f"  Entities: {len(result.text_analysis.entities)}")
        
        if result.image_analysis:
            print(f"\n🖼️  Image Analysis:")
            print(f"  Confidence: {result.image_analysis.get('confidence', 0):.2f}")
            print(f"  Extracted Text: {result.image_analysis.get('extracted_text', 'None')[:100]}...")
            print(f"  Entities: {len(result.image_analysis.get('entities', []))}")
        
        print(f"\n🔗 Integrated Entities ({len(result.integrated_entities)}):")
        for entity in result.integrated_entities[:10]:
            print(f"  • {entity.text} ({entity.label}) - confidence: {entity.confidence:.2f}")
        
        print(f"\n🔄 Cross-Modal Relationships:")
        for rel_type, items in result.cross_modal_relationships.items():
            if items:
                print(f"  {rel_type}: {len(items)} items")
        
        print(f"\n📋 Final Summary:")
        print(f"  {result.final_summary}")
        
        print("=" * 50)


async def main():
    """メイン関数"""
    demo = DocumentAIDemo()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main error: {e}", exc_info=True)