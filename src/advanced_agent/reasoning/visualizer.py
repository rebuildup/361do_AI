"""
Visualization system for reasoning results
推論結果可視化システム
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from io import StringIO
import base64

from .result_parser import StructuredResult, ParsedElement, ParseResultType
from .cot_engine import CoTResponse, ReasoningStep, CoTStep, ReasoningState

logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """可視化タイプ"""
    FLOWCHART = "flowchart"
    TIMELINE = "timeline"
    HIERARCHY = "hierarchy"
    NETWORK = "network"
    HEATMAP = "heatmap"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    TEXT_ANALYSIS = "text_analysis"
    QUALITY_METRICS = "quality_metrics"


@dataclass
class VisualizationConfig:
    """可視化設定"""
    width: int = 800
    height: int = 600
    theme: str = "light"  # "light", "dark", "colorful"
    font_size: int = 12
    color_palette: str = "viridis"
    interactive: bool = True
    export_format: str = "html"  # "html", "png", "svg", "json"
    show_metadata: bool = True
    animation: bool = False


@dataclass
class VisualizationResult:
    """可視化結果"""
    visualization_type: VisualizationType
    title: str
    data: Any
    config: VisualizationConfig
    html_content: Optional[str] = None
    image_data: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)


class ReasoningVisualizer:
    """推論結果可視化器"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.color_palettes = self._initialize_color_palettes()
        self.visualization_templates = self._initialize_templates()
        
        # 可視化統計
        self.visualization_stats = {
            "total_visualizations": 0,
            "type_distribution": {},
            "average_creation_time": 0.0,
            "creation_times": []
        }
        
        # スタイル設定
        self._setup_matplotlib_style()
        
        logger.info("Reasoning visualizer initialized")
    
    def _initialize_color_palettes(self) -> Dict[str, List[str]]:
        """カラーパレット初期化"""
        return {
            "viridis": ["#440154", "#31688e", "#35b779", "#6ece58", "#fde725"],
            "plasma": ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636"],
            "inferno": ["#000004", "#420a68", "#932567", "#dd513a", "#fca50a"],
            "magma": ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d"],
            "cool": ["#00a1d5", "#00b4d8", "#00c9c7", "#00dbb6", "#00eda5"],
            "warm": ["#ff6b6b", "#ff8e53", "#ffa726", "#ffb74d", "#ffcc80"],
            "neutral": ["#2c3e50", "#34495e", "#7f8c8d", "#95a5a6", "#bdc3c7"]
        }
    
    def _initialize_templates(self) -> Dict[VisualizationType, Dict[str, Any]]:
        """可視化テンプレート初期化"""
        return {
            VisualizationType.FLOWCHART: {
                "node_style": "rounded",
                "edge_style": "curved",
                "layout": "hierarchical"
            },
            VisualizationType.TIMELINE: {
                "orientation": "horizontal",
                "show_milestones": True,
                "show_duration": True
            },
            VisualizationType.HIERARCHY: {
                "direction": "top_down",
                "spacing": "uniform",
                "show_labels": True
            },
            VisualizationType.NETWORK: {
                "layout": "spring",
                "node_size": "degree",
                "edge_width": "weight"
            },
            VisualizationType.HEATMAP: {
                "color_scheme": "RdYlBu",
                "show_values": True,
                "annotate": True
            }
        }
    
    def _setup_matplotlib_style(self):
        """matplotlibスタイル設定"""
        plt.style.use('default')
        sns.set_palette(self.color_palettes.get(self.config.color_palette, "viridis"))
        
        # フォント設定
        plt.rcParams['font.size'] = self.config.font_size
        plt.rcParams['figure.figsize'] = (self.config.width / 100, self.config.height / 100)
        
        if self.config.theme == "dark":
            plt.style.use('dark_background')
    
    def visualize_reasoning_flow(self, response: CoTResponse) -> VisualizationResult:
        """推論フローの可視化"""
        import time
        start_time = time.time()
        
        try:
            # 推論ステップの分析
            steps_data = self._analyze_reasoning_steps(response.reasoning_steps)
            
            # フローチャート作成
            if self.config.interactive:
                fig = self._create_interactive_flowchart(steps_data, response)
                html_content = fig.to_html(include_plotlyjs='cdn')
            else:
                fig = self._create_static_flowchart(steps_data, response)
                html_content = self._fig_to_html(fig)
            
            creation_time = time.time() - start_time
            
            result = VisualizationResult(
                visualization_type=VisualizationType.FLOWCHART,
                title=f"Reasoning Flow - {response.request_id}",
                data=steps_data,
                config=self.config,
                html_content=html_content,
                metadata={
                    "step_count": len(response.reasoning_steps),
                    "total_time": response.processing_time,
                    "confidence": response.final_confidence,
                    "model": response.model_used
                }
            )
            
            self._update_visualization_stats(result, creation_time)
            
            logger.info(f"Reasoning flow visualized in {creation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning flow visualization failed: {e}")
            return self._create_error_visualization(str(e), start_time)
    
    def visualize_structured_result(self, result: StructuredResult) -> VisualizationResult:
        """構造化結果の可視化"""
        import time
        start_time = time.time()
        
        try:
            # 構造分析
            structure_data = self._analyze_structure(result)
            
            # 階層構造の可視化
            if self.config.interactive:
                fig = self._create_interactive_hierarchy(structure_data, result)
                html_content = fig.to_html(include_plotlyjs='cdn')
            else:
                fig = self._create_static_hierarchy(structure_data, result)
                html_content = self._fig_to_html(fig)
            
            creation_time = time.time() - start_time
            
            visualization_result = VisualizationResult(
                visualization_type=VisualizationType.HIERARCHY,
                title=f"Structured Result - {result.structure_type.value}",
                data=structure_data,
                config=self.config,
                html_content=html_content,
                metadata={
                    "structure_type": result.structure_type.value,
                    "element_count": len(result.parsed_elements),
                    "confidence": result.confidence,
                    "parsing_time": result.parsing_time
                }
            )
            
            self._update_visualization_stats(visualization_result, creation_time)
            
            logger.info(f"Structured result visualized in {creation_time:.3f}s")
            
            return visualization_result
            
        except Exception as e:
            logger.error(f"Structured result visualization failed: {e}")
            return self._create_error_visualization(str(e), start_time)
    
    def visualize_quality_metrics(self, evaluations: List[Any]) -> VisualizationResult:
        """品質メトリクスの可視化"""
        import time
        start_time = time.time()
        
        try:
            # メトリクスデータの準備
            metrics_data = self._prepare_quality_metrics_data(evaluations)
            
            # ヒートマップ作成
            if self.config.interactive:
                fig = self._create_interactive_heatmap(metrics_data)
                html_content = fig.to_html(include_plotlyjs='cdn')
            else:
                fig = self._create_static_heatmap(metrics_data)
                html_content = self._fig_to_html(fig)
            
            creation_time = time.time() - start_time
            
            result = VisualizationResult(
                visualization_type=VisualizationType.HEATMAP,
                title="Quality Metrics Heatmap",
                data=metrics_data,
                config=self.config,
                html_content=html_content,
                metadata={
                    "evaluation_count": len(evaluations),
                    "metrics_count": len(metrics_data.get("metrics", [])),
                    "average_score": metrics_data.get("average_score", 0.0)
                }
            )
            
            self._update_visualization_stats(result, creation_time)
            
            logger.info(f"Quality metrics visualized in {creation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Quality metrics visualization failed: {e}")
            return self._create_error_visualization(str(e), start_time)
    
    def visualize_text_analysis(self, text: str, analysis_type: str = "sentiment") -> VisualizationResult:
        """テキスト分析の可視化"""
        import time
        start_time = time.time()
        
        try:
            # テキスト分析
            analysis_data = self._analyze_text(text, analysis_type)
            
            # 可視化タイプの決定
            if analysis_type == "sentiment":
                viz_type = VisualizationType.BAR_CHART
                if self.config.interactive:
                    fig = self._create_sentiment_chart(analysis_data)
                    html_content = fig.to_html(include_plotlyjs='cdn')
                else:
                    fig = self._create_static_sentiment_chart(analysis_data)
                    html_content = self._fig_to_html(fig)
            else:
                viz_type = VisualizationType.TEXT_ANALYSIS
                html_content = self._create_text_analysis_html(analysis_data)
            
            creation_time = time.time() - start_time
            
            result = VisualizationResult(
                visualization_type=viz_type,
                title=f"Text Analysis - {analysis_type}",
                data=analysis_data,
                config=self.config,
                html_content=html_content,
                metadata={
                    "text_length": len(text),
                    "analysis_type": analysis_type,
                    "word_count": analysis_data.get("word_count", 0)
                }
            )
            
            self._update_visualization_stats(result, creation_time)
            
            logger.info(f"Text analysis visualized in {creation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Text analysis visualization failed: {e}")
            return self._create_error_visualization(str(e), start_time)
    
    def _analyze_reasoning_steps(self, steps: List[ReasoningStep]) -> Dict[str, Any]:
        """推論ステップの分析"""
        analysis = {
            "steps": [],
            "step_types": {},
            "total_steps": len(steps),
            "average_confidence": 0.0,
            "time_distribution": []
        }
        
        total_confidence = 0.0
        
        for i, step in enumerate(steps):
            step_data = {
                "step_number": i + 1,
                "type": step.step_type.value,
                "content": step.content,
                "confidence": getattr(step, 'confidence', 0.8),
                "duration": getattr(step, 'duration', 1.0)
            }
            
            analysis["steps"].append(step_data)
            
            # ステップタイプの集計
            step_type = step.step_type.value
            analysis["step_types"][step_type] = analysis["step_types"].get(step_type, 0) + 1
            
            total_confidence += step_data["confidence"]
            analysis["time_distribution"].append(step_data["duration"])
        
        analysis["average_confidence"] = total_confidence / len(steps) if steps else 0.0
        
        return analysis
    
    def _analyze_structure(self, result: StructuredResult) -> Dict[str, Any]:
        """構造分析"""
        analysis = {
            "structure_type": result.structure_type.value,
            "elements": [],
            "element_types": {},
            "confidence_distribution": [],
            "extracted_data_summary": {}
        }
        
        for element in result.parsed_elements:
            element_data = {
                "type": element.element_type.value,
                "confidence": element.confidence,
                "position": element.position,
                "metadata": element.metadata
            }
            
            analysis["elements"].append(element_data)
            
            # 要素タイプの集計
            element_type = element.element_type.value
            analysis["element_types"][element_type] = analysis["element_types"].get(element_type, 0) + 1
            
            analysis["confidence_distribution"].append(element.confidence)
        
        # 抽出データのサマリー
        for key, data in result.extracted_data.items():
            analysis["extracted_data_summary"][key] = {
                "count": data["count"],
                "type": data["type"]
            }
        
        return analysis
    
    def _prepare_quality_metrics_data(self, evaluations: List[Any]) -> Dict[str, Any]:
        """品質メトリクスデータの準備"""
        metrics_data = {
            "metrics": [],
            "scores": [],
            "dimensions": [],
            "average_score": 0.0
        }
        
        total_score = 0.0
        
        for evaluation in evaluations:
            if hasattr(evaluation, 'dimension_scores'):
                for dimension, score in evaluation.dimension_scores.items():
                    metrics_data["metrics"].append(dimension.value)
                    metrics_data["scores"].append(score.score)
                    metrics_data["dimensions"].append(dimension.value)
                    total_score += score.score
        
        metrics_data["average_score"] = total_score / len(metrics_data["scores"]) if metrics_data["scores"] else 0.0
        
        return metrics_data
    
    def _analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """テキスト分析"""
        analysis = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len([s for s in text.split('.') if s.strip()]),
            "analysis_type": analysis_type
        }
        
        if analysis_type == "sentiment":
            # 簡単な感情分析（実際の実装ではより高度な分析を使用）
            positive_words = ["良い", "素晴らしい", "優秀", "成功", "達成", "満足"]
            negative_words = ["悪い", "問題", "失敗", "困難", "不満", "間違い"]
            
            positive_count = sum(text.count(word) for word in positive_words)
            negative_count = sum(text.count(word) for word in negative_words)
            
            analysis["sentiment"] = {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": max(0, analysis["word_count"] - positive_count - negative_count),
                "score": (positive_count - negative_count) / max(1, analysis["word_count"])
            }
        
        return analysis
    
    def _create_interactive_flowchart(self, steps_data: Dict[str, Any], response: CoTResponse) -> go.Figure:
        """インタラクティブフローチャート作成"""
        fig = go.Figure()
        
        # ノードの位置計算
        positions = self._calculate_flowchart_positions(steps_data["steps"])
        
        # ノードの追加
        for i, (step, pos) in enumerate(zip(steps_data["steps"], positions)):
            fig.add_trace(go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode='markers+text',
                marker=dict(
                    size=50,
                    color=self._get_step_color(step["type"]),
                    line=dict(width=2, color='white')
                ),
                text=[f"Step {step['step_number']}"],
                textposition="middle center",
                name=f"Step {step['step_number']}",
                hovertemplate=f"<b>{step['type']}</b><br>{step['content'][:100]}...<extra></extra>"
            ))
        
        # エッジの追加
        for i in range(len(positions) - 1):
            fig.add_trace(go.Scatter(
                x=[positions[i][0], positions[i+1][0]],
                y=[positions[i][1], positions[i+1][1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # レイアウト設定
        fig.update_layout(
            title=f"Reasoning Flow - {response.request_id}",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def _create_static_flowchart(self, steps_data: Dict[str, Any], response: CoTResponse) -> plt.Figure:
        """静的フローチャート作成"""
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        # ノードの位置計算
        positions = self._calculate_flowchart_positions(steps_data["steps"])
        
        # ノードの描画
        for i, (step, pos) in enumerate(zip(steps_data["steps"], positions)):
            # ノードの描画
            circle = plt.Circle(pos, 0.1, 
                              color=self._get_step_color(step["type"]), 
                              alpha=0.7)
            ax.add_patch(circle)
            
            # テキストの追加
            ax.text(pos[0], pos[1], f"{step['step_number']}", 
                   ha='center', va='center', fontweight='bold', color='white')
        
        # エッジの描画
        for i in range(len(positions) - 1):
            ax.plot([positions[i][0], positions[i+1][0]], 
                   [positions[i][1], positions[i+1][1]], 
                   'k-', alpha=0.5, linewidth=2)
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Reasoning Flow - {response.request_id}")
        
        return fig
    
    def _create_interactive_hierarchy(self, structure_data: Dict[str, Any], result: StructuredResult) -> go.Figure:
        """インタラクティブ階層構造作成"""
        fig = go.Figure()
        
        # サンバースト図の作成
        labels = []
        parents = []
        values = []
        colors = []
        
        # ルートノード
        labels.append(result.structure_type.value)
        parents.append("")
        values.append(len(result.parsed_elements))
        colors.append(self._get_structure_color(result.structure_type))
        
        # 要素ノード
        for element in result.parsed_elements:
            labels.append(element.element_type.value)
            parents.append(result.structure_type.value)
            values.append(1)
            colors.append(self._get_element_color(element.element_type))
        
        fig.add_trace(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Structure Hierarchy - {result.structure_type.value}",
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def _create_static_hierarchy(self, structure_data: Dict[str, Any], result: StructuredResult) -> plt.Figure:
        """静的階層構造作成"""
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        # 円グラフの作成
        element_types = structure_data["element_types"]
        if element_types:
            labels = list(element_types.keys())
            sizes = list(element_types.values())
            colors = [self._get_element_color(ParseResultType(label)) for label in labels]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            
            ax.set_title(f"Structure Distribution - {result.structure_type.value}")
        
        return fig
    
    def _create_interactive_heatmap(self, metrics_data: Dict[str, Any]) -> go.Figure:
        """インタラクティブヒートマップ作成"""
        # データの準備
        metrics = metrics_data["metrics"]
        scores = metrics_data["scores"]
        
        # ヒートマップデータの作成
        heatmap_data = np.array(scores).reshape(1, -1)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metrics,
            y=['Quality Scores'],
            colorscale='RdYlBu',
            showscale=True,
            hovertemplate='<b>%{x}</b><br>Score: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Quality Metrics Heatmap",
            xaxis_title="Metrics",
            yaxis_title="",
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def _create_static_heatmap(self, metrics_data: Dict[str, Any]) -> plt.Figure:
        """静的ヒートマップ作成"""
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        # データの準備
        metrics = metrics_data["metrics"]
        scores = metrics_data["scores"]
        
        # ヒートマップの作成
        heatmap_data = np.array(scores).reshape(1, -1)
        
        im = ax.imshow(heatmap_data, cmap='RdYlBu', aspect='auto')
        
        # カラーバーの追加
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Quality Score')
        
        # 軸の設定
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['Quality Scores'])
        
        # 値の表示
        for i in range(len(metrics)):
            ax.text(i, 0, f'{scores[i]:.3f}', ha='center', va='center', 
                   color='white' if scores[i] < 0.5 else 'black')
        
        ax.set_title("Quality Metrics Heatmap")
        
        return fig
    
    def _create_sentiment_chart(self, analysis_data: Dict[str, Any]) -> go.Figure:
        """感情分析チャート作成"""
        sentiment = analysis_data["sentiment"]
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Positive', 'Negative', 'Neutral'],
                y=[sentiment["positive"], sentiment["negative"], sentiment["neutral"]],
                marker_color=['green', 'red', 'gray']
            )
        ])
        
        fig.update_layout(
            title="Sentiment Analysis",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def _create_static_sentiment_chart(self, analysis_data: Dict[str, Any]) -> plt.Figure:
        """静的感情分析チャート作成"""
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        sentiment = analysis_data["sentiment"]
        
        categories = ['Positive', 'Negative', 'Neutral']
        values = [sentiment["positive"], sentiment["negative"], sentiment["neutral"]]
        colors = ['green', 'red', 'gray']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # 値の表示
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(value), ha='center', va='bottom')
        
        ax.set_title("Sentiment Analysis")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        
        return fig
    
    def _create_text_analysis_html(self, analysis_data: Dict[str, Any]) -> str:
        """テキスト分析HTML作成"""
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
            <h2>Text Analysis Results</h2>
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px;">
                <h3>Basic Statistics</h3>
                <ul>
                    <li>Text Length: {analysis_data['text_length']} characters</li>
                    <li>Word Count: {analysis_data['word_count']} words</li>
                    <li>Sentence Count: {analysis_data['sentence_count']} sentences</li>
                </ul>
            </div>
        </div>
        """
        return html
    
    def _calculate_flowchart_positions(self, steps: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """フローチャート位置計算"""
        positions = []
        num_steps = len(steps)
        
        if num_steps == 1:
            return [(0.5, 0.5)]
        
        # 水平配置
        for i in range(num_steps):
            x = i / (num_steps - 1) if num_steps > 1 else 0.5
            y = 0.5
            positions.append((x, y))
        
        return positions
    
    def _get_step_color(self, step_type: str) -> str:
        """ステップタイプの色取得"""
        color_map = {
            "thought": "#3498db",
            "action": "#e74c3c",
            "observation": "#f39c12",
            "conclusion": "#27ae60"
        }
        return color_map.get(step_type.lower(), "#95a5a6")
    
    def _get_structure_color(self, structure_type: ParseResultType) -> str:
        """構造タイプの色取得"""
        color_map = {
            ParseResultType.JSON: "#e74c3c",
            ParseResultType.YAML: "#3498db",
            ParseResultType.CODE: "#f39c12",
            ParseResultType.TABLE: "#27ae60",
            ParseResultType.LIST: "#9b59b6",
            ParseResultType.STRUCTURED: "#1abc9c",
            ParseResultType.TEXT: "#95a5a6"
        }
        return color_map.get(structure_type, "#95a5a6")
    
    def _get_element_color(self, element_type: ParseResultType) -> str:
        """要素タイプの色取得"""
        return self._get_structure_color(element_type)
    
    def _fig_to_html(self, fig: plt.Figure) -> str:
        """matplotlib図をHTMLに変換"""
        import io
        import base64
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        html = f'<img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto;">'
        
        plt.close(fig)
        return html
    
    def _update_visualization_stats(self, result: VisualizationResult, creation_time: float):
        """可視化統計更新"""
        self.visualization_stats["total_visualizations"] += 1
        
        # タイプ分布更新
        viz_type = result.visualization_type.value
        self.visualization_stats["type_distribution"][viz_type] = \
            self.visualization_stats["type_distribution"].get(viz_type, 0) + 1
        
        # 平均作成時間更新
        total_viz = self.visualization_stats["total_visualizations"]
        current_avg = self.visualization_stats["average_creation_time"]
        self.visualization_stats["average_creation_time"] = \
            (current_avg * (total_viz - 1) + creation_time) / total_viz
        
        # 作成時間記録
        self.visualization_stats["creation_times"].append(creation_time)
    
    def _create_error_visualization(self, error_message: str, start_time: float) -> VisualizationResult:
        """エラー可視化の作成"""
        import time
        creation_time = time.time() - start_time
        
        error_html = f"""
        <div style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
            <h2 style="color: #e74c3c;">Visualization Error</h2>
            <p style="color: #7f8c8d;">{error_message}</p>
        </div>
        """
        
        return VisualizationResult(
            visualization_type=VisualizationType.TEXT_ANALYSIS,
            title="Error",
            data={"error": error_message},
            config=self.config,
            html_content=error_html,
            metadata={"error": error_message, "creation_time": creation_time}
        )
    
    def get_visualization_statistics(self) -> Dict[str, Any]:
        """可視化統計取得"""
        import statistics
        
        return {
            "total_visualizations": self.visualization_stats["total_visualizations"],
            "type_distribution": self.visualization_stats["type_distribution"],
            "average_creation_time": self.visualization_stats["average_creation_time"],
            "creation_time_std": (
                statistics.stdev(self.visualization_stats["creation_times"])
                if len(self.visualization_stats["creation_times"]) > 1 else 0.0
            )
        }
    
    def reset_statistics(self):
        """統計リセット"""
        self.visualization_stats = {
            "total_visualizations": 0,
            "type_distribution": {},
            "average_creation_time": 0.0,
            "creation_times": []
        }


# 便利関数
def visualize_reasoning_response(response: CoTResponse, config: Optional[VisualizationConfig] = None) -> VisualizationResult:
    """推論レスポンス可視化（便利関数）"""
    visualizer = ReasoningVisualizer(config)
    return visualizer.visualize_reasoning_flow(response)


def create_visualization_dashboard(results: List[VisualizationResult]) -> str:
    """可視化ダッシュボード作成"""
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><title>Reasoning Visualization Dashboard</title></head><body>",
        "<h1>Reasoning Visualization Dashboard</h1>"
    ]
    
    for i, result in enumerate(results):
        html_parts.append(f"<h2>{i+1}. {result.title}</h2>")
        if result.html_content:
            html_parts.append(result.html_content)
        else:
            html_parts.append("<p>No visualization available</p>")
        html_parts.append("<hr>")
    
    html_parts.extend(["</body></html>"])
    
    return "\n".join(html_parts)


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
        response_text="この問題を段階的に解決します。答えは42です。",
        processing_time=5.0,
        reasoning_steps=test_steps,
        final_confidence=0.8,
        step_count=4,
        total_thinking_time=4.5,
        quality_score=0.7,
        model_used="qwen2:7b-instruct",
        state=ReasoningState.COMPLETED
    )
    
    # 可視化実行
    config = VisualizationConfig(width=800, height=600, interactive=True)
    visualizer = ReasoningVisualizer(config)
    
    # 推論フローの可視化
    flow_result = visualizer.visualize_reasoning_flow(test_response)
    print(f"Flow visualization created: {flow_result.title}")
    
    # テキスト分析の可視化
    text_result = visualizer.visualize_text_analysis(test_response.response_text, "sentiment")
    print(f"Text analysis created: {text_result.title}")
    
    # ダッシュボード作成
    dashboard = create_visualization_dashboard([flow_result, text_result])
    print("Dashboard created with 2 visualizations")
    
    # 統計表示
    stats = visualizer.get_visualization_statistics()
    print(f"Visualization statistics: {stats}")
