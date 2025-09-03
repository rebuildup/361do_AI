"""
Advanced Self-Learning AI Agent
オープンソースライブラリを最大限活用した高性能自己学習AIエージェント
"""

__version__ = "0.1.0"
__author__ = "Advanced Agent Team"
__description__ = "RTX 4050 6GB VRAM 最適化自己学習AIエージェント"

# 主要コンポーネントのインポート
from .core import *
from .memory import *
from .learning import *
from .monitoring import *
# from .interfaces import *

__all__ = [
    "__version__",
    "__author__", 
    "__description__"
]