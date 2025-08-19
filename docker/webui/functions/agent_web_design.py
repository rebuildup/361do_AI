"""
title: AI Agent Web Design
author: AI Agent Studio
version: 1.0.0
description: AI エージェントを使用したWebデザイン生成機能
"""

import json
import requests
from typing import Dict, Any
from pydantic import BaseModel, Field


class WebDesignRequest(BaseModel):
    requirements: str = Field(..., description="Webサイトの要件")
    style: str = Field("modern", description="デザインスタイル")
    colors: str = Field("", description="カラーパレット")


class Tools:
    def __init__(self):
        self.agent_base_url = "http://agent:8000"
    
    def generate_web_design(self, requirements: str, style: str = "modern", colors: str = "") -> str:
        """
        AI エージェントを使用してWebデザインを生成
        
        Args:
            requirements: Webサイトの要件
            style: デザインスタイル
            colors: カラーパレット
            
        Returns:
            生成されたWebデザインのHTMLとCSS
        """
        try:
            # デザイン生成リクエストを構築
            design_prompt = f"""
Webデザイン生成:
要件: {requirements}
スタイル: {style}
カラー: {colors}

HTML、CSS、JavaScriptを含む完全なWebページを生成してください。
レスポンシブデザインで、モダンなUIを心がけてください。
"""
            
            # エージェントAPIにリクエストを送信
            response = requests.post(
                f"{self.agent_base_url}/api/web-design/generate",
                json={
                    "requirements": design_prompt
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # デザイン結果をフォーマット
                if "html" in result and "css" in result:
                    return f"""
## 生成されたWebデザイン

### HTML
```html
{result.get('html', '')}
```

### CSS  
```css
{result.get('css', '')}
```

### JavaScript
```javascript
{result.get('javascript', '')}
```

### プレビュー
{result.get('preview_url', 'プレビューは利用できません')}
"""
                else:
                    return result.get("response", "Webデザインを生成できませんでした。")
            else:
                return f"Webデザイン生成エラー: HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Webデザイン生成エラー: {str(e)}"
        except Exception as e:
            return f"予期しないエラー: {str(e)}"


# Tools インスタンスを作成
tools = Tools()
