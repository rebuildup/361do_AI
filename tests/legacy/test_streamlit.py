#!/usr/bin/env python3
"""
テスト用のシンプルなStreamlitアプリ
"""

import streamlit as st
import asyncio
import time
import re

def generate_intelligent_response(user_input: str) -> str:
    """自然言語理解に基づく応答生成（推論部分付き）"""
    
    user_input_lower = user_input.lower()
    
    # 挨拶の応答
    if any(greeting in user_input_lower for greeting in ["こんにちは", "はじめまして", "よろしく", "hello", "hi"]):
        return """<think>
ユーザーから挨拶を受けました。友好的で親しみやすい応答を返し、自己学習型AIエージェントとしての能力を紹介する必要があります。
</think>
こんにちは！私は自己学習型AIエージェントです。自然言語で指示していただければ、必要なツールを自発的に使用してお手伝いします。"""
    
    # 情報要求の応答
    elif any(phrase in user_input_lower for phrase in ["教えて", "説明して", "について", "とは"]):
        # 最新情報が必要かどうかを判断
        if any(phrase in user_input_lower for phrase in ["最新", "現在", "リアルタイム", "ニュース"]):
            return """<think>
ユーザーは最新の情報を求めています。Web検索ツールを使用してリアルタイムの情報を取得する必要があります。
</think>
最新の情報が必要ですね。検索して詳細な情報を取得します。

🔍 **Web検索を実行中...**

最新の情報を調べて、詳細な回答を提供します。"""
        else:
            return f"""<think>
ユーザーは一般的な情報を求めています。既存の知識で回答し、必要に応じて追加の検索も提案します。
</think>
「{user_input}」について説明いたします。必要に応じて、詳細な情報を検索して補完いたします。"""
    
    # システム情報の要求
    elif any(phrase in user_input_lower for phrase in ["システム", "状態", "環境", "メモリ", "ディスク", "プロセス"]):
        return """<think>
ユーザーはシステム情報の確認を求めています。コマンド実行ツールを使用してシステムの状態を確認する必要があります。
</think>
システム情報の確認が必要ですね。システム情報を確認します。

💻 **システム情報を取得中...**

現在のシステム状態を確認して、詳細な情報を提供します。"""
    
    # ファイル操作の要求
    elif any(phrase in user_input_lower for phrase in ["ファイル", "設定", "コード", "プログラム", "スクリプト"]):
        return """<think>
ユーザーはファイル操作を求めています。ファイル管理ツールを使用してファイルの内容を確認し、必要な操作を実行する必要があります。
</think>
ファイル操作が必要ですね。ファイルを確認します。

📁 **ファイル操作を実行中...**

ファイルの内容を確認して、必要な操作を実行します。"""
    
    # 検索の要求
    elif any(phrase in user_input_lower for phrase in ["検索", "調べて", "探して", "情報を"]):
        return f"""<think>
ユーザーは検索を求めています。Web検索ツールを使用して関連する情報を検索し、包括的な回答を提供する必要があります。
</think>
「{user_input}」について検索します。検索して詳細な情報を取得します。

🔍 **Web検索を実行中...**

関連する情報を検索して、包括的な回答を提供します。"""
    
    # 機能についての質問
    elif any(phrase in user_input_lower for phrase in ["機能", "できること", "能力", "特徴"]):
        return """<think>
ユーザーは私の機能について質問しています。自己学習型AIエージェントとしての全ての機能を整理して説明する必要があります。
</think>
私には以下の機能があり、自然言語で指示していただければ自発的に使用します：

🤖 **自己学習機能**
- プロンプトの書き換えと最適化
- チューニングデータの動的操作
- 進化システムによる能力向上

🔍 **Web検索機能**
- 最新情報の取得
- リアルタイムデータの検索
- ニュースや記事の収集

💻 **システム操作機能**
- コマンドの実行
- システム情報の確認
- 環境状態の監視

📁 **ファイル操作機能**
- ファイルの読み書き
- 設定ファイルの管理
- コードの生成と編集

🔗 **MCP連携機能**
- 外部ツールとの連携
- API連携
- 外部サービスとの統合

これらの機能について詳しく知りたい場合は、具体的に質問してください。"""
    
    # デフォルトの応答
    else:
        return f"""<think>
ユーザーの要求を分析し、適切なツールを使用して対応する必要があります。自然言語で指示された内容を理解し、最適な解決策を提供します。
</think>
「{user_input}」について理解しました。

私の能力を活用して、最適な解決策を提供いたします。必要に応じて、以下のツールを自発的に使用します：

- 🔍 **Web検索**: 最新情報の取得
- 💻 **システム操作**: コマンドの実行
- 📁 **ファイル操作**: ファイルの管理
- 🔗 **MCP連携**: 外部ツールとの連携

具体的な指示をいただければ、適切なツールを使用して対応いたします。"""

# 推論部分の解析関数
def parse_reasoning_content(text: str) -> tuple[str, str]:
    """テキストから推論部分と通常の応答を分離"""
    # <think>タグで囲まれた推論部分を検出
    think_pattern = r'<think>(.*?)</think>'
    reasoning_matches = re.findall(think_pattern, text, re.DOTALL)
    
    # 推論部分を除去した通常の応答
    clean_text = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
    
    # 推論部分を結合
    reasoning_text = '\n\n'.join(reasoning_matches) if reasoning_matches else ""
    
    return reasoning_text, clean_text

# ストリーミング出力関数
def stream_text(text: str, placeholder):
    """テキストを1文字ずつ表示"""
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(f'<div style="animation: typing 0.05s linear;">{display_text}</div>', unsafe_allow_html=True)
        time.sleep(0.02)  # ストリーミング効果

st.set_page_config(
    page_title="テスト用AIエージェント",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 テスト用AIエージェント")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 推論部分の解析と表示
        if message["role"] == "assistant":
            reasoning_text, clean_response = parse_reasoning_content(message["content"])
            
            # 推論部分がある場合は表示
            if reasoning_text:
                with st.expander("🧠 エージェントの思考過程", expanded=False):
                    st.markdown(f'<div style="background-color: rgba(240, 248, 255, 0.8); border-left: 4px solid #1f77b4; border-radius: 4px; padding: 12px; margin: 8px 0; font-style: italic; color: #2c3e50;">{reasoning_text}</div>', unsafe_allow_html=True)
            
            # 通常の応答を表示
            st.markdown(clean_response if clean_response else message["content"])
        else:
            st.markdown(message["content"])

# チャット入力
user_input = st.chat_input("メッセージを入力してください...")

if user_input:
    # ユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # AI応答を生成
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 自然言語理解に基づく応答生成
            response = generate_intelligent_response(user_input)
            
            # 推論部分の解析
            reasoning_text, clean_response = parse_reasoning_content(response)
            
            # 推論部分がある場合は表示
            if reasoning_text:
                with st.expander("🧠 エージェントの思考過程", expanded=True):
                    st.markdown(f'<div style="background-color: rgba(240, 248, 255, 0.8); border-left: 4px solid #1f77b4; border-radius: 4px; padding: 12px; margin: 8px 0; font-style: italic; color: #2c3e50;">{reasoning_text}</div>', unsafe_allow_html=True)
            
            # ストリーミング出力で応答を表示
            response_placeholder = st.empty()
            if st.session_state.streaming_enabled:
                stream_text(clean_response if clean_response else response, response_placeholder)
            else:
                response_placeholder.markdown(clean_response if clean_response else response)
            
            # メッセージを履歴に追加
            st.session_state.messages.append({"role": "assistant", "content": response})

# サイドバーに情報を表示
with st.sidebar:
    st.header("🤖 AIエージェント情報")
    st.info("テスト用AIエージェントが稼働中です")
    
    # ストリーミング設定
    st.subheader("⚙️ 設定")
    st.session_state.streaming_enabled = st.checkbox("ストリーミング出力", value=st.session_state.streaming_enabled)
    
    # セッション情報
    st.subheader("📊 セッション情報")
    st.info(f"メッセージ数: {len(st.session_state.messages)}")
    
    st.subheader("実装済み機能")
    st.success("✅ 自然言語理解")
    st.success("✅ 自己学習機能") 
    st.success("✅ Web検索機能")
    st.success("✅ コマンド実行機能")
    st.success("✅ ファイル操作機能")
    st.success("✅ MCP使用機能")
    st.success("✅ 進化システム")
    st.success("✅ 報酬システム")
    st.success("✅ 永続セッション")
    st.success("✅ 推論能力")
    st.success("✅ ストリーミング出力")
    st.success("✅ 思考過程表示")
    
    st.subheader("テスト方法")
    st.markdown("""
    自然言語で指示してください：
    - こんにちは
    - 最新のニュースを教えて
    - システムの状態を確認して
    - ファイルを確認して
    - 機能について教えて
    """)