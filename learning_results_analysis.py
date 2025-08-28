#!/usr/bin/env python3
"""
Multi-Agent Learning Results Analysis
8時間マルチエージェント学習結果の詳細分析
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

def analyze_learning_session(json_file):
    """学習セッション結果の詳細分析"""
    
    print("=" * 80)
    print("🎉 8時間マルチエージェント学習システム 結果分析")
    print("=" * 80)
    
    # JSONファイル読み込み
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ファイル読み込みエラー: {e}")
        return
    
    # 基本情報分析
    session_info = data.get('session_info', {})
    learning_stats = data.get('learning_stats', {})
    agent_roles = data.get('agent_roles', {})
    conversation_history = data.get('conversation_history', [])
    
    print(f"📅 実行期間:")
    print(f"  開始時刻: {session_info.get('start_time', 'Unknown')}")
    print(f"  終了時刻: {session_info.get('end_time', 'Unknown')}")
    print(f"  総実行時間: {session_info.get('total_runtime_hours', 0):.2f}時間")
    print(f"  制限時間: {session_info.get('time_limit_hours', 0)}時間")
    
    # 学習統計分析
    print(f"\n📊 学習統計:")
    print(f"  総会話数: {learning_stats.get('total_conversations', 0)}")
    print(f"  学習サイクル数: {learning_stats.get('total_learning_cycles', 0)}")
    print(f"  知識共有数: {learning_stats.get('knowledge_shared', 0)}")
    print(f"  改善実施数: {learning_stats.get('improvements_made', 0)}")
    
    # エージェント別パフォーマンス分析
    print(f"\n🤖 エージェント別パフォーマンス:")
    agent_interactions = learning_stats.get('agent_interactions', {})
    
    for agent_id, stats in agent_interactions.items():
        agent_name = agent_roles.get(agent_id, {}).get('name', agent_id)
        focus = agent_roles.get(agent_id, {}).get('focus', 'Unknown')
        
        print(f"  {agent_name} ({agent_id}):")
        print(f"    専門分野: {focus}")
        print(f"    発言数: {stats.get('messages_sent', 0)}")
        print(f"    学習サイクル: {stats.get('learning_cycles', 0)}")
        print(f"    知識貢献: {stats.get('knowledge_contributions', 0)}")
    
    # 会話内容分析
    print(f"\n💬 会話内容分析:")
    if conversation_history:
        print(f"  記録された会話数: {len(conversation_history)}")
        
        # エージェント別発言数
        agent_message_counts = Counter()
        total_execution_time = 0
        successful_conversations = 0
        
        for conv in conversation_history:
            agent_name = conv.get('agent_name', 'Unknown')
            agent_message_counts[agent_name] += 1
            total_execution_time += conv.get('execution_time', 0)
            if conv.get('success', False):
                successful_conversations += 1
        
        print(f"  成功した会話: {successful_conversations}/{len(conversation_history)} ({successful_conversations/len(conversation_history)*100:.1f}%)")
        print(f"  平均応答時間: {total_execution_time/len(conversation_history):.2f}秒")
        
        print(f"\n  エージェント別発言数:")
        for agent_name, count in agent_message_counts.most_common():
            print(f"    {agent_name}: {count}回")
        
        # 使用されたツール分析
        all_tools = []
        intent_analysis = Counter()
        
        for conv in conversation_history:
            tools_used = conv.get('tools_used', [])
            all_tools.extend(tools_used)
            
            intent = conv.get('intent', {})
            primary_intent = intent.get('primary_intent', 'unknown')
            intent_analysis[primary_intent] += 1
        
        if all_tools:
            print(f"\n  使用ツール統計:")
            tool_counts = Counter(all_tools)
            for tool, count in tool_counts.most_common():
                print(f"    {tool}: {count}回")
        
        print(f"\n  会話意図分析:")
        for intent, count in intent_analysis.most_common():
            print(f"    {intent}: {count}回 ({count/len(conversation_history)*100:.1f}%)")
    
    # 学習効果分析
    print(f"\n📈 学習効果分析:")
    
    # 時間あたりの効率性
    runtime_hours = session_info.get('total_runtime_hours', 1)
    conversations_per_hour = learning_stats.get('total_conversations', 0) / runtime_hours
    learning_cycles_per_hour = learning_stats.get('total_learning_cycles', 0) / runtime_hours
    knowledge_per_hour = learning_stats.get('knowledge_shared', 0) / runtime_hours
    
    print(f"  時間あたり会話数: {conversations_per_hour:.1f}回/時間")
    print(f"  時間あたり学習サイクル: {learning_cycles_per_hour:.1f}回/時間")
    print(f"  時間あたり知識共有: {knowledge_per_hour:.1f}件/時間")
    
    # エージェント間の協調性分析
    total_messages = sum(stats.get('messages_sent', 0) for stats in agent_interactions.values())
    total_contributions = sum(stats.get('knowledge_contributions', 0) for stats in agent_interactions.values())
    
    if total_messages > 0:
        print(f"\n🤝 エージェント間協調性:")
        print(f"  総発言数: {total_messages}")
        print(f"  総知識貢献: {total_contributions}")
        print(f"  発言あたり知識貢献率: {total_contributions/total_messages:.2f}")
        
        # バランス分析
        message_distribution = [stats.get('messages_sent', 0) for stats in agent_interactions.values()]
        max_messages = max(message_distribution)
        min_messages = min(message_distribution)
        balance_ratio = min_messages / max_messages if max_messages > 0 else 0
        
        print(f"  発言バランス比: {balance_ratio:.2f} (1.0が完全バランス)")
    
    # 会話トピック分析
    if conversation_history:
        print(f"\n📝 会話トピック分析:")
        
        # 会話の長さ分析
        response_lengths = []
        for conv in conversation_history:
            content = conv.get('content', '')
            response_lengths.append(len(content))
        
        if response_lengths:
            avg_length = sum(response_lengths) / len(response_lengths)
            max_length = max(response_lengths)
            min_length = min(response_lengths)
            
            print(f"  平均応答長: {avg_length:.0f}文字")
            print(f"  最長応答: {max_length}文字")
            print(f"  最短応答: {min_length}文字")
    
    # 学習データベース分析（可能な場合）
    try:
        analyze_database_impact()
    except Exception as e:
        print(f"\n⚠️ データベース分析をスキップ: {e}")
    
    # 推奨事項
    print(f"\n💡 学習効果と推奨事項:")
    
    success_rate = successful_conversations / len(conversation_history) * 100 if conversation_history else 0
    
    if success_rate >= 95:
        print(f"  ✅ 優秀: 会話成功率 {success_rate:.1f}% - システムは非常に安定して動作")
    elif success_rate >= 80:
        print(f"  ✅ 良好: 会話成功率 {success_rate:.1f}% - システムは安定して動作")
    else:
        print(f"  ⚠️ 改善必要: 会話成功率 {success_rate:.1f}% - システムの安定性向上が必要")
    
    if balance_ratio >= 0.8:
        print(f"  ✅ エージェント間のバランスが良好 (比率: {balance_ratio:.2f})")
    else:
        print(f"  ⚠️ エージェント間の発言バランスに偏りあり (比率: {balance_ratio:.2f})")
    
    if conversations_per_hour >= 1.0:
        print(f"  ✅ 効率的な学習ペース ({conversations_per_hour:.1f}会話/時間)")
    else:
        print(f"  ⚠️ 学習ペースが低い可能性 ({conversations_per_hour:.1f}会話/時間)")
    
    # 総合評価
    print(f"\n🏆 総合評価:")
    
    score = 0
    max_score = 100
    
    # 実行時間達成度 (25点)
    time_achievement = min(runtime_hours / session_info.get('time_limit_hours', 8), 1.0)
    time_score = time_achievement * 25
    score += time_score
    print(f"  実行時間達成度: {time_achievement*100:.1f}% ({time_score:.1f}/25点)")
    
    # 会話成功率 (25点)
    success_score = (success_rate / 100) * 25
    score += success_score
    print(f"  会話成功率: {success_rate:.1f}% ({success_score:.1f}/25点)")
    
    # エージェント協調性 (25点)
    balance_score = balance_ratio * 25
    score += balance_score
    print(f"  エージェント協調性: {balance_ratio*100:.1f}% ({balance_score:.1f}/25点)")
    
    # 学習効率性 (25点)
    efficiency_score = min(conversations_per_hour / 2.0, 1.0) * 25  # 2会話/時間を満点とする
    score += efficiency_score
    print(f"  学習効率性: {min(conversations_per_hour/2.0*100, 100):.1f}% ({efficiency_score:.1f}/25点)")
    
    print(f"\n  🎯 総合スコア: {score:.1f}/{max_score}点")
    
    if score >= 90:
        grade = "S (優秀)"
    elif score >= 80:
        grade = "A (良好)"
    elif score >= 70:
        grade = "B (普通)"
    elif score >= 60:
        grade = "C (改善必要)"
    else:
        grade = "D (大幅改善必要)"
    
    print(f"  📊 評価: {grade}")
    
    print("=" * 80)

def analyze_database_impact():
    """データベースへの学習効果分析"""
    print(f"\n🗄️ データベース学習効果分析:")
    
    try:
        # データベース統計の簡易確認
        import sqlite3
        
        db_path = "data/agent.db"
        if Path(db_path).exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 学習データ数確認
            cursor.execute("SELECT COUNT(*) FROM learning_data")
            learning_data_count = cursor.fetchone()[0]
            
            # 知識アイテム数確認
            cursor.execute("SELECT COUNT(*) FROM knowledge_items")
            knowledge_count = cursor.fetchone()[0]
            
            print(f"  学習データ総数: {learning_data_count}件")
            print(f"  知識アイテム数: {knowledge_count}件")
            
            # 最近追加されたデータ
            cursor.execute("""
                SELECT COUNT(*) FROM learning_data 
                WHERE created_at > datetime('now', '-1 day')
            """)
            recent_data = cursor.fetchone()[0]
            
            print(f"  過去24時間の新規学習データ: {recent_data}件")
            
            conn.close()
        else:
            print(f"  データベースファイルが見つかりません: {db_path}")
            
    except Exception as e:
        print(f"  データベース分析エラー: {e}")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="マルチエージェント学習結果分析")
    parser.add_argument("--file", help="分析するJSONファイル")
    
    args = parser.parse_args()
    
    # 最新の結果ファイルを自動検出
    if not args.file:
        json_files = list(Path('.').glob('multi_agent_learning_session_*.json'))
        if json_files:
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            args.file = str(latest_file)
            print(f"最新の結果ファイルを使用: {args.file}")
        else:
            print("結果ファイルが見つかりません")
            return
    
    # 分析実行
    analyze_learning_session(args.file)

if __name__ == "__main__":
    main()