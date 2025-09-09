import React, { useState, useMemo } from 'react';
import {
  Search,
  Calendar,
  MessageSquare,
  Clock,
  Filter,
  SortAsc,
  SortDesc,
  Archive,
  Download,
} from 'lucide-react';
import { cn, formatDate } from '@/utils';
import { useSessionManager } from '@/hooks/useSessionManager';
import { useApp } from '@/contexts/AppContext';

interface SessionHistoryProps {
  className?: string;
}

type SortOption = 'date' | 'name' | 'messages';
type SortDirection = 'asc' | 'desc';

const SessionHistory: React.FC<SessionHistoryProps> = ({ className }) => {
  const { state } = useApp();
  const { sessions, switchSession, deleteSession } = useSessionManager();

  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortOption>('date');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [dateFilter, setDateFilter] = useState<
    'all' | 'today' | 'week' | 'month'
  >('all');
  const [showFilters, setShowFilters] = useState(false);

  // Filter and sort sessions
  const filteredAndSortedSessions = useMemo(() => {
    let filtered = sessions;

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        sessionData =>
          sessionData.session.session_name?.toLowerCase().includes(query) ||
          sessionData.messages.some(msg => {
            const text =
              typeof (msg as any).content === 'string'
                ? (msg as any).content
                : '';
            return text.toLowerCase().includes(query);
          })
      );
    }

    // Apply date filter
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
    const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

    switch (dateFilter) {
      case 'today':
        filtered = filtered.filter(s => s.lastActivity >= today);
        break;
      case 'week':
        filtered = filtered.filter(s => s.lastActivity >= weekAgo);
        break;
      case 'month':
        filtered = filtered.filter(s => s.lastActivity >= monthAgo);
        break;
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'date':
          comparison = a.lastActivity.getTime() - b.lastActivity.getTime();
          break;
        case 'name': {
          const nameA = a.session.session_name || '';
          const nameB = b.session.session_name || '';
          comparison = nameA.localeCompare(nameB);
          break;
        }
        case 'messages':
          comparison = a.messageCount - b.messageCount;
          break;
      }

      return sortDirection === 'desc' ? -comparison : comparison;
    });

    return filtered;
  }, [sessions, searchQuery, sortBy, sortDirection, dateFilter]);

  const handleSort = (option: SortOption) => {
    if (sortBy === option) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(option);
      setSortDirection('desc');
    }
  };

  const exportSession = async (sessionId: string) => {
    const sessionData = sessions.find(s => s.session.session_id === sessionId);
    if (!sessionData) return;

    const exportData = {
      session: sessionData.session,
      messages: sessionData.messages,
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session-${sessionData.session.session_name || sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  type Msg = { role: 'user' | 'assistant'; content?: string };
  const getSessionSummary = (messages: Msg[]) => {
    const userMessages = messages.filter(m => m.role === 'user');
    const assistantMessages = messages.filter(m => m.role === 'assistant');

    const truncate = (s?: string, n = 100) =>
      s ? (s.length > n ? `${s.slice(0, n)}...` : s) : '';

    return {
      userMessages: userMessages.length,
      assistantMessages: assistantMessages.length,
      totalMessages: messages.length,
      firstMessage: truncate(messages[0]?.content),
      lastMessage: truncate(messages[messages.length - 1]?.content),
    };
  };

  return (
    <div
      className={cn('bg-gray-900 border border-gray-800 rounded-lg', className)}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-medium flex items-center gap-2">
            <Archive size={20} />
            セッション履歴
          </h3>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              'p-2 rounded-lg transition-colors',
              showFilters
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            )}
          >
            <Filter size={16} />
          </button>
        </div>

        {/* Search */}
        <div className="relative mb-4">
          <Search
            size={16}
            className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
          />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="セッションを検索..."
            className="w-full bg-gray-800 border border-gray-700 text-white pl-10 pr-4 py-2 rounded-lg focus:outline-none focus:border-gray-600"
          />
        </div>

        {/* Filters */}
        {showFilters && (
          <div className="space-y-3 p-3 bg-gray-800 rounded-lg">
            {/* Date Filter */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">期間</label>
              <div className="flex gap-2">
                {[
                  { value: 'all', label: 'すべて' },
                  { value: 'today', label: '今日' },
                  { value: 'week', label: '1週間' },
                  { value: 'month', label: '1ヶ月' },
                ].map(option => (
                  <button
                    key={option.value}
                    onClick={() => setDateFilter(option.value as any)}
                    className={cn(
                      'px-3 py-1 text-sm rounded transition-colors',
                      dateFilter === option.value
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    )}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Sort Options */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">並び順</label>
              <div className="flex gap-2">
                {[
                  { value: 'date', label: '日時', icon: Calendar },
                  { value: 'name', label: '名前', icon: MessageSquare },
                  {
                    value: 'messages',
                    label: 'メッセージ数',
                    icon: MessageSquare,
                  },
                ].map(option => (
                  <button
                    key={option.value}
                    onClick={() => handleSort(option.value as SortOption)}
                    className={cn(
                      'flex items-center gap-2 px-3 py-1 text-sm rounded transition-colors',
                      sortBy === option.value
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    )}
                  >
                    <option.icon size={14} />
                    {option.label}
                    {sortBy === option.value &&
                      (sortDirection === 'desc' ? (
                        <SortDesc size={14} />
                      ) : (
                        <SortAsc size={14} />
                      ))}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sessions List */}
      <div className="max-h-96 overflow-y-auto">
        {filteredAndSortedSessions.length === 0 ? (
          <div className="p-8 text-center text-gray-400">
            <Archive size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-sm">
              {searchQuery
                ? '検索結果が見つかりません'
                : 'セッション履歴がありません'}
            </p>
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {filteredAndSortedSessions.map(sessionData => {
              const isActive =
                state.currentSession?.session_id ===
                sessionData.session.session_id;
              const summary = getSessionSummary(sessionData.messages);

              return (
                <div
                  key={sessionData.session.session_id}
                  className={cn(
                    'p-4 hover:bg-gray-800 transition-colors',
                    isActive && 'bg-gray-800 border-l-4 border-blue-500'
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div
                      className="flex-1 cursor-pointer"
                      onClick={() =>
                        switchSession(sessionData.session.session_id)
                      }
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <h4
                          className={cn(
                            'font-medium',
                            isActive ? 'text-white' : 'text-gray-300'
                          )}
                        >
                          {sessionData.session.session_name ||
                            'Untitled Session'}
                        </h4>
                        {isActive && (
                          <span className="px-2 py-0.5 bg-blue-600 text-blue-100 text-xs rounded-full">
                            アクティブ
                          </span>
                        )}
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm text-gray-400 mb-2">
                        <div className="flex items-center gap-1">
                          <Clock size={12} />
                          <span>{formatDate(sessionData.lastActivity)}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <MessageSquare size={12} />
                          <span>{summary.totalMessages}件のメッセージ</span>
                        </div>
                      </div>

                      {summary.firstMessage && (
                        <div className="text-xs text-gray-500 mb-1">
                          <span className="font-medium">最初:</span>{' '}
                          {summary.firstMessage}
                        </div>
                      )}

                      {summary.lastMessage && summary.totalMessages > 1 && (
                        <div className="text-xs text-gray-500">
                          <span className="font-medium">最新:</span>{' '}
                          {summary.lastMessage}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-2 ml-4">
                      <button
                        onClick={() =>
                          exportSession(sessionData.session.session_id)
                        }
                        className="p-2 text-gray-400 hover:text-blue-400 transition-colors"
                        title="エクスポート"
                      >
                        <Download size={16} />
                      </button>
                      <button
                        onClick={() =>
                          deleteSession(sessionData.session.session_id)
                        }
                        className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                        title="削除"
                      >
                        <Archive size={16} />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Summary */}
      <div className="p-4 border-t border-gray-800 bg-gray-800/50">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <span>
            {filteredAndSortedSessions.length} / {sessions.length} セッション
          </span>
          <span>
            合計 {sessions.reduce((sum, s) => sum + s.messageCount, 0)}{' '}
            メッセージ
          </span>
        </div>
      </div>
    </div>
  );
};

export default SessionHistory;
