import React, { useState } from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  Edit3,
  Check,
  X,
  Clock,
  User,
  MoreVertical,
  AlertCircle,
} from 'lucide-react';
import { cn, formatRelativeTime } from '@/utils';
import { useApp } from '@/contexts/AppContext';
import { useSessionManager } from '@/hooks/useSessionManager';

interface SessionManagerProps {
  className?: string;
}

const SessionManager: React.FC<SessionManagerProps> = ({ className }) => {
  const { state } = useApp();
  const {
    sessions,
    isLoading,
    error,
    createSession,
    switchSession,
    deleteSession,
    renameSession,
    clearAllSessions,
  } = useSessionManager();

  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [newSessionName, setNewSessionName] = useState('');
  const [showNewSessionForm, setShowNewSessionForm] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);

  const handleCreateSession = async () => {
    if (!newSessionName.trim() && !showNewSessionForm) {
      setShowNewSessionForm(true);
      return;
    }

    const sessionName = newSessionName.trim() || undefined;
    await createSession(sessionName);
    setNewSessionName('');
    setShowNewSessionForm(false);
  };

  const handleRenameStart = (sessionId: string, currentName: string) => {
    setEditingSessionId(sessionId);
    setEditingName(currentName);
    setActiveDropdown(null);
  };

  const handleRenameConfirm = async () => {
    if (editingSessionId && editingName.trim()) {
      await renameSession(editingSessionId, editingName.trim());
    }
    setEditingSessionId(null);
    setEditingName('');
  };

  const handleRenameCancel = () => {
    setEditingSessionId(null);
    setEditingName('');
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (
      window.confirm('このセッションを削除しますか？この操作は取り消せません。')
    ) {
      await deleteSession(sessionId);
    }
    setActiveDropdown(null);
  };

  const handleClearAllSessions = async () => {
    if (
      window.confirm(
        'すべてのセッションを削除しますか？この操作は取り消せません。'
      )
    ) {
      await clearAllSessions();
    }
  };

  const getSessionPreview = (messages: any[]) => {
    if (messages.length === 0) return '新しいセッション';
    const lastUserMessage = messages.filter(m => m.role === 'user').pop();
    return (
      lastUserMessage?.content?.substring(0, 50) + '...' || '新しいセッション'
    );
  };

  return (
    <div
      className={cn('bg-gray-900 border border-gray-800 rounded-lg', className)}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-medium flex items-center gap-2">
            <MessageSquare size={20} />
            セッション管理
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCreateSession}
              disabled={isLoading}
              className="flex items-center gap-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded transition-colors disabled:opacity-50"
            >
              <Plus size={16} />
              新規
            </button>
            {sessions.length > 1 && (
              <button
                onClick={handleClearAllSessions}
                className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                title="すべてクリア"
              >
                <Trash2 size={16} />
              </button>
            )}
          </div>
        </div>

        {/* New Session Form */}
        {showNewSessionForm && (
          <div className="mt-3 flex gap-2">
            <input
              type="text"
              value={newSessionName}
              onChange={e => setNewSessionName(e.target.value)}
              placeholder="セッション名（オプション）"
              className="flex-1 bg-gray-800 border border-gray-700 text-white text-sm rounded px-2 py-1 focus:outline-none focus:border-gray-600"
              onKeyDown={e => {
                if (e.key === 'Enter') handleCreateSession();
                if (e.key === 'Escape') setShowNewSessionForm(false);
              }}
              autoFocus
            />
            <button
              onClick={handleCreateSession}
              className="p-1 text-green-400 hover:text-green-300"
            >
              <Check size={16} />
            </button>
            <button
              onClick={() => setShowNewSessionForm(false)}
              className="p-1 text-gray-400 hover:text-gray-300"
            >
              <X size={16} />
            </button>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-3 bg-red-900 border-b border-red-700 text-red-100 text-sm flex items-center gap-2">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Sessions List */}
      <div className="max-h-96 overflow-y-auto">
        {sessions.length === 0 ? (
          <div className="p-8 text-center text-gray-400">
            <MessageSquare size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-sm">セッションがありません</p>
            <p className="text-xs mt-1">新しいセッションを作成してください</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {sessions
              .sort(
                (a, b) => b.lastActivity.getTime() - a.lastActivity.getTime()
              )
              .map(sessionData => {
                const isActive =
                  state.currentSession?.session_id ===
                  sessionData.session.session_id;
                const isEditing =
                  editingSessionId === sessionData.session.session_id;

                return (
                  <div
                    key={sessionData.session.session_id}
                    className={cn(
                      'p-3 hover:bg-gray-800 transition-colors cursor-pointer relative',
                      isActive && 'bg-gray-800 border-l-4 border-blue-500'
                    )}
                    onClick={() =>
                      !isEditing &&
                      switchSession(sessionData.session.session_id)
                    }
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        {isEditing ? (
                          <div className="flex gap-2 mb-2">
                            <input
                              type="text"
                              value={editingName}
                              onChange={e => setEditingName(e.target.value)}
                              className="flex-1 bg-gray-700 border border-gray-600 text-white text-sm rounded px-2 py-1 focus:outline-none focus:border-gray-500"
                              onKeyDown={e => {
                                if (e.key === 'Enter') handleRenameConfirm();
                                if (e.key === 'Escape') handleRenameCancel();
                              }}
                              autoFocus
                              onClick={e => e.stopPropagation()}
                            />
                            <button
                              onClick={e => {
                                e.stopPropagation();
                                handleRenameConfirm();
                              }}
                              className="p-1 text-green-400 hover:text-green-300"
                            >
                              <Check size={16} />
                            </button>
                            <button
                              onClick={e => {
                                e.stopPropagation();
                                handleRenameCancel();
                              }}
                              className="p-1 text-gray-400 hover:text-gray-300"
                            >
                              <X size={16} />
                            </button>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 mb-1">
                            <h4
                              className={cn(
                                'font-medium truncate',
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
                        )}

                        <p className="text-sm text-gray-400 truncate mb-2">
                          {getSessionPreview(sessionData.messages)}
                        </p>

                        <div className="flex items-center gap-4 text-xs text-gray-500">
                          <div className="flex items-center gap-1">
                            <Clock size={12} />
                            <span>
                              {formatRelativeTime(sessionData.lastActivity)}
                            </span>
                          </div>
                          <div className="flex items-center gap-1">
                            <MessageSquare size={12} />
                            <span>{sessionData.messageCount}件</span>
                          </div>
                          {sessionData.session.user_id && (
                            <div className="flex items-center gap-1">
                              <User size={12} />
                              <span>{sessionData.session.user_id}</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {!isEditing && (
                        <div className="relative">
                          <button
                            onClick={e => {
                              e.stopPropagation();
                              setActiveDropdown(
                                activeDropdown ===
                                  sessionData.session.session_id
                                  ? null
                                  : sessionData.session.session_id
                              );
                            }}
                            className="p-1 text-gray-400 hover:text-gray-300 rounded"
                          >
                            <MoreVertical size={16} />
                          </button>

                          {activeDropdown ===
                            sessionData.session.session_id && (
                            <>
                              <div
                                className="fixed inset-0 z-10"
                                onClick={() => setActiveDropdown(null)}
                              />
                              <div className="absolute right-0 top-8 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-20 min-w-32">
                                <button
                                  onClick={e => {
                                    e.stopPropagation();
                                    handleRenameStart(
                                      sessionData.session.session_id,
                                      sessionData.session.session_name || ''
                                    );
                                  }}
                                  className="w-full px-3 py-2 text-left text-sm text-gray-300 hover:bg-gray-700 flex items-center gap-2"
                                >
                                  <Edit3 size={14} />
                                  名前を変更
                                </button>
                                <button
                                  onClick={e => {
                                    e.stopPropagation();
                                    handleDeleteSession(
                                      sessionData.session.session_id
                                    );
                                  }}
                                  className="w-full px-3 py-2 text-left text-sm text-red-400 hover:bg-gray-700 flex items-center gap-2"
                                >
                                  <Trash2 size={14} />
                                  削除
                                </button>
                              </div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
          </div>
        )}
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="p-4 text-center">
          <div className="inline-flex items-center gap-2 text-gray-400 text-sm">
            <div className="w-4 h-4 border-2 border-gray-600 border-t-gray-400 rounded-full animate-spin" />
            処理中...
          </div>
        </div>
      )}
    </div>
  );
};

export default SessionManager;
