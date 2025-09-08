import { useState, useEffect, useCallback } from 'react';
import { generateId, storage } from '@/utils';
import { buildApiUrl } from '@/services/api';
import { useApp, useAppActions } from '@/contexts/AppContext';
import { sessionPersistence } from '@/services/sessionPersistence';
import type { SessionResponse, UIMessage } from '@/types';

interface SessionData {
  session: SessionResponse;
  messages: UIMessage[];
  lastActivity: Date;
  messageCount: number;
}

interface SessionManagerState {
  sessions: SessionData[];
  isLoading: boolean;
  error: string | null;
}

export const useSessionManager = () => {
  const { state } = useApp();
  const { setCurrentSession, setMessages } = useAppActions();

  const [sessionState, setSessionState] = useState<SessionManagerState>({
    sessions: [],
    isLoading: false,
    error: null,
  });

  // Load sessions from persistent storage on mount
  useEffect(() => {
    loadSessionsFromStorage();
  }, []);

  // Save sessions to localStorage whenever sessions change
  useEffect(() => {
    if (sessionState.sessions.length > 0) {
      saveSessionsToStorage();
    }
  }, [sessionState.sessions]);

  const loadSessionsFromStorage = async () => {
    try {
      setSessionState(prev => ({ ...prev, isLoading: true }));

      // Get or create persistent session
      const persistentSession =
        await sessionPersistence.createOrRestoreSession('default_user');

      // Convert to SessionData format for compatibility
      const sessionData: SessionData = {
        session: {
          session_id: persistentSession.sessionId,
          created_at: persistentSession.metadata.createdAt,
          user_id: persistentSession.userId,
          session_name: persistentSession.sessionName,
          metadata: persistentSession.metadata,
        },
        messages: persistentSession.messages,
        lastActivity: new Date(persistentSession.metadata.lastActivity),
        messageCount: persistentSession.metadata.messageCount,
      };

      setSessionState(prev => ({
        ...prev,
        sessions: [sessionData],
        isLoading: false,
      }));

      // Set as current session
      setCurrentSession(sessionData.session);
      setMessages(persistentSession.messages);
    } catch (error) {
      console.error('Failed to load persistent session:', error);
      setSessionState(prev => ({
        ...prev,
        error: 'セッションの読み込みに失敗しました',
        isLoading: false,
      }));
    }
  };

  const saveSessionsToStorage = () => {
    try {
      storage.set('chat_sessions', sessionState.sessions);
    } catch (error) {
      console.error('Failed to save sessions to storage:', error);
    }
  };

  const createSession = useCallback(
    async (
      sessionName?: string,
      userId?: string
    ): Promise<SessionResponse | null> => {
      setSessionState(prev => ({ ...prev, isLoading: true, error: null }));

      try {
        const sessionId = generateId();
        const now = new Date();

        // Try to create session via API
        let apiSession: SessionResponse | null = null;
        try {
          const response = await fetch(buildApiUrl('/sessions'), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              user_id: userId,
              session_name:
                sessionName || `セッション ${now.toLocaleString('ja-JP')}`,
              metadata: {
                created_at: now.toISOString(),
                user_agent: navigator.userAgent,
              },
            }),
          });

          if (response.ok) {
            apiSession = await response.json();
          }
        } catch (apiError) {
          console.warn('Failed to create session via API:', apiError);
        }

        // Create local session (fallback or complement to API)
        const newSession: SessionResponse = apiSession || {
          session_id: sessionId,
          created_at: now.toISOString(),
          user_id: userId,
          session_name:
            sessionName || `セッション ${now.toLocaleString('ja-JP')}`,
          metadata: {
            created_at: now.toISOString(),
            user_agent: navigator.userAgent,
          },
        };

        const sessionData: SessionData = {
          session: newSession,
          messages: [],
          lastActivity: now,
          messageCount: 0,
        };

        setSessionState(prev => ({
          ...prev,
          sessions: [sessionData, ...prev.sessions],
          isLoading: false,
        }));

        // Set as current session
        setCurrentSession(newSession);
        setMessages([]);

        return newSession;
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'セッションの作成に失敗しました';
        setSessionState(prev => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
        return null;
      }
    },
    [setCurrentSession, setMessages]
  );

  const switchSession = useCallback(
    async (sessionId: string) => {
      const sessionData = sessionState.sessions.find(
        s => s.session.session_id === sessionId
      );
      if (!sessionData) {
        setSessionState(prev => ({
          ...prev,
          error: 'セッションが見つかりません',
        }));
        return;
      }

      try {
        // Try to load session from API
        try {
          const response = await fetch(buildApiUrl(`/sessions/${sessionId}`));
          if (response.ok) {
            const apiSession = await response.json();
            // Update local session with API data if available
            sessionData.session = { ...sessionData.session, ...apiSession };
          }
        } catch (apiError) {
          console.warn('Failed to load session from API:', apiError);
        }

        setCurrentSession(sessionData.session);
        setMessages(sessionData.messages);

        // Update last activity
        sessionData.lastActivity = new Date();
        setSessionState(prev => ({
          ...prev,
          sessions: prev.sessions.map(s =>
            s.session.session_id === sessionId ? sessionData : s
          ),
        }));
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'セッションの切り替えに失敗しました';
        setSessionState(prev => ({
          ...prev,
          error: errorMessage,
        }));
      }
    },
    [sessionState.sessions, setCurrentSession, setMessages]
  );

  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        // Try to delete from API
        try {
          await fetch(buildApiUrl(`/sessions/${sessionId}`), {
            method: 'DELETE',
          });
        } catch (apiError) {
          console.warn('Failed to delete session from API:', apiError);
        }

        // Remove from local state
        setSessionState(prev => ({
          ...prev,
          sessions: prev.sessions.filter(
            s => s.session.session_id !== sessionId
          ),
        }));

        // If we deleted the current session, switch to another or create new
        if (state.currentSession?.session_id === sessionId) {
          const remainingSessions = sessionState.sessions.filter(
            s => s.session.session_id !== sessionId
          );
          if (remainingSessions.length > 0) {
            const mostRecent = remainingSessions.sort(
              (a, b) => b.lastActivity.getTime() - a.lastActivity.getTime()
            )[0];
            setCurrentSession(mostRecent.session);
            setMessages(mostRecent.messages);
          } else {
            // Create a new session
            await createSession();
          }
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'セッションの削除に失敗しました';
        setSessionState(prev => ({
          ...prev,
          error: errorMessage,
        }));
      }
    },
    [
      sessionState.sessions,
      state.currentSession,
      setCurrentSession,
      setMessages,
      createSession,
    ]
  );

  const updateSessionMessages = useCallback(
    async (sessionId: string, messages: UIMessage[]) => {
      // Update persistent storage
      await sessionPersistence.updateSessionMessages(sessionId, messages);

      // Update local state
      setSessionState(prev => ({
        ...prev,
        sessions: prev.sessions.map(sessionData =>
          sessionData.session.session_id === sessionId
            ? {
                ...sessionData,
                messages,
                messageCount: messages.length,
                lastActivity: new Date(),
              }
            : sessionData
        ),
      }));

      // Update global state
      setMessages(messages);
    },
    [setMessages]
  );

  const renameSession = useCallback(
    async (sessionId: string, newName: string) => {
      try {
        // Try to update via API
        try {
          const response = await fetch(buildApiUrl(`/sessions/${sessionId}`), {
            method: 'PATCH',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              session_name: newName,
            }),
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
        } catch (apiError) {
          console.warn('Failed to rename session via API:', apiError);
        }

        // Update local state
        setSessionState(prev => ({
          ...prev,
          sessions: prev.sessions.map(sessionData =>
            sessionData.session.session_id === sessionId
              ? {
                  ...sessionData,
                  session: {
                    ...sessionData.session,
                    session_name: newName,
                  },
                }
              : sessionData
          ),
        }));

        // Update current session if it's the one being renamed
        if (state.currentSession?.session_id === sessionId) {
          setCurrentSession({
            ...state.currentSession,
            session_name: newName,
          });
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'セッション名の変更に失敗しました';
        setSessionState(prev => ({
          ...prev,
          error: errorMessage,
        }));
      }
    },
    [state.currentSession, setCurrentSession]
  );

  const clearAllSessions = useCallback(async () => {
    try {
      // Try to clear via API
      try {
        await fetch(buildApiUrl('/sessions'), {
          method: 'DELETE',
        });
      } catch (apiError) {
        console.warn('Failed to clear sessions via API:', apiError);
      }

      // Clear local storage
      storage.remove('chat_sessions');

      setSessionState({
        sessions: [],
        isLoading: false,
        error: null,
      });

      // Create a new session
      await createSession();
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : 'セッションのクリアに失敗しました';
      setSessionState(prev => ({
        ...prev,
        error: errorMessage,
      }));
    }
  }, [createSession]);

  return {
    sessions: sessionState.sessions,
    isLoading: sessionState.isLoading,
    error: sessionState.error,
    createSession,
    switchSession,
    deleteSession,
    updateSessionMessages,
    renameSession,
    clearAllSessions,
  };
};
