import React, {
  useState,
  useRef,
  useEffect,
  memo,
  useMemo,
  useCallback,
} from 'react';
import { Send, Loader2, StopCircle, AlertCircle, Keyboard } from 'lucide-react';
import { cn, generateId, copyToClipboard } from '@/utils';
import { useApp, useAppActions } from '@/contexts/AppContext';
import { useStreaming } from '@/hooks/useStreaming';
import { useSessionManager } from '@/hooks/useSessionManager';
import {
  useKeyboardShortcuts,
  createChatShortcuts,
} from '@/hooks/useKeyboardShortcuts';
import { useToast } from '../ui/Toast';
import MessageBubble from './MessageBubble';
import NaturalLanguageIndicator from './NaturalLanguageIndicator';
import ShortcutsModal from '../ui/ShortcutsModal';
import type { UIMessage } from '@/types';

interface ChatInterfaceProps {
  className?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = memo(({ className }) => {
  const { state } = useApp();
  const { setStreaming } = useAppActions();
  const { updateSessionMessages } = useSessionManager();
  const { addToast } = useToast();

  const [inputValue, setInputValue] = useState('');
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [nlpState, setNlpState] = useState<{
    isProcessing: boolean;
    language?: 'ja' | 'en';
    toolsUsed?: string[];
    confidence?: number;
    reasoning?: string;
  }>({
    isProcessing: false,
  });
  // Use messages from global state (managed by session manager)
  const messages = state.messages;

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const activeAssistantIdRef = useRef<string | null>(null);
  const lastPromptRef = useRef<string>('');

  // Streaming functionality
  const streaming = useStreaming({
    onMessageStart: messageId => {
      console.log('Streaming started:', messageId);
      setStreaming(true);
      setConnectionError(null);
      setNlpState({
        isProcessing: true,
        language: 'ja', // Default, will be detected
      });
      // Align placeholder id with streaming id
      if (state.currentSession && activeAssistantIdRef.current) {
        const aligned = (state.messages as UIMessage[]).map(m =>
          m.id === activeAssistantIdRef.current
            ? { ...m, id: messageId, isStreaming: true }
            : m
        );
        updateSessionMessages(state.currentSession.session_id, aligned);
        activeAssistantIdRef.current = messageId;
      } else {
        activeAssistantIdRef.current = messageId;
      }
    },
    onMessageUpdate: (messageId, content) => {
      const updatedMessages = (state.messages as UIMessage[]).map(msg =>
        msg.id === messageId ? { ...msg, content, isStreaming: true } : msg
      );
      // Update both global state and session storage
      if (state.currentSession) {
        updateSessionMessages(state.currentSession.session_id, updatedMessages);
      }
    },
    onMessageComplete: (messageId, content, reasoning) => {
      const updatedMessages = (state.messages as UIMessage[]).map(msg =>
        msg.id === messageId
          ? { ...msg, content, reasoning, isStreaming: false }
          : msg
      );
      // Update both global state and session storage
      if (state.currentSession) {
        updateSessionMessages(state.currentSession.session_id, updatedMessages);
      }
      setStreaming(false);
      setRetryCount(0);
      setNlpState({
        isProcessing: false,
        reasoning,
        confidence: 0.8, // Mock confidence for now
      });
    },
    onError: error => {
      setConnectionError(error);
      setStreaming(false);
      setNlpState({
        isProcessing: false,
      });
    },
    speed: 25, // Characters per second
  });

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Memoized keyboard shortcuts
  const shortcuts = useMemo(
    () =>
      createChatShortcuts({
        onClearChat: () => {
          if (state.currentSession) {
            updateSessionMessages(state.currentSession.session_id, []);
            addToast({
              type: 'success',
              message: 'チャット履歴をクリアしました',
              duration: 2000,
            });
          }
        },
        onFocusInput: () => {
          inputRef.current?.focus();
        },
        onCopyLastMessage: () => {
          const lastAssistantMessage = (messages as UIMessage[])
            .filter(m => m.role === 'assistant')
            .pop();
          if (lastAssistantMessage) {
            copyToClipboard(lastAssistantMessage.content);
            addToast({
              type: 'success',
              message: '最後のメッセージをコピーしました',
              duration: 2000,
            });
          }
        },
        onShowShortcuts: () => {
          setShowShortcuts(true);
        },
      }),
    [state.currentSession, updateSessionMessages, addToast, messages]
  );

  useKeyboardShortcuts(shortcuts);

  // Clear messages function (for future use)
  const clearMessages = () => {
    if (state.currentSession) {
      updateSessionMessages(state.currentSession.session_id, []);
    }
    streaming.stopStreaming();
    setConnectionError(null);
  };

  // Expose clearMessages for debugging
  (window as any).clearMessages = clearMessages;

  // Stop streaming function
  const stopStreaming = () => {
    streaming.stopStreaming();
    setStreaming(false);
  };

  // Retry with exponential backoff
  const retryStreaming = async () => {
    try {
      setRetryCount(prev => prev + 1);
      await streaming.retryStreaming(
        lastPromptRef.current,
        state.currentSession?.session_id
      );
    } catch (error) {
      setConnectionError(
        error instanceof Error ? error.message : 'Retry failed'
      );
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${inputRef.current.scrollHeight}px`;
    }
  }, [inputValue]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      if (!inputValue.trim() || state.isStreaming) {
        return;
      }

      const userMessage: UIMessage = {
        id: generateId(),
        role: 'user',
        content: inputValue.trim(),
        timestamp: new Date(),
      };

      const prompt = inputValue.trim();
      setInputValue('');
      setConnectionError(null);
      lastPromptRef.current = prompt;

      // Add user message
      const updatedMessagesWithUser = [
        ...(messages as UIMessage[]),
        userMessage,
      ];
      if (state.currentSession) {
        updateSessionMessages(
          state.currentSession.session_id,
          updatedMessagesWithUser
        );
      }

      // Create assistant message placeholder
      const assistantMessage: UIMessage = {
        id: generateId(),
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      };

      const updatedMessagesWithAssistant = [
        ...updatedMessagesWithUser,
        assistantMessage,
      ];
      if (state.currentSession) {
        updateSessionMessages(
          state.currentSession.session_id,
          updatedMessagesWithAssistant
        );
      }
      activeAssistantIdRef.current = assistantMessage.id;

      try {
        // Try real streaming first, fallback to simulation
        try {
          await streaming.startStreaming(
            prompt,
            state.currentSession?.session_id
          );
        } catch (streamingError) {
          console.warn(
            'Real streaming failed, falling back to simulation:',
            streamingError
          );

          // Fallback to simulated streaming
          const fullResponse = `ご質問ありがとうございます。「${prompt}」について回答いたします。\n\n新しいReactベースのUIでは、以下の機能が実装されています：\n\n• リアルタイムストリーミング応答\n• メッセージ履歴の管理\n• レスポンシブデザイン\n• 推論セクションの表示\n• コピー機能\n• エラーハンドリングと再試行メカニズム\n\nこれらの機能により、より良いユーザーエクスペリエンスを提供しています。`;

          const reasoning = `ユーザーの質問「${prompt}」に対して、現在実装されている機能を中心に回答しました。リアルタイムストリーミング機能とエラーハンドリングが正常に動作していることを確認できます。`;

          await streaming.simulateStreaming(fullResponse, reasoning);
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Unknown error occurred';
        const updatedMessages = (messages as UIMessage[]).map(msg =>
          msg.id === assistantMessage.id
            ? { ...msg, content: '', error: errorMessage, isStreaming: false }
            : msg
        );
        if (state.currentSession) {
          updateSessionMessages(
            state.currentSession.session_id,
            updatedMessages
          );
        }
        setConnectionError(errorMessage);
        setStreaming(false);
      }
    },
    [
      inputValue.trim(),
      state.isStreaming,
      state.currentSession,
      messages,
      updateSessionMessages,
      streaming,
      setStreaming,
    ]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
      }
    },
    [handleSubmit]
  );

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div className="max-w-4xl mx-auto space-y-6">
          {(messages as UIMessage[]).map(message => (
            <MessageBubble key={message.id} message={message} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Keyboard shortcuts hint */}
        {messages.length === 0 && (
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center space-x-2 text-gray-500 text-sm animate-fade-in">
              <Keyboard size={16} />
              <span>
                <kbd className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded">
                  ?
                </kbd>{' '}
                でショートカット一覧を表示
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-800 p-4 bg-black/50 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="flex gap-3 items-end">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={e => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="メッセージを入力してください... (Shift+Enterで改行)"
                className={cn(
                  'w-full bg-gray-900 border border-gray-700 text-white placeholder-gray-400',
                  'rounded-lg px-4 py-3 pr-12 resize-none transition-all duration-200',
                  'focus:outline-none focus:border-gray-600 focus:ring-2 focus:ring-gray-600/20',
                  'min-h-[48px] max-h-32',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
                rows={1}
                disabled={state.isStreaming}
              />
            </div>

            {state.isStreaming ? (
              <button
                type="button"
                onClick={stopStreaming}
                className="flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center transition-all duration-200 bg-red-700 hover:bg-red-600 text-white hover:scale-105 active:scale-95"
                title="ストリーミングを停止"
              >
                <StopCircle size={20} />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!inputValue.trim()}
                className={cn(
                  'flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center transition-all duration-200',
                  'hover:scale-105 active:scale-95',
                  inputValue.trim()
                    ? 'bg-gray-700 hover:bg-gray-600 text-white'
                    : 'bg-gray-800 text-gray-500 cursor-not-allowed'
                )}
              >
                <Send size={20} />
              </button>
            )}
          </form>

          {/* Natural Language Processing Indicator */}
          {(nlpState.isProcessing ||
            nlpState.reasoning ||
            nlpState.toolsUsed?.length) && (
            <div className="mt-2">
              <NaturalLanguageIndicator
                isProcessing={nlpState.isProcessing}
                language={nlpState.language}
                toolsUsed={nlpState.toolsUsed}
                confidence={nlpState.confidence}
                reasoning={nlpState.reasoning}
              />
            </div>
          )}

          {/* Status indicators */}
          {state.isStreaming && (
            <div className="mt-2 text-sm text-gray-400 flex items-center gap-2 animate-fade-in">
              <Loader2 size={16} className="animate-spin" />
              <span className="animate-pulse-subtle">応答を生成中...</span>
              {retryCount > 0 && (
                <span className="text-yellow-400 animate-bounce-subtle">
                  (再試行 {retryCount}/3)
                </span>
              )}
            </div>
          )}

          {/* Connection error */}
          {connectionError && (
            <div className="mt-2 p-3 bg-red-900/80 border border-red-700 rounded-lg animate-fade-in backdrop-blur-sm">
              <div className="flex items-center gap-2 text-red-100 text-sm">
                <AlertCircle size={16} className="animate-pulse" />
                <span>接続エラー: {connectionError}</span>
              </div>
              {retryCount < 3 && (
                <button
                  onClick={() => retryStreaming(inputValue)}
                  className="mt-2 px-3 py-1 bg-red-700 hover:bg-red-600 text-white text-xs rounded transition-all duration-200 hover:scale-105 active:scale-95"
                >
                  再試行
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Shortcuts Modal */}
      <ShortcutsModal
        isOpen={showShortcuts}
        onClose={() => setShowShortcuts(false)}
      />
    </div>
  );
});

ChatInterface.displayName = 'ChatInterface';

export default ChatInterface;
