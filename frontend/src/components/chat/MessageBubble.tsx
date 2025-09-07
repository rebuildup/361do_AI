import React, { useState, memo, useCallback } from 'react';
import { Copy, ChevronDown, ChevronUp, User, Bot, Check } from 'lucide-react';
import { cn, formatRelativeTime, copyToClipboard } from '@/utils';
import { SimpleStreamingText } from './StreamingText';
import { useToast } from '../ui/Toast';
import type { UIMessage } from '@/types';

interface MessageBubbleProps {
  message: UIMessage;
  className?: string;
}

const MessageBubble: React.FC<MessageBubbleProps> = memo(
  ({ message, className }) => {
    const [reasoningExpanded, setReasoningExpanded] = useState(false);
    const [copySuccess, setCopySuccess] = useState(false);
    const { addToast } = useToast();

    const isUser = message.role === 'user';
    const isAssistant = message.role === 'assistant';

    const handleCopy = useCallback(async () => {
      const success = await copyToClipboard(message.content);
      if (success) {
        setCopySuccess(true);
        addToast({
          type: 'success',
          message: 'メッセージをクリップボードにコピーしました',
          duration: 2000,
        });
        setTimeout(() => setCopySuccess(false), 2000);
      } else {
        addToast({
          type: 'error',
          message: 'コピーに失敗しました',
          duration: 3000,
        });
      }
    }, [message.content, addToast]);

    const toggleReasoning = useCallback(() => {
      setReasoningExpanded(!reasoningExpanded);
    }, [reasoningExpanded]);

    return (
      <div
        className={cn(
          'flex gap-3 max-w-4xl animate-fade-in',
          isUser ? 'ml-auto flex-row-reverse' : 'mr-auto',
          className
        )}
      >
        {/* Avatar */}
        <div
          className={cn(
            'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all duration-200',
            'hover:scale-105',
            isUser ? 'bg-gray-700' : 'bg-gray-800'
          )}
        >
          {isUser ? (
            <User size={16} className="text-gray-300" />
          ) : (
            <Bot size={16} className="text-gray-300" />
          )}
        </div>

        {/* Message Content */}
        <div
          className={cn('flex-1 min-w-0', isUser ? 'text-right' : 'text-left')}
        >
          {/* Message Bubble */}
          <div
            className={cn(
              'inline-block max-w-[80%] rounded-lg px-4 py-3 relative group',
              'transition-all duration-200 hover:scale-[1.02]',
              isUser
                ? 'bg-gray-700 text-white hover:bg-gray-600'
                : 'bg-gray-900 text-white border border-gray-800 hover:border-gray-700'
            )}
          >
            {/* Copy Button */}
            <button
              onClick={handleCopy}
              className={cn(
                'absolute top-2 opacity-0 group-hover:opacity-100 transition-all duration-200',
                'p-1 rounded hover:bg-gray-600 hover:scale-110',
                isUser ? 'left-2' : 'right-2'
              )}
              title={copySuccess ? 'コピーしました！' : 'コピー'}
            >
              {copySuccess ? (
                <Check
                  size={14}
                  className="text-green-400 animate-bounce-subtle"
                />
              ) : (
                <Copy size={14} className="text-gray-400" />
              )}
            </button>

            {/* Message Text */}
            <div className="pr-8">
              {message.isStreaming ? (
                <SimpleStreamingText text={message.content} />
              ) : (
                <p className="whitespace-pre-wrap break-words">
                  {message.content}
                </p>
              )}
            </div>

            {/* Error Display */}
            {message.error && (
              <div className="mt-2 p-2 bg-red-900 border border-red-700 rounded text-red-100 text-sm">
                エラー: {message.error}
              </div>
            )}
          </div>

          {/* Reasoning Section */}
          {message.reasoning && isAssistant && (
            <div className="mt-2 max-w-[80%] inline-block">
              <button
                onClick={toggleReasoning}
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-300 transition-all duration-200 hover:scale-105"
              >
                <div className="transition-transform duration-200">
                  {reasoningExpanded ? (
                    <ChevronUp size={16} />
                  ) : (
                    <ChevronDown size={16} />
                  )}
                </div>
                推論を{reasoningExpanded ? '隠す' : '表示'}
              </button>

              {reasoningExpanded && (
                <div className="mt-2 bg-gray-800 border-l-4 border-gray-600 rounded-lg p-3 text-sm text-gray-300 italic animate-fade-in">
                  <div className="whitespace-pre-wrap break-words">
                    {message.reasoning}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Timestamp */}
          <div
            className={cn(
              'mt-1 text-xs text-gray-500',
              isUser ? 'text-right' : 'text-left'
            )}
          >
            {formatRelativeTime(message.timestamp)}
          </div>
        </div>
      </div>
    );
  }
);

MessageBubble.displayName = 'MessageBubble';

export default MessageBubble;
