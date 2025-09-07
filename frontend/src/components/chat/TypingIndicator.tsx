import React from 'react';
import { Bot } from 'lucide-react';
import { cn } from '@/utils';

interface TypingIndicatorProps {
  className?: string;
  message?: string;
}

const TypingIndicator: React.FC<TypingIndicatorProps> = ({
  className,
  message = 'AIが入力中...',
}) => {
  return (
    <div className={cn('flex gap-3 max-w-4xl mr-auto', className)}>
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center">
        <Bot size={16} className="text-gray-300" />
      </div>

      {/* Typing bubble */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 max-w-[200px]">
        <div className="flex items-center gap-2 text-gray-400 text-sm">
          <span>{message}</span>
          <div className="flex gap-1">
            <div
              className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
              style={{ animationDelay: '0ms' }}
            />
            <div
              className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
              style={{ animationDelay: '150ms' }}
            />
            <div
              className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
              style={{ animationDelay: '300ms' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
