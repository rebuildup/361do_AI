import React, { useState, useEffect } from 'react';
import { cn } from '@/utils';

interface StreamingTextProps {
  text: string;
  speed?: number; // Characters per second
  onComplete?: () => void;
  className?: string;
}

const StreamingText: React.FC<StreamingTextProps> = ({
  text,
  speed = 30,
  onComplete,
  className,
}) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (text.length === 0) {
      setDisplayedText('');
      setIsComplete(true);
      return;
    }

    setDisplayedText('');
    setIsComplete(false);
    let index = 0;

    const interval = setInterval(() => {
      if (index < text.length) {
        setDisplayedText(text.slice(0, index + 1));
        index++;
      } else {
        setIsComplete(true);
        clearInterval(interval);
        onComplete?.();
      }
    }, 1000 / speed);

    return () => clearInterval(interval);
  }, [text, speed, onComplete]);

  return (
    <div className={cn('relative', className)}>
      <span className="whitespace-pre-wrap break-words">{displayedText}</span>
      {!isComplete && (
        <span className="inline-block w-2 h-5 bg-gray-400 ml-1 animate-pulse" />
      )}
    </div>
  );
};

// Simple streaming text component for immediate display with cursor
interface SimpleStreamingTextProps {
  text: string;
  className?: string;
}

export const SimpleStreamingText: React.FC<SimpleStreamingTextProps> = ({
  text,
  className,
}) => {
  return (
    <div className={cn('relative', className)}>
      <span className="whitespace-pre-wrap break-words">{text}</span>
      <span className="inline-block w-2 h-5 bg-gray-400 ml-1 animate-pulse" />
    </div>
  );
};

export default StreamingText;
