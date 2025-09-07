import React from 'react';
import { cn } from '@/utils';
import { ChatInterface } from './chat';

interface SimpleMainContentProps {
  className?: string;
}

const SimpleMainContent: React.FC<SimpleMainContentProps> = ({ className }) => {
  return <ChatInterface className={cn('flex-1', className)} />;
};

export default SimpleMainContent;
