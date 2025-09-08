import React from 'react';
import { Menu, Settings } from 'lucide-react';
import { cn } from '@/utils';

interface HeaderProps {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  agentStatus: 'idle' | 'processing' | 'error';
  className?: string;
}

const Header: React.FC<HeaderProps> = ({
  sidebarCollapsed,
  onToggleSidebar,
  agentStatus,
  className,
}) => {
  const getStatusColor = () => {
    // Use monochrome base and accent for consistency
    return '';
  };

  const getStatusText = () => {
    switch (agentStatus) {
      case 'processing':
        return '処理中';
      case 'error':
        return 'エラー';
      case 'idle':
      default:
        return '稼働中';
    }
  };

  return (
    <header
      className={cn(
        'px-4 py-3',
        'bg-[var(--color-background)] border-b border-[var(--color-border)]',
        className
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-lg transition-colors"
            aria-label={
              sidebarCollapsed ? 'サイドバーを展開' : 'サイドバーを折りたたむ'
            }
          >
            <Menu size={20} className="text-[var(--color-text)]" />
          </button>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">
            361do_AI
          </h1>
        </div>

        <div className="flex items-center space-x-3">
          <div className="px-3 py-1 rounded-full text-sm font-medium text-[var(--color-text)] border border-[var(--color-border)] bg-[var(--color-background-secondary)]">
            {getStatusText()}
          </div>

          <button
            className="p-2 rounded-lg transition-colors"
            aria-label="設定"
          >
            <Settings
              size={20}
              className="text-[var(--color-text-secondary)] opacity-80"
            />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
