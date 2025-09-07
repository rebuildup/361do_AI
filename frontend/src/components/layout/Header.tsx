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
    switch (agentStatus) {
      case 'processing':
        return 'bg-yellow-600 text-yellow-100';
      case 'error':
        return 'bg-red-600 text-red-100';
      case 'idle':
      default:
        return 'bg-green-600 text-green-100';
    }
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
        'bg-gray-950 border-b border-gray-900 px-4 py-3',
        className
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <button
            onClick={onToggleSidebar}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
            aria-label={
              sidebarCollapsed ? 'サイドバーを展開' : 'サイドバーを折りたたむ'
            }
          >
            <Menu size={20} className="text-white" />
          </button>
          <h1 className="text-xl font-semibold text-white">361do_AI</h1>
        </div>

        <div className="flex items-center space-x-3">
          <div
            className={cn(
              'px-3 py-1 rounded-full text-sm font-medium',
              getStatusColor()
            )}
          >
            {getStatusText()}
          </div>

          <button
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
            aria-label="設定"
          >
            <Settings size={20} className="text-gray-300" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
