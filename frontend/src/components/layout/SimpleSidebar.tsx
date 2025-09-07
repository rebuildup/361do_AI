import React, { useState } from 'react';
import {
  MessageCircle,
  Settings,
  Bot,
  Zap,
  BarChart3,
  X,
  History,
} from 'lucide-react';
import { cn } from '@/utils';
import {
  ModelSelector,
  AgentStatus,
  ConfigurationPanel,
} from '@/components/settings';
import { SessionManager, SessionHistory } from '@/components/session';

interface SimpleSidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  className?: string;
}

const SimpleSidebar: React.FC<SimpleSidebarProps> = ({
  collapsed,
  onToggle,
  className,
}) => {
  const [activeTab, setActiveTab] = useState<
    'chat' | 'agent' | 'tools' | 'analytics' | 'settings' | 'history'
  >('chat');

  const navigationItems = [
    { id: 'chat' as const, icon: MessageCircle, label: 'チャット' },
    { id: 'history' as const, icon: History, label: '履歴' },
    { id: 'agent' as const, icon: Bot, label: 'エージェント' },
    { id: 'tools' as const, icon: Zap, label: 'ツール' },
    { id: 'analytics' as const, icon: BarChart3, label: '分析' },
    { id: 'settings' as const, icon: Settings, label: '設定' },
  ];

  const handleTabClick = (tabId: typeof activeTab) => {
    setActiveTab(tabId);
  };

  return (
    <>
      {/* Mobile overlay */}
      {window.innerWidth < 768 && !collapsed && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          'bg-gray-950 border-r border-gray-900 transition-all duration-300 flex flex-col relative z-50',
          'backdrop-blur-sm',
          collapsed ? 'w-12' : 'w-72',
          // Mobile responsive classes
          window.innerWidth < 768 &&
            !collapsed &&
            'fixed inset-y-0 left-0 animate-slide-in-left',
          window.innerWidth < 768 && collapsed && 'hidden',
          className
        )}
      >
        {/* Mobile close button */}
        {window.innerWidth < 768 && !collapsed && (
          <button
            onClick={onToggle}
            className="absolute top-4 right-4 p-2 hover:bg-gray-800 rounded-lg transition-colors lg:hidden"
            aria-label="サイドバーを閉じる"
          >
            <X size={20} className="text-white" />
          </button>
        )}

        <div className="flex-1 overflow-y-auto">
          {/* Navigation */}
          <div className="p-4">
            {collapsed ? (
              // Collapsed navigation - icons only
              <div className="flex flex-col space-y-2">
                {navigationItems.map(item => (
                  <button
                    key={item.id}
                    onClick={() => handleTabClick(item.id)}
                    className={cn(
                      'p-3 rounded-lg transition-all duration-200 flex items-center justify-center',
                      'hover:scale-105 active:scale-95',
                      activeTab === item.id
                        ? 'bg-gray-800 text-white'
                        : 'hover:bg-gray-800 text-gray-400'
                    )}
                    title={item.label}
                  >
                    <item.icon size={20} />
                  </button>
                ))}
              </div>
            ) : (
              // Expanded navigation
              <div className="space-y-2">
                {navigationItems.map(item => (
                  <button
                    key={item.id}
                    onClick={() => handleTabClick(item.id)}
                    className={cn(
                      'w-full p-3 rounded-lg transition-all duration-200 flex items-center space-x-3 text-left',
                      'hover:scale-[1.02] active:scale-[0.98]',
                      activeTab === item.id
                        ? 'bg-gray-800 text-white'
                        : 'hover:bg-gray-800 text-gray-400'
                    )}
                  >
                    <item.icon size={20} />
                    <span className="font-medium">{item.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Content Area - Only show when expanded */}
          {!collapsed && (
            <div className="p-4 border-t border-gray-900 animate-fade-in">
              {activeTab === 'chat' && (
                <div className="space-y-4">
                  <SessionManager />
                  <ModelSelector />
                  <AgentStatus showDetails={false} />
                </div>
              )}

              {activeTab === 'history' && (
                <div className="space-y-4">
                  <SessionHistory />
                </div>
              )}

              {activeTab === 'agent' && (
                <div className="space-y-4">
                  <AgentStatus showDetails={true} />
                </div>
              )}

              {activeTab === 'settings' && (
                <div className="space-y-4">
                  <ConfigurationPanel />
                </div>
              )}

              {activeTab === 'tools' && (
                <div className="text-center py-8">
                  <Zap size={48} className="text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400 text-sm">
                    ツール機能は開発中です
                  </p>
                </div>
              )}

              {activeTab === 'analytics' && (
                <div className="text-center py-8">
                  <BarChart3 size={48} className="text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400 text-sm">分析機能は開発中です</p>
                </div>
              )}
            </div>
          )}
        </div>
      </aside>
    </>
  );
};

export default SimpleSidebar;
