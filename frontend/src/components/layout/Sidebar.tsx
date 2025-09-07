import React, { useState } from 'react';
import { MessageCircle, Settings, Bot, Zap, BarChart3, X } from 'lucide-react';
import { cn, isMobile } from '@/utils';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  agentStatus: 'idle' | 'processing' | 'error';
  onStatusChange: (status: 'idle' | 'processing' | 'error') => void;
  className?: string;
}

const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onToggle,
  agentStatus,
  onStatusChange,
  className,
}) => {
  const [selectedModel, setSelectedModel] = useState('deepseek-r1:7b');
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);

  const availableModels = [
    { id: 'deepseek-r1:7b', name: 'DeepSeek R1 7B' },
    { id: 'qwen2.5:7b-instruct-q4_k_m', name: 'Qwen2.5 7B Instruct' },
    { id: 'llama3.2:3b', name: 'Llama 3.2 3B' },
  ];

  const navigationItems = [
    { id: 'chat', icon: MessageCircle, label: 'チャット', active: true },
    { id: 'agent', icon: Bot, label: 'エージェント', active: false },
    { id: 'tools', icon: Zap, label: 'ツール', active: false },
    { id: 'analytics', icon: BarChart3, label: '分析', active: false },
    { id: 'settings', icon: Settings, label: '設定', active: false },
  ];

  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
    onStatusChange('processing');
    // Simulate model loading
    setTimeout(() => {
      onStatusChange('idle');
    }, 2000);
  };

  return (
    <>
      {/* Mobile overlay */}
      {isMobile() && !collapsed && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          'bg-gray-950 border-r border-gray-900 transition-all duration-300 flex flex-col relative z-50',
          collapsed ? 'w-12' : 'w-72',
          // Mobile responsive classes
          isMobile() && !collapsed && 'fixed inset-y-0 left-0',
          isMobile() && collapsed && 'hidden',
          className
        )}
      >
        {/* Mobile close button */}
        {isMobile() && !collapsed && (
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
                    className={cn(
                      'p-3 rounded-lg transition-colors flex items-center justify-center',
                      item.active
                        ? 'bg-primary-900 text-white'
                        : 'hover:bg-background-tertiary text-text-secondary'
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
                    className={cn(
                      'w-full p-3 rounded-lg transition-colors flex items-center space-x-3 text-left',
                      item.active
                        ? 'bg-primary-900 text-white'
                        : 'hover:bg-background-tertiary text-text-secondary'
                    )}
                  >
                    <item.icon size={20} />
                    <span className="font-medium">{item.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Model Selection and Settings - Only show when expanded */}
          {!collapsed && (
            <div className="p-4 border-t border-border">
              <div className="space-y-6">
                {/* Model Selection */}
                <div>
                  <h3 className="text-sm font-medium text-text-secondary mb-3">
                    モデル選択
                  </h3>
                  <select
                    value={selectedModel}
                    onChange={e => handleModelChange(e.target.value)}
                    className="input-primary w-full"
                  >
                    {availableModels.map(model => (
                      <option key={model.id} value={model.id}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Agent Settings */}
                <div>
                  <h3 className="text-sm font-medium text-text-secondary mb-3">
                    エージェント設定
                  </h3>
                  <div className="space-y-4">
                    {/* Streaming Toggle */}
                    <label className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        checked={streamingEnabled}
                        onChange={e => setStreamingEnabled(e.target.checked)}
                        className="w-4 h-4 rounded border-border bg-background-secondary text-primary-900 focus:ring-primary-900 focus:ring-2"
                      />
                      <span className="text-sm text-text">ストリーミング</span>
                    </label>

                    {/* Temperature Slider */}
                    <div>
                      <label className="flex items-center justify-between text-sm text-text-secondary mb-2">
                        <span>温度</span>
                        <span className="text-text">{temperature}</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={temperature}
                        onChange={e =>
                          setTemperature(parseFloat(e.target.value))
                        }
                        className="w-full h-2 bg-background-tertiary rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>

                    {/* Max Tokens */}
                    <div>
                      <label className="flex items-center justify-between text-sm text-text-secondary mb-2">
                        <span>最大トークン数</span>
                        <span className="text-text">{maxTokens}</span>
                      </label>
                      <input
                        type="range"
                        min="256"
                        max="4096"
                        step="256"
                        value={maxTokens}
                        onChange={e => setMaxTokens(parseInt(e.target.value))}
                        className="w-full h-2 bg-background-tertiary rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>
                  </div>
                </div>

                {/* Agent Status */}
                <div>
                  <h3 className="text-sm font-medium text-text-secondary mb-3">
                    エージェント状態
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-text-secondary">状態</span>
                      <span
                        className={cn(
                          'px-2 py-1 rounded-full text-xs font-medium',
                          agentStatus === 'idle' &&
                            'bg-green-900 text-green-100',
                          agentStatus === 'processing' &&
                            'bg-yellow-900 text-yellow-100',
                          agentStatus === 'error' && 'bg-red-900 text-red-100'
                        )}
                      >
                        {agentStatus === 'idle' && '待機中'}
                        {agentStatus === 'processing' && '処理中'}
                        {agentStatus === 'error' && 'エラー'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-text-secondary">モデル</span>
                      <span className="text-text text-xs">
                        {
                          availableModels.find(m => m.id === selectedModel)
                            ?.name
                        }
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
