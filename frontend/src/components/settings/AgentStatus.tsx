import React, { useState, useEffect } from 'react';
import {
  Bot,
  Activity,
  Cpu,
  HardDrive,
  Zap,
  AlertTriangle,
  CheckCircle,
  Loader2,
} from 'lucide-react';
import { cn, formatFileSize } from '@/utils';
import { useApp } from '@/contexts/AppContext';

interface AgentStatusProps {
  className?: string;
  showDetails?: boolean;
}

interface SystemStats {
  cpu: {
    usage: number;
    temperature?: number;
  };
  memory: {
    usage: number;
    total: number;
  };
  gpu?: {
    usage: number;
    memory: number;
    temperature?: number;
  };
  uptime: number;
  modelLoaded: boolean;
  lastActivity: Date;
}

const AgentStatus: React.FC<AgentStatusProps> = ({
  className,
  showDetails = true,
}) => {
  const { state } = useApp();
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch system stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/v1/system/stats');
        if (response.ok) {
          const data = await response.json();
          setSystemStats(data);
          setError(null);
        } else {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch {
        // Simulate system stats for development
        const mockStats: SystemStats = {
          cpu: {
            usage: Math.random() * 100,
            temperature: 45 + Math.random() * 20,
          },
          memory: {
            usage: 4.2 + Math.random() * 2,
            total: 16,
          },
          gpu: {
            usage: Math.random() * 100,
            memory: 6.8 + Math.random() * 1.2,
            temperature: 55 + Math.random() * 15,
          },
          uptime: Date.now() - Math.random() * 86400000, // Random uptime up to 24h
          modelLoaded: state.activeModel !== '',
          lastActivity: new Date(Date.now() - Math.random() * 300000), // Last 5 minutes
        };
        setSystemStats(mockStats);
        setError(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [state.activeModel]);

  const getStatusColor = () => {
    switch (state.agentStatus) {
      case 'processing':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      case 'idle':
      default:
        return 'text-green-400';
    }
  };

  const getStatusIcon = () => {
    switch (state.agentStatus) {
      case 'processing':
        return <Loader2 size={16} className="animate-spin" />;
      case 'error':
        return <AlertTriangle size={16} />;
      case 'idle':
      default:
        return <CheckCircle size={16} />;
    }
  };

  const getStatusText = () => {
    switch (state.agentStatus) {
      case 'processing':
        return '処理中';
      case 'error':
        return 'エラー';
      case 'idle':
      default:
        return '待機中';
    }
  };

  const formatUptime = (uptime: number) => {
    const seconds = Math.floor((Date.now() - uptime) / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (hours > 0) {
      return `${hours}時間${minutes}分`;
    }
    return `${minutes}分`;
  };

  const getUsageColor = (usage: number, threshold = 80) => {
    if (usage > threshold) return 'text-red-400';
    if (usage > threshold * 0.7) return 'text-yellow-400';
    return 'text-green-400';
  };

  if (isLoading) {
    return (
      <div
        className={cn(
          'bg-gray-900 border border-gray-800 rounded-lg p-4',
          className
        )}
      >
        <div className="flex items-center gap-2 text-gray-400">
          <Loader2 size={16} className="animate-spin" />
          <span className="text-sm">システム情報を読み込み中...</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'bg-gray-900 border border-gray-800 rounded-lg p-4',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Bot size={20} className="text-gray-300" />
        <h3 className="text-white font-medium">エージェント状態</h3>
      </div>

      {/* Status Overview */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-sm">状態</span>
          <div className={cn('flex items-center gap-2', getStatusColor())}>
            {getStatusIcon()}
            <span className="text-sm font-medium">{getStatusText()}</span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-sm">アクティブモデル</span>
          <span className="text-white text-sm truncate max-w-[150px]">
            {state.activeModel || 'なし'}
          </span>
        </div>

        {systemStats && (
          <>
            <div className="flex items-center justify-between">
              <span className="text-gray-400 text-sm">稼働時間</span>
              <span className="text-white text-sm">
                {formatUptime(systemStats.uptime)}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-gray-400 text-sm">最終活動</span>
              <span className="text-white text-sm">
                {Math.floor(
                  (Date.now() - systemStats.lastActivity.getTime()) / 1000
                )}
                秒前
              </span>
            </div>
          </>
        )}
      </div>

      {/* Detailed Stats */}
      {showDetails && systemStats && (
        <div className="mt-4 pt-4 border-t border-gray-800 space-y-3">
          {/* CPU Usage */}
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-gray-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">CPU</span>
                <span className={getUsageColor(systemStats.cpu.usage)}>
                  {systemStats.cpu.usage.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2 mt-1">
                <div
                  className={cn(
                    'h-2 rounded-full transition-all',
                    systemStats.cpu.usage > 80
                      ? 'bg-red-500'
                      : systemStats.cpu.usage > 56
                        ? 'bg-yellow-500'
                        : 'bg-green-500'
                  )}
                  style={{ width: `${Math.min(systemStats.cpu.usage, 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Memory Usage */}
          <div className="flex items-center gap-2">
            <HardDrive size={16} className="text-gray-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">メモリ</span>
                <span
                  className={getUsageColor(
                    (systemStats.memory.usage / systemStats.memory.total) * 100
                  )}
                >
                  {formatFileSize(
                    systemStats.memory.usage * 1024 * 1024 * 1024
                  )}{' '}
                  /{' '}
                  {formatFileSize(
                    systemStats.memory.total * 1024 * 1024 * 1024
                  )}
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2 mt-1">
                <div
                  className={cn(
                    'h-2 rounded-full transition-all',
                    systemStats.memory.usage / systemStats.memory.total > 0.8
                      ? 'bg-red-500'
                      : systemStats.memory.usage / systemStats.memory.total >
                          0.7
                        ? 'bg-yellow-500'
                        : 'bg-green-500'
                  )}
                  style={{
                    width: `${Math.min((systemStats.memory.usage / systemStats.memory.total) * 100, 100)}%`,
                  }}
                />
              </div>
            </div>
          </div>

          {/* GPU Usage (if available) */}
          {systemStats.gpu && (
            <div className="flex items-center gap-2">
              <Zap size={16} className="text-gray-400" />
              <div className="flex-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">GPU</span>
                  <span className={getUsageColor(systemStats.gpu.usage)}>
                    {systemStats.gpu.usage.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2 mt-1">
                  <div
                    className={cn(
                      'h-2 rounded-full transition-all',
                      systemStats.gpu.usage > 80
                        ? 'bg-red-500'
                        : systemStats.gpu.usage > 56
                          ? 'bg-yellow-500'
                          : 'bg-green-500'
                    )}
                    style={{
                      width: `${Math.min(systemStats.gpu.usage, 100)}%`,
                    }}
                  />
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  VRAM{' '}
                  {formatFileSize(systemStats.gpu.memory * 1024 * 1024 * 1024)}
                </div>
              </div>
            </div>
          )}

          {/* Activity Indicator */}
          <div className="flex items-center gap-2">
            <Activity size={16} className="text-gray-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">活動状況</span>
                <div
                  className={cn(
                    'w-2 h-2 rounded-full',
                    state.agentStatus === 'processing'
                      ? 'bg-yellow-400 animate-pulse'
                      : state.agentStatus === 'error'
                        ? 'bg-red-400'
                        : 'bg-green-400'
                  )}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-2 bg-red-900 border border-red-700 rounded text-red-100 text-sm flex items-center gap-2">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};

export default AgentStatus;
