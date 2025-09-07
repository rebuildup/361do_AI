import React from 'react';
import { Wifi, WifiOff, AlertCircle, CheckCircle } from 'lucide-react';
import { cn } from '@/utils';

interface ConnectionStatusProps {
  status: 'connected' | 'connecting' | 'disconnected' | 'error';
  error?: string;
  onRetry?: () => void;
  className?: string;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  status,
  error,
  onRetry,
  className,
}) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'connected':
        return {
          icon: CheckCircle,
          text: '接続済み',
          color: 'text-green-400',
          bgColor: 'bg-green-900',
          borderColor: 'border-green-700',
        };
      case 'connecting':
        return {
          icon: Wifi,
          text: '接続中...',
          color: 'text-yellow-400',
          bgColor: 'bg-yellow-900',
          borderColor: 'border-yellow-700',
        };
      case 'disconnected':
        return {
          icon: WifiOff,
          text: '切断済み',
          color: 'text-gray-400',
          bgColor: 'bg-gray-900',
          borderColor: 'border-gray-700',
        };
      case 'error':
        return {
          icon: AlertCircle,
          text: 'エラー',
          color: 'text-red-400',
          bgColor: 'bg-red-900',
          borderColor: 'border-red-700',
        };
      default:
        return {
          icon: WifiOff,
          text: '不明',
          color: 'text-gray-400',
          bgColor: 'bg-gray-900',
          borderColor: 'border-gray-700',
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <div
      className={cn(
        'flex items-center gap-2 px-3 py-2 rounded-lg border text-sm',
        config.bgColor,
        config.borderColor,
        className
      )}
    >
      <Icon
        size={16}
        className={cn(config.color, status === 'connecting' && 'animate-pulse')}
      />
      <span className={config.color}>{config.text}</span>

      {error && <span className="text-red-300 text-xs ml-2">{error}</span>}

      {status === 'error' && onRetry && (
        <button
          onClick={onRetry}
          className="ml-2 px-2 py-1 bg-red-700 hover:bg-red-600 text-white text-xs rounded transition-colors"
        >
          再接続
        </button>
      )}
    </div>
  );
};

export default ConnectionStatus;
