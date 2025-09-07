/**
 * Fallback UI Components
 * フォールバック UI コンポーネント
 */

import React from 'react';
import {
  AlertTriangle,
  WifiOff,
  Server,
  RefreshCw,
  MessageCircle,
  Settings,
  User,
} from 'lucide-react';

interface FallbackProps {
  onRetry?: () => void;
  message?: string;
  showRetry?: boolean;
}

/**
 * Generic loading fallback
 */
export const LoadingFallback: React.FC<{ message?: string }> = ({
  message = '読み込み中...',
}) => (
  <div className="flex flex-col items-center justify-center p-8 text-gray-400">
    <div className="w-8 h-8 border-2 border-gray-600 border-t-blue-500 rounded-full animate-spin mb-4" />
    <p className="text-sm">{message}</p>
  </div>
);

/**
 * Network error fallback
 */
export const NetworkErrorFallback: React.FC<FallbackProps> = ({
  onRetry,
  message = 'ネットワーク接続を確認してください',
  showRetry = true,
}) => (
  <div className="flex flex-col items-center justify-center p-8 text-center">
    <WifiOff className="w-12 h-12 text-red-400 mb-4" />
    <h3 className="text-lg font-semibold text-red-400 mb-2">接続エラー</h3>
    <p className="text-gray-400 mb-4">{message}</p>
    {showRetry && onRetry && (
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        <RefreshCw className="w-4 h-4" />
        再試行
      </button>
    )}
  </div>
);

/**
 * Backend unavailable fallback
 */
export const BackendUnavailableFallback: React.FC<FallbackProps> = ({
  onRetry,
  message = 'サーバーに接続できません',
  showRetry = true,
}) => (
  <div className="flex flex-col items-center justify-center p-8 text-center">
    <Server className="w-12 h-12 text-orange-400 mb-4" />
    <h3 className="text-lg font-semibold text-orange-400 mb-2">
      サーバーエラー
    </h3>
    <p className="text-gray-400 mb-4">{message}</p>
    <div className="bg-orange-900/20 border border-orange-700 rounded p-3 mb-4 text-sm text-orange-300">
      <p>一部の機能が制限される場合があります。</p>
    </div>
    {showRetry && onRetry && (
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 transition-colors"
      >
        <RefreshCw className="w-4 h-4" />
        再接続を試行
      </button>
    )}
  </div>
);

/**
 * Chat interface fallback
 */
export const ChatFallback: React.FC<FallbackProps> = ({
  onRetry,
  message = 'チャット機能を読み込めませんでした',
}) => (
  <div className="flex flex-col items-center justify-center min-h-96 p-8 text-center bg-gray-900/50 rounded-lg border border-gray-700">
    <MessageCircle className="w-16 h-16 text-gray-500 mb-4" />
    <h3 className="text-xl font-semibold text-gray-300 mb-2">チャット機能</h3>
    <p className="text-gray-400 mb-6">{message}</p>

    <div className="bg-gray-800 rounded-lg p-4 mb-6 text-left max-w-md">
      <h4 className="font-semibold text-gray-300 mb-2">利用可能な代替手段:</h4>
      <ul className="text-sm text-gray-400 space-y-1">
        <li>• ページを再読み込みしてください</li>
        <li>• ネットワーク接続を確認してください</li>
        <li>• しばらく時間をおいてから再度お試しください</li>
      </ul>
    </div>

    {onRetry && (
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
      >
        <RefreshCw className="w-4 h-4" />
        チャットを再読み込み
      </button>
    )}
  </div>
);

/**
 * Settings fallback
 */
export const SettingsFallback: React.FC<FallbackProps> = ({
  onRetry,
  message = '設定を読み込めませんでした',
}) => (
  <div className="flex flex-col items-center justify-center p-8 text-center">
    <Settings className="w-12 h-12 text-gray-500 mb-4" />
    <h3 className="text-lg font-semibold text-gray-300 mb-2">設定</h3>
    <p className="text-gray-400 mb-4">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
      >
        <RefreshCw className="w-4 h-4" />
        設定を再読み込み
      </button>
    )}
  </div>
);

/**
 * User profile fallback
 */
export const UserProfileFallback: React.FC<FallbackProps> = ({
  onRetry,
  message = 'ユーザー情報を読み込めませんでした',
}) => (
  <div className="flex flex-col items-center justify-center p-6 text-center">
    <User className="w-10 h-10 text-gray-500 mb-3" />
    <h4 className="font-semibold text-gray-300 mb-1">ユーザー情報</h4>
    <p className="text-sm text-gray-400 mb-3">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="text-sm px-3 py-1 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
      >
        再読み込み
      </button>
    )}
  </div>
);

/**
 * Offline mode banner
 */
export const OfflineModeBanner: React.FC = () => (
  <div className="bg-yellow-900/20 border-b border-yellow-700 px-4 py-2">
    <div className="flex items-center justify-center gap-2 text-yellow-300">
      <WifiOff className="w-4 h-4" />
      <span className="text-sm">
        オフラインモード - 一部の機能が制限されています
      </span>
    </div>
  </div>
);

/**
 * Backend disconnected banner
 */
export const BackendDisconnectedBanner: React.FC<{ onRetry?: () => void }> = ({
  onRetry,
}) => (
  <div className="bg-red-900/20 border-b border-red-700 px-4 py-2">
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 text-red-300">
        <Server className="w-4 h-4" />
        <span className="text-sm">サーバーに接続できません</span>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="text-sm px-2 py-1 bg-red-700 text-white rounded hover:bg-red-600 transition-colors"
        >
          再接続
        </button>
      )}
    </div>
  </div>
);

/**
 * Feature unavailable fallback
 */
export const FeatureUnavailableFallback: React.FC<{
  featureName: string;
  reason?: string;
  onRetry?: () => void;
}> = ({ featureName, reason = '現在利用できません', onRetry }) => (
  <div className="flex flex-col items-center justify-center p-6 text-center bg-gray-800/50 rounded-lg border border-gray-600">
    <AlertTriangle className="w-8 h-8 text-yellow-400 mb-3" />
    <h4 className="font-semibold text-gray-300 mb-1">{featureName}</h4>
    <p className="text-sm text-gray-400 mb-3">{reason}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="text-sm px-3 py-1 bg-yellow-700 text-white rounded hover:bg-yellow-600 transition-colors"
      >
        再試行
      </button>
    )}
  </div>
);

/**
 * Minimal chat interface for degraded mode
 */
export const MinimalChatInterface: React.FC = () => (
  <div className="flex flex-col h-full bg-gray-900 rounded-lg border border-gray-700">
    <div className="flex items-center justify-between p-4 border-b border-gray-700">
      <h3 className="font-semibold text-gray-300">簡易チャット</h3>
      <div className="text-xs text-yellow-400 bg-yellow-900/20 px-2 py-1 rounded">
        制限モード
      </div>
    </div>

    <div className="flex-1 p-4 text-center text-gray-400">
      <MessageCircle className="w-12 h-12 mx-auto mb-3 text-gray-500" />
      <p className="mb-2">フル機能のチャットは現在利用できません</p>
      <p className="text-sm">
        ネットワーク接続を確認してページを再読み込みしてください
      </p>
    </div>

    <div className="p-4 border-t border-gray-700">
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="メッセージを入力... (現在無効)"
          disabled
          className="flex-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded text-gray-500 cursor-not-allowed"
        />
        <button
          disabled
          className="px-4 py-2 bg-gray-700 text-gray-500 rounded cursor-not-allowed"
        >
          送信
        </button>
      </div>
    </div>
  </div>
);

/**
 * Emergency contact information
 */
export const EmergencyContactInfo: React.FC = () => (
  <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4 text-center">
    <h4 className="font-semibold text-blue-300 mb-2">サポートが必要ですか？</h4>
    <p className="text-sm text-gray-400 mb-3">
      問題が解決しない場合は、以下の方法でサポートにお問い合わせください：
    </p>
    <div className="space-y-2 text-sm">
      <div className="text-gray-300">
        <strong>エラー情報:</strong>{' '}
        ブラウザの開発者ツール（F12）でコンソールを確認
      </div>
      <div className="text-gray-300">
        <strong>一時的な解決策:</strong> ページの再読み込み（Ctrl+F5）
      </div>
    </div>
  </div>
);

export default {
  LoadingFallback,
  NetworkErrorFallback,
  BackendUnavailableFallback,
  ChatFallback,
  SettingsFallback,
  UserProfileFallback,
  OfflineModeBanner,
  BackendDisconnectedBanner,
  FeatureUnavailableFallback,
  MinimalChatInterface,
  EmergencyContactInfo,
};
