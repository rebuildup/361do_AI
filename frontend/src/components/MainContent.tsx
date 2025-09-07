import React from 'react';
import { cn } from '@/utils';

interface MainContentProps {
  className?: string;
}

const MainContent: React.FC<MainContentProps> = ({ className }) => {
  return (
    <div className={cn('flex-1 flex flex-col', className)}>
      {/* Chat Messages Area */}
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="main-content">
          <div className="space-y-4">
            {/* Welcome Message */}
            <div className="card">
              <h2 className="text-lg font-semibold mb-2">
                React + Tailwind CSS WebUIへようこそ
              </h2>
              <p className="text-text-secondary">
                StreamlitからReactへの移行が完了しました。
                モノクロダークテーマとレスポンシブデザインが適用されています。
              </p>
            </div>

            {/* Sample Chat Messages */}
            <div className="space-y-3">
              <div className="message-user">
                <p>こんにちは！新しいUIはどうですか？</p>
                <div className="text-xs text-gray-300 mt-1">
                  {new Date().toLocaleTimeString('ja-JP')}
                </div>
              </div>

              <div className="message-assistant">
                <p>
                  新しいReact + Tailwind CSSベースのUIが正常に動作しています！
                  レスポンシブデザインとモノクロダークテーマが適用されています。
                </p>
                <div className="reasoning-section">
                  <p className="text-sm">
                    推論: ユーザーからの質問に対して、現在のUI状態を確認し、
                    実装された機能について説明しています。
                  </p>
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {new Date().toLocaleTimeString('ja-JP')}
                </div>
              </div>

              <div className="message-user">
                <p>
                  レスポンシブデザインのテストをしてみましょう。サイドバーの動作はどうですか？
                </p>
                <div className="text-xs text-gray-300 mt-1">
                  {new Date().toLocaleTimeString('ja-JP')}
                </div>
              </div>

              <div className="message-assistant">
                <p>レスポンシブデザインが正常に動作しています：</p>
                <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
                  <li>デスクトップ（&gt;1056px）: サイドバー展開（288px幅）</li>
                  <li>
                    タブレット（768px-1056px）:
                    サイドバー自動折りたたみ（48px幅）
                  </li>
                  <li>モバイル（&lt;768px）: オーバーレイサイドバー</li>
                  <li>メインコンテンツ: 最大768px幅で中央配置</li>
                </ul>
                <div className="text-xs text-gray-400 mt-1">
                  {new Date().toLocaleTimeString('ja-JP')}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Input */}
      <div className="border-t border-border p-4">
        <div className="main-content">
          <div className="flex space-x-2">
            <input
              type="text"
              placeholder="メッセージを入力してください..."
              className="input-primary flex-1"
            />
            <button className="btn-primary">送信</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainContent;
