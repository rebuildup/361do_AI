import React from 'react';
import { Keyboard } from 'lucide-react';
import Modal from './Modal';

interface Shortcut {
  key: string;
  description: string;
}

interface ShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
  shortcuts?: Shortcut[];
}

const defaultShortcuts: Shortcut[] = [
  { key: 'Ctrl+N', description: '新しいチャットを開始' },
  { key: 'Ctrl+Shift+K', description: 'チャット履歴をクリア' },
  { key: 'Ctrl+B', description: 'サイドバーの表示/非表示' },
  { key: '/', description: '入力フィールドにフォーカス' },
  { key: 'Ctrl+Shift+C', description: '最後のメッセージをコピー' },
  { key: 'Enter', description: 'メッセージを送信' },
  { key: 'Shift+Enter', description: '改行を挿入' },
  { key: 'Escape', description: 'モーダルを閉じる' },
  { key: '?', description: 'このヘルプを表示' },
];

const ShortcutsModal: React.FC<ShortcutsModalProps> = ({
  isOpen,
  onClose,
  shortcuts = defaultShortcuts,
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="キーボードショートカット"
      size="md"
    >
      <div className="space-y-4">
        <div className="flex items-center space-x-2 text-gray-300">
          <Keyboard size={20} />
          <p>利用可能なキーボードショートカット:</p>
        </div>

        <div className="space-y-2">
          {shortcuts.map((shortcut, index) => (
            <div
              key={index}
              className="flex items-center justify-between py-2 px-3 rounded-lg bg-gray-800/50 hover:bg-gray-800 transition-colors"
            >
              <span className="text-gray-300">{shortcut.description}</span>
              <kbd className="px-2 py-1 text-xs font-mono bg-gray-700 border border-gray-600 rounded text-gray-200">
                {shortcut.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="pt-4 border-t border-gray-700">
          <p className="text-sm text-gray-400">
            ヒント:
            入力フィールドにフォーカスがある時は、一部のショートカットが無効になります。
          </p>
        </div>
      </div>
    </Modal>
  );
};

export default ShortcutsModal;
