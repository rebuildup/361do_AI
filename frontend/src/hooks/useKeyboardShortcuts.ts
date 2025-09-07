import { useEffect, useCallback } from 'react';

interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;
  callback: (event: KeyboardEvent) => void;
  preventDefault?: boolean;
  description?: string;
}

interface UseKeyboardShortcutsOptions {
  enabled?: boolean;
  target?: HTMLElement | Document;
}

export const useKeyboardShortcuts = (
  shortcuts: KeyboardShortcut[],
  options: UseKeyboardShortcutsOptions = {}
) => {
  const { enabled = true, target = document } = options;

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      // Don't trigger shortcuts when typing in input fields
      const activeElement = document.activeElement as HTMLElement | null;
      if (
        activeElement &&
        (activeElement.tagName === 'INPUT' ||
          activeElement.tagName === 'TEXTAREA' ||
          (activeElement as HTMLElement).isContentEditable)
      ) {
        return;
      }

      for (const shortcut of shortcuts) {
        const {
          key,
          ctrlKey = false,
          shiftKey = false,
          altKey = false,
          metaKey = false,
          callback,
          preventDefault = true,
        } = shortcut;

        const keyMatches = event.key.toLowerCase() === key.toLowerCase();
        const ctrlMatches = event.ctrlKey === ctrlKey;
        const shiftMatches = event.shiftKey === shiftKey;
        const altMatches = event.altKey === altKey;
        const metaMatches = event.metaKey === metaKey;

        if (
          keyMatches &&
          ctrlMatches &&
          shiftMatches &&
          altMatches &&
          metaMatches
        ) {
          if (preventDefault) {
            event.preventDefault();
          }
          callback(event);
          break;
        }
      }
    },
    [shortcuts, enabled]
  );

  useEffect(() => {
    if (!enabled) return;

    const eventTarget: Document | HTMLElement = target ?? document;
    eventTarget.addEventListener('keydown', handleKeyDown as EventListener);
    return () =>
      eventTarget.removeEventListener(
        'keydown',
        handleKeyDown as EventListener
      );
  }, [handleKeyDown, enabled, target]);
};

// Predefined shortcuts for common actions
export const createChatShortcuts = (actions: {
  onNewChat?: () => void;
  onClearChat?: () => void;
  onToggleSidebar?: () => void;
  onFocusInput?: () => void;
  onCopyLastMessage?: () => void;
  onShowShortcuts?: () => void;
}) => {
  const shortcuts: KeyboardShortcut[] = [];

  if (actions.onNewChat) {
    shortcuts.push({
      key: 'n',
      ctrlKey: true,
      callback: actions.onNewChat,
      description: 'Ctrl+N: 新しいチャット',
    });
  }

  if (actions.onClearChat) {
    shortcuts.push({
      key: 'k',
      ctrlKey: true,
      shiftKey: true,
      callback: actions.onClearChat,
      description: 'Ctrl+Shift+K: チャットをクリア',
    });
  }

  if (actions.onToggleSidebar) {
    shortcuts.push({
      key: 'b',
      ctrlKey: true,
      callback: actions.onToggleSidebar,
      description: 'Ctrl+B: サイドバーの切り替え',
    });
  }

  if (actions.onFocusInput) {
    shortcuts.push({
      key: '/',
      callback: actions.onFocusInput,
      description: '/: 入力フィールドにフォーカス',
    });
  }

  if (actions.onCopyLastMessage) {
    shortcuts.push({
      key: 'c',
      ctrlKey: true,
      shiftKey: true,
      callback: actions.onCopyLastMessage,
      description: 'Ctrl+Shift+C: 最後のメッセージをコピー',
    });
  }

  if (actions.onShowShortcuts) {
    shortcuts.push({
      key: '?',
      shiftKey: true,
      callback: actions.onShowShortcuts,
      description: '?: ショートカット一覧を表示',
    });
  }

  return shortcuts;
};
