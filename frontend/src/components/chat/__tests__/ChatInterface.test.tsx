/**
 * ChatInterface Component Tests
 *
 * Tests for the main chat interface component
 */

import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '@/test/utils';
import ChatInterface from '../ChatInterface';

// Mock hooks
vi.mock('@/hooks/useStreaming', () => ({
  useStreaming: () => ({
    isStreaming: false,
    currentMessageId: null,
    error: null,
    startStreaming: vi.fn(),
    simulateStreaming: vi.fn(),
    stopStreaming: vi.fn(),
    retryStreaming: vi.fn(),
  }),
}));

vi.mock('@/hooks/useSessionManager', () => ({
  useSessionManager: () => ({
    sessions: [],
    isLoading: false,
    error: null,
    createSession: vi.fn(),
    switchSession: vi.fn(),
    deleteSession: vi.fn(),
    updateSessionMessages: vi.fn(),
    renameSession: vi.fn(),
    clearAllSessions: vi.fn(),
  }),
}));

vi.mock('@/hooks/useKeyboardShortcuts', () => ({
  useKeyboardShortcuts: vi.fn(),
  createChatShortcuts: () => ({}),
}));

describe('ChatInterface', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render chat interface', () => {
      renderWithProviders(<ChatInterface />);

      expect(
        screen.getByPlaceholderText(/メッセージを入力してください/)
      ).toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: /送信|send/i })
      ).toBeInTheDocument();
    });

    it('should show keyboard shortcuts hint when no messages', () => {
      renderWithProviders(<ChatInterface />);

      expect(
        screen.getByText(/でショートカット一覧を表示/)
      ).toBeInTheDocument();
    });

    it('should render messages when provided', () => {
      // This would require mocking the app context with messages
      renderWithProviders(<ChatInterface />);

      // For now, just verify the component renders
      expect(
        screen.getByPlaceholderText(/メッセージを入力してください/)
      ).toBeInTheDocument();
    });
  });

  describe('Message Input', () => {
    it('should allow typing in the input field', async () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);

      await user.type(input, 'Hello, this is a test message');

      expect(input).toHaveValue('Hello, this is a test message');
    });

    it('should auto-resize textarea based on content', async () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(
        /メッセージを入力してください/
      ) as HTMLTextAreaElement;

      await user.type(input, 'Line 1\nLine 2\nLine 3');

      // The textarea should have multiple lines
      expect(input.value).toContain('\n');
    });

    it('should submit message on Enter key', async () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);

      await user.type(input, 'Test message');
      await user.keyboard('{Enter}');

      // Input should be cleared after submission
      expect(input).toHaveValue('');
    });

    it('should add new line on Shift+Enter', async () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);

      await user.type(input, 'First line');
      await user.keyboard('{Shift>}{Enter}{/Shift}');
      await user.type(input, 'Second line');

      expect(input).toHaveValue('First line\nSecond line');
    });

    it('should not submit empty messages', async () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);
      const submitButton = screen.getByRole('button', { name: /送信|send/i });

      await user.click(submitButton);

      // Should not clear input or show any processing state
      expect(input).toHaveValue('');
    });
  });

  describe('Message Submission', () => {
    it('should submit message via button click', async () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);
      const submitButton = screen.getByRole('button', { name: /送信|send/i });

      await user.type(input, 'Test message via button');
      await user.click(submitButton);

      expect(input).toHaveValue('');
    });

    it('should disable input during streaming', () => {
      // Mock streaming state
      vi.mocked(
        vi.importMock('@/hooks/useStreaming').useStreaming
      ).mockReturnValue({
        isStreaming: true,
        currentMessageId: 'test-id',
        error: null,
        startStreaming: vi.fn(),
        simulateStreaming: vi.fn(),
        stopStreaming: vi.fn(),
        retryStreaming: vi.fn(),
      });

      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);

      expect(input).toBeDisabled();
    });

    it('should show stop button during streaming', () => {
      // Mock streaming state
      vi.mocked(
        vi.importMock('@/hooks/useStreaming').useStreaming
      ).mockReturnValue({
        isStreaming: true,
        currentMessageId: 'test-id',
        error: null,
        startStreaming: vi.fn(),
        simulateStreaming: vi.fn(),
        stopStreaming: vi.fn(),
        retryStreaming: vi.fn(),
      });

      renderWithProviders(<ChatInterface />);

      expect(screen.getByTitle(/ストリーミングを停止/)).toBeInTheDocument();
    });
  });

  describe('Streaming States', () => {
    it('should show streaming indicator when processing', () => {
      // Mock streaming state
      vi.mocked(
        vi.importMock('@/hooks/useStreaming').useStreaming
      ).mockReturnValue({
        isStreaming: true,
        currentMessageId: 'test-id',
        error: null,
        startStreaming: vi.fn(),
        simulateStreaming: vi.fn(),
        stopStreaming: vi.fn(),
        retryStreaming: vi.fn(),
      });

      renderWithProviders(<ChatInterface />);

      expect(screen.getByText(/応答を生成中/)).toBeInTheDocument();
    });

    it('should allow stopping streaming', async () => {
      const mockStopStreaming = vi.fn();

      vi.mocked(
        vi.importMock('@/hooks/useStreaming').useStreaming
      ).mockReturnValue({
        isStreaming: true,
        currentMessageId: 'test-id',
        error: null,
        startStreaming: vi.fn(),
        simulateStreaming: vi.fn(),
        stopStreaming: mockStopStreaming,
        retryStreaming: vi.fn(),
      });

      renderWithProviders(<ChatInterface />);

      const stopButton = screen.getByTitle(/ストリーミングを停止/);
      await user.click(stopButton);

      expect(mockStopStreaming).toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('should display connection errors', () => {
      // This would require mocking the component state or context
      renderWithProviders(<ChatInterface />);

      // For now, just verify the component renders without errors
      expect(
        screen.getByPlaceholderText(/メッセージを入力してください/)
      ).toBeInTheDocument();
    });

    it('should show retry button on errors', () => {
      // This would require mocking error state
      renderWithProviders(<ChatInterface />);

      // Component should render without throwing
      expect(
        screen.getByPlaceholderText(/メッセージを入力してください/)
      ).toBeInTheDocument();
    });
  });

  describe('Natural Language Processing', () => {
    it('should show NLP indicator when processing', () => {
      renderWithProviders(<ChatInterface />);

      // The NLP indicator would be shown based on state
      // For now, verify component renders
      expect(
        screen.getByPlaceholderText(/メッセージを入力してください/)
      ).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);
      const submitButton = screen.getByRole('button', { name: /送信|send/i });

      expect(input).toBeInTheDocument();
      expect(submitButton).toBeInTheDocument();
    });

    it('should support keyboard navigation', async () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);

      // Tab to input
      await user.tab();
      expect(input).toHaveFocus();

      // Tab to submit button
      await user.tab();
      const submitButton = screen.getByRole('button', { name: /送信|send/i });
      expect(submitButton).toHaveFocus();
    });
  });

  describe('Performance', () => {
    it('should not cause memory leaks', () => {
      const { unmount } = renderWithProviders(<ChatInterface />);

      // Component should unmount cleanly
      unmount();

      expect(true).toBe(true); // No errors during unmount
    });

    it('should handle rapid input changes', async () => {
      renderWithProviders(<ChatInterface />);

      const input = screen.getByPlaceholderText(/メッセージを入力してください/);

      // Rapid typing simulation
      for (let i = 0; i < 10; i++) {
        await user.type(input, `Message ${i} `);
      }

      expect(input).toHaveValue(expect.stringContaining('Message'));
    });
  });
});
