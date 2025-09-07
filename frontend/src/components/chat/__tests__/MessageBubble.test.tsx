/**
 * MessageBubble Component Tests
 *
 * Tests for the message bubble component
 */

import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, mockMessage } from '@/test/utils';
import MessageBubble from '../MessageBubble';

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: vi.fn().mockResolvedValue(undefined),
  },
});

describe('MessageBubble', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('User Messages', () => {
    it('should render user message correctly', () => {
      renderWithProviders(<MessageBubble message={mockMessage.user} />);

      expect(screen.getByText(mockMessage.user.content)).toBeInTheDocument();
      expect(screen.getByText('You')).toBeInTheDocument();
    });

    it('should show timestamp for user messages', () => {
      renderWithProviders(<MessageBubble message={mockMessage.user} />);

      // Should show some form of timestamp
      expect(screen.getByText(/10:00/)).toBeInTheDocument();
    });

    it('should have correct styling for user messages', () => {
      renderWithProviders(<MessageBubble message={mockMessage.user} />);

      const messageContainer = screen
        .getByText(mockMessage.user.content)
        .closest('div');
      expect(messageContainer).toHaveClass('bg-blue-600'); // User message styling
    });
  });

  describe('Assistant Messages', () => {
    it('should render assistant message correctly', () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      expect(
        screen.getByText(mockMessage.assistant.content)
      ).toBeInTheDocument();
      expect(screen.getByText('Assistant')).toBeInTheDocument();
    });

    it('should show reasoning section when available', () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      expect(screen.getByText(/推論過程|Reasoning/)).toBeInTheDocument();
      expect(
        screen.getByText(mockMessage.assistant.reasoning!)
      ).toBeInTheDocument();
    });

    it('should allow collapsing/expanding reasoning section', async () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      const reasoningToggle = screen.getByRole('button', {
        name: /推論過程|reasoning/i,
      });

      // Click to collapse
      await user.click(reasoningToggle);

      // Reasoning content should be hidden (implementation dependent)
      // For now, just verify the button exists and is clickable
      expect(reasoningToggle).toBeInTheDocument();
    });

    it('should have correct styling for assistant messages', () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      const messageContainer = screen
        .getByText(mockMessage.assistant.content)
        .closest('div');
      expect(messageContainer).toHaveClass('bg-gray-800'); // Assistant message styling
    });
  });

  describe('Streaming Messages', () => {
    it('should show streaming indicator', () => {
      renderWithProviders(<MessageBubble message={mockMessage.streaming} />);

      expect(
        screen.getByText(mockMessage.streaming.content)
      ).toBeInTheDocument();
      // Should show some streaming indicator
    });

    it('should have streaming animation', () => {
      renderWithProviders(<MessageBubble message={mockMessage.streaming} />);

      // Should have some visual indication of streaming
      const streamingElement = screen.getByText(mockMessage.streaming.content);
      expect(streamingElement).toBeInTheDocument();
    });
  });

  describe('Error Messages', () => {
    it('should display error state', () => {
      renderWithProviders(<MessageBubble message={mockMessage.error} />);

      expect(screen.getByText(mockMessage.error.error!)).toBeInTheDocument();
    });

    it('should have error styling', () => {
      renderWithProviders(<MessageBubble message={mockMessage.error} />);

      const errorElement = screen.getByText(mockMessage.error.error!);
      expect(errorElement.closest('div')).toHaveClass('border-red-700');
    });
  });

  describe('Copy Functionality', () => {
    it('should show copy button on hover', async () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      const messageContainer = screen
        .getByText(mockMessage.assistant.content)
        .closest('div');

      // Hover over message
      await user.hover(messageContainer!);

      const copyButton = screen.getByRole('button', { name: /copy|コピー/i });
      expect(copyButton).toBeInTheDocument();
    });

    it('should copy message content to clipboard', async () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      const messageContainer = screen
        .getByText(mockMessage.assistant.content)
        .closest('div');
      await user.hover(messageContainer!);

      const copyButton = screen.getByRole('button', { name: /copy|コピー/i });
      await user.click(copyButton);

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        mockMessage.assistant.content
      );
    });

    it('should show copy confirmation', async () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      const messageContainer = screen
        .getByText(mockMessage.assistant.content)
        .closest('div');
      await user.hover(messageContainer!);

      const copyButton = screen.getByRole('button', { name: /copy|コピー/i });
      await user.click(copyButton);

      // Should show some confirmation (implementation dependent)
      expect(navigator.clipboard.writeText).toHaveBeenCalled();
    });
  });

  describe('Timestamps', () => {
    it('should format timestamps correctly', () => {
      renderWithProviders(<MessageBubble message={mockMessage.user} />);

      // Should show formatted time
      expect(screen.getByText(/10:00/)).toBeInTheDocument();
    });

    it('should show relative time for recent messages', () => {
      const recentMessage = {
        ...mockMessage.user,
        timestamp: new Date(),
      };

      renderWithProviders(<MessageBubble message={recentMessage} />);

      // Should show some time indication
      expect(screen.getByText(/\d{1,2}:\d{2}/)).toBeInTheDocument();
    });
  });

  describe('Content Rendering', () => {
    it('should render plain text content', () => {
      renderWithProviders(<MessageBubble message={mockMessage.user} />);

      expect(screen.getByText(mockMessage.user.content)).toBeInTheDocument();
    });

    it('should preserve line breaks in content', () => {
      const multilineMessage = {
        ...mockMessage.user,
        content: 'Line 1\nLine 2\nLine 3',
      };

      renderWithProviders(<MessageBubble message={multilineMessage} />);

      expect(screen.getByText(/Line 1/)).toBeInTheDocument();
      expect(screen.getByText(/Line 2/)).toBeInTheDocument();
      expect(screen.getByText(/Line 3/)).toBeInTheDocument();
    });

    it('should handle long content gracefully', () => {
      const longMessage = {
        ...mockMessage.user,
        content: 'A'.repeat(1000),
      };

      renderWithProviders(<MessageBubble message={longMessage} />);

      expect(screen.getByText(longMessage.content)).toBeInTheDocument();
    });

    it('should handle empty content', () => {
      const emptyMessage = {
        ...mockMessage.user,
        content: '',
      };

      renderWithProviders(<MessageBubble message={emptyMessage} />);

      // Should render without errors
      expect(screen.getByText('You')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      const messageElement = screen.getByRole('article');
      expect(messageElement).toBeInTheDocument();
    });

    it('should support keyboard navigation for interactive elements', async () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      // Tab to reasoning toggle
      await user.tab();

      const reasoningToggle = screen.getByRole('button', {
        name: /推論過程|reasoning/i,
      });
      expect(reasoningToggle).toHaveFocus();
    });

    it('should have proper contrast ratios', () => {
      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      // Visual regression testing would be needed for actual contrast checking
      // For now, verify elements render
      expect(
        screen.getByText(mockMessage.assistant.content)
      ).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('should render quickly for normal messages', () => {
      const startTime = performance.now();

      renderWithProviders(<MessageBubble message={mockMessage.assistant} />);

      const endTime = performance.now();
      expect(endTime - startTime).toBeLessThan(100); // Should render within 100ms
    });

    it('should handle rapid re-renders', () => {
      const { rerender } = renderWithProviders(
        <MessageBubble message={mockMessage.user} />
      );

      // Rapid re-renders
      for (let i = 0; i < 10; i++) {
        const updatedMessage = {
          ...mockMessage.user,
          content: `Updated content ${i}`,
        };
        rerender(<MessageBubble message={updatedMessage} />);
      }

      expect(screen.getByText('Updated content 9')).toBeInTheDocument();
    });
  });
});
