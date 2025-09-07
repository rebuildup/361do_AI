/**
 * NaturalLanguageIndicator Component Tests
 *
 * Tests for the natural language processing indicator component
 */

import React from 'react';
import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '@/test/utils';
import { NaturalLanguageIndicator } from '../NaturalLanguageIndicator';

describe('NaturalLanguageIndicator', () => {
  describe('Rendering Conditions', () => {
    it('should not render when no props are provided', () => {
      const { container } = renderWithProviders(<NaturalLanguageIndicator />);

      expect(container.firstChild).toBeNull();
    });

    it('should render when processing', () => {
      renderWithProviders(<NaturalLanguageIndicator isProcessing={true} />);

      expect(
        screen.getByText(/自然言語処理|Natural Language Processing/)
      ).toBeInTheDocument();
    });

    it('should render when tools are used', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          toolsUsed={['web_search', 'file_operations']}
        />
      );

      expect(
        screen.getByText(/自然言語処理|Natural Language Processing/)
      ).toBeInTheDocument();
    });

    it('should render when reasoning is provided', () => {
      renderWithProviders(
        <NaturalLanguageIndicator reasoning="Test reasoning process" />
      );

      expect(
        screen.getByText(/自然言語処理|Natural Language Processing/)
      ).toBeInTheDocument();
    });
  });

  describe('Language Display', () => {
    it('should show Japanese language indicator', () => {
      renderWithProviders(
        <NaturalLanguageIndicator isProcessing={true} language="ja" />
      );

      expect(screen.getByText('JA')).toBeInTheDocument();
      expect(screen.getByText('自然言語処理')).toBeInTheDocument();
    });

    it('should show English language indicator', () => {
      renderWithProviders(
        <NaturalLanguageIndicator isProcessing={true} language="en" />
      );

      expect(screen.getByText('EN')).toBeInTheDocument();
      expect(
        screen.getByText('Natural Language Processing')
      ).toBeInTheDocument();
    });
  });

  describe('Tools Display', () => {
    it('should display used tools with Japanese labels', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          toolsUsed={['web_search', 'file_operations']}
          language="ja"
        />
      );

      expect(screen.getByText('使用ツール:')).toBeInTheDocument();
      expect(screen.getByText('Web検索')).toBeInTheDocument();
      expect(screen.getByText('ファイル操作')).toBeInTheDocument();
    });

    it('should display used tools with English labels', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          toolsUsed={['command_execution', 'mcp_integration']}
          language="en"
        />
      );

      expect(screen.getByText('Tools Used:')).toBeInTheDocument();
      expect(screen.getByText('Command Execution')).toBeInTheDocument();
      expect(screen.getByText('MCP Integration')).toBeInTheDocument();
    });

    it('should handle unknown tools gracefully', () => {
      renderWithProviders(
        <NaturalLanguageIndicator toolsUsed={['unknown_tool']} language="ja" />
      );

      expect(screen.getByText('unknown_tool')).toBeInTheDocument();
    });

    it('should show all tool icons', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          toolsUsed={[
            'web_search',
            'command_execution',
            'file_operations',
            'mcp_integration',
          ]}
        />
      );

      // Should render tool badges
      expect(screen.getByText('Web検索')).toBeInTheDocument();
      expect(screen.getByText('コマンド実行')).toBeInTheDocument();
      expect(screen.getByText('ファイル操作')).toBeInTheDocument();
      expect(screen.getByText('MCP連携')).toBeInTheDocument();
    });
  });

  describe('Confidence Score', () => {
    it('should display high confidence with green bar', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          confidence={0.9}
          language="ja"
          isProcessing={true}
        />
      );

      expect(screen.getByText('信頼度:')).toBeInTheDocument();
      expect(screen.getByText('90%')).toBeInTheDocument();

      const progressBar =
        screen.getByText('90%').previousElementSibling?.firstElementChild;
      expect(progressBar).toHaveClass('bg-green-400');
    });

    it('should display medium confidence with yellow bar', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          confidence={0.7}
          language="en"
          isProcessing={true}
        />
      );

      expect(screen.getByText('Confidence:')).toBeInTheDocument();
      expect(screen.getByText('70%')).toBeInTheDocument();

      const progressBar =
        screen.getByText('70%').previousElementSibling?.firstElementChild;
      expect(progressBar).toHaveClass('bg-yellow-400');
    });

    it('should display low confidence with red bar', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          confidence={0.3}
          language="ja"
          isProcessing={true}
        />
      );

      expect(screen.getByText('30%')).toBeInTheDocument();

      const progressBar =
        screen.getByText('30%').previousElementSibling?.firstElementChild;
      expect(progressBar).toHaveClass('bg-red-400');
    });

    it('should handle edge confidence values', () => {
      renderWithProviders(
        <NaturalLanguageIndicator confidence={0} isProcessing={true} />
      );

      expect(screen.getByText('0%')).toBeInTheDocument();

      renderWithProviders(
        <NaturalLanguageIndicator confidence={1} isProcessing={true} />
      );

      expect(screen.getByText('100%')).toBeInTheDocument();
    });
  });

  describe('Reasoning Display', () => {
    it('should display reasoning in Japanese', () => {
      const reasoning = 'ユーザーの質問を分析し、適切なツールを選択しました。';

      renderWithProviders(
        <NaturalLanguageIndicator reasoning={reasoning} language="ja" />
      );

      expect(screen.getByText('推論過程:')).toBeInTheDocument();
      expect(screen.getByText(reasoning)).toBeInTheDocument();
    });

    it('should display reasoning in English', () => {
      const reasoning = 'Analyzed user query and selected appropriate tools.';

      renderWithProviders(
        <NaturalLanguageIndicator reasoning={reasoning} language="en" />
      );

      expect(screen.getByText('Reasoning:')).toBeInTheDocument();
      expect(screen.getByText(reasoning)).toBeInTheDocument();
    });

    it('should handle multiline reasoning', () => {
      const reasoning =
        'Step 1: Analyze input\nStep 2: Select tools\nStep 3: Generate response';

      renderWithProviders(
        <NaturalLanguageIndicator reasoning={reasoning} language="en" />
      );

      expect(screen.getByText(reasoning)).toBeInTheDocument();
    });

    it('should handle long reasoning text', () => {
      const longReasoning = 'A'.repeat(500);

      renderWithProviders(
        <NaturalLanguageIndicator reasoning={longReasoning} language="ja" />
      );

      expect(screen.getByText(longReasoning)).toBeInTheDocument();
    });
  });

  describe('Processing Animation', () => {
    it('should show processing animation when active', () => {
      renderWithProviders(
        <NaturalLanguageIndicator isProcessing={true} language="ja" />
      );

      expect(screen.getByText('自然言語を解析中...')).toBeInTheDocument();

      // Should have animated dots
      const dots = screen
        .getAllByRole('generic')
        .filter(el => el.className.includes('animate-bounce'));
      expect(dots.length).toBeGreaterThan(0);
    });

    it('should show processing animation in English', () => {
      renderWithProviders(
        <NaturalLanguageIndicator isProcessing={true} language="en" />
      );

      expect(
        screen.getByText('Analyzing natural language...')
      ).toBeInTheDocument();
    });

    it('should not show processing animation when not processing', () => {
      renderWithProviders(
        <NaturalLanguageIndicator toolsUsed={['web_search']} language="ja" />
      );

      expect(screen.queryByText('自然言語を解析中...')).not.toBeInTheDocument();
    });
  });

  describe('Combined States', () => {
    it('should display all elements when fully populated', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          isProcessing={true}
          language="ja"
          toolsUsed={['web_search', 'file_operations']}
          confidence={0.85}
          reasoning="完全なテスト推論プロセス"
        />
      );

      expect(screen.getByText('自然言語処理')).toBeInTheDocument();
      expect(screen.getByText('JA')).toBeInTheDocument();
      expect(screen.getByText('Web検索')).toBeInTheDocument();
      expect(screen.getByText('ファイル操作')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
      expect(screen.getByText('完全なテスト推論プロセス')).toBeInTheDocument();
      expect(screen.getByText('自然言語を解析中...')).toBeInTheDocument();
    });

    it('should handle partial state combinations', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          toolsUsed={['web_search']}
          confidence={0.6}
          language="en"
        />
      );

      expect(
        screen.getByText('Natural Language Processing')
      ).toBeInTheDocument();
      expect(screen.getByText('Web Search')).toBeInTheDocument();
      expect(screen.getByText('60%')).toBeInTheDocument();
      expect(
        screen.queryByText('Analyzing natural language...')
      ).not.toBeInTheDocument();
    });
  });

  describe('Styling and Layout', () => {
    it('should have proper CSS classes', () => {
      const { container } = renderWithProviders(
        <NaturalLanguageIndicator
          isProcessing={true}
          className="custom-class"
        />
      );

      const indicator = container.firstChild as HTMLElement;
      expect(indicator).toHaveClass('custom-class');
      expect(indicator).toHaveClass('bg-gray-900/80');
      expect(indicator).toHaveClass('border-gray-700');
    });

    it('should be responsive', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          toolsUsed={['web_search', 'file_operations', 'command_execution']}
        />
      );

      // Should handle multiple tools in a responsive layout
      expect(screen.getByText('Web検索')).toBeInTheDocument();
      expect(screen.getByText('ファイル操作')).toBeInTheDocument();
      expect(screen.getByText('コマンド実行')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper semantic structure', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          isProcessing={true}
          reasoning="Test reasoning"
        />
      );

      // Should have proper heading structure and content organization
      expect(
        screen.getByText(/自然言語処理|Natural Language Processing/)
      ).toBeInTheDocument();
    });

    it('should support screen readers', () => {
      renderWithProviders(
        <NaturalLanguageIndicator
          confidence={0.8}
          toolsUsed={['web_search']}
          language="en"
        />
      );

      // Content should be readable by screen readers
      expect(screen.getByText('Confidence:')).toBeInTheDocument();
      expect(screen.getByText('Tools Used:')).toBeInTheDocument();
    });
  });
});
