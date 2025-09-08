/**
 * Natural Language Processing Indicator
 *
 * Shows when the agent is processing natural language and using tools
 */

import React from 'react';
import {
  Brain,
  Search,
  Terminal,
  FileText,
  Plug,
  Zap,
  Languages,
  TrendingUp,
} from 'lucide-react';
import { cn } from '@/utils';

interface NLPIndicatorProps {
  isProcessing?: boolean;
  language?: 'ja' | 'en';
  toolsUsed?: string[];
  confidence?: number;
  reasoning?: string;
  className?: string;
}

const toolIcons: Record<
  string,
  React.ComponentType<{ size?: number; className?: string }>
> = {
  web_search: Search,
  command_execution: Terminal,
  file_operations: FileText,
  mcp_integration: Plug,
};

const toolLabels: Record<string, { ja: string; en: string }> = {
  web_search: { ja: 'Web検索', en: 'Web Search' },
  command_execution: { ja: 'コマンド実行', en: 'Command Execution' },
  file_operations: { ja: 'ファイル操作', en: 'File Operations' },
  mcp_integration: { ja: 'MCP連携', en: 'MCP Integration' },
};

export const NaturalLanguageIndicator: React.FC<NLPIndicatorProps> = ({
  isProcessing = false,
  language = 'ja',
  toolsUsed = [],
  confidence,
  reasoning,
  className,
}) => {
  if (!isProcessing && toolsUsed.length === 0 && !reasoning) {
    return null;
  }

  return (
    <div
      className={cn(
        'rounded-lg p-3 backdrop-blur-sm',
        'bg-[var(--color-background-secondary)] border border-[var(--color-border)]',
        'animate-fade-in',
        className
      )}
    >
      {/* Processing Header */}
      <div className="flex items-center gap-2 mb-2">
        <Brain
          size={16}
          className={cn(
            'text-[var(--color-accent)]',
            isProcessing && 'animate-pulse'
          )}
        />
        <span className="text-sm font-medium text-[var(--color-text)]">
          {language === 'ja' ? '自然言語処理' : 'Natural Language Processing'}
        </span>

        {/* Language Indicator */}
        <div className="flex items-center gap-1 ml-auto">
          <Languages
            size={12}
            className="text-[var(--color-text-secondary)] opacity-70"
          />
          <span className="text-xs text-[var(--color-text-secondary)] opacity-70 uppercase">
            {language}
          </span>
        </div>
      </div>

      {/* Tools Used */}
      {toolsUsed.length > 0 && (
        <div className="mb-2">
          <div className="text-xs text-[var(--color-text-secondary)] opacity-80 mb-1">
            {language === 'ja' ? '使用ツール:' : 'Tools Used:'}
          </div>
          <div className="flex flex-wrap gap-1">
            {toolsUsed.map((tool, index) => {
              const IconComponent = toolIcons[tool] || Zap;
              const label = toolLabels[tool]?.[language] || tool;

              return (
                <div
                  key={index}
                  className="flex items-center gap-1 px-2 py-1 rounded text-xs"
                  style={{
                    backgroundColor: 'var(--color-background-tertiary)',
                    color: 'var(--color-text)',
                  }}
                >
                  <IconComponent
                    size={12}
                    className="text-[var(--color-accent)]"
                  />
                  <span>{label}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Confidence Score */}
      {confidence !== undefined && (
        <div className="mb-2">
          <div className="flex items-center gap-2">
            <TrendingUp size={12} className="text-green-400" />
            <span className="text-xs text-gray-400">
              {language === 'ja' ? '信頼度:' : 'Confidence:'}
            </span>
            <div className="flex-1 bg-gray-800 rounded-full h-1.5">
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-300',
                  confidence >= 0.8
                    ? 'bg-green-400'
                    : confidence >= 0.6
                      ? 'bg-yellow-400'
                      : 'bg-red-400'
                )}
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
            <span className="text-xs text-gray-300 font-mono">
              {(confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}

      {/* Reasoning */}
      {reasoning && (
        <div className="text-xs text-gray-400 bg-gray-800/50 rounded p-2 border-l-2 border-blue-400">
          <div className="font-medium mb-1">
            {language === 'ja' ? '推論過程:' : 'Reasoning:'}
          </div>
          <div className="whitespace-pre-wrap">{reasoning}</div>
        </div>
      )}

      {/* Processing Animation */}
      {isProcessing && (
        <div className="flex items-center gap-2 mt-2 text-xs text-gray-400">
          <div className="flex gap-1">
            <div
              className="w-1 h-1 bg-blue-400 rounded-full animate-bounce"
              style={{ animationDelay: '0ms' }}
            />
            <div
              className="w-1 h-1 bg-blue-400 rounded-full animate-bounce"
              style={{ animationDelay: '150ms' }}
            />
            <div
              className="w-1 h-1 bg-blue-400 rounded-full animate-bounce"
              style={{ animationDelay: '300ms' }}
            />
          </div>
          <span className="animate-pulse">
            {language === 'ja'
              ? '自然言語を解析中...'
              : 'Analyzing natural language...'}
          </span>
        </div>
      )}
    </div>
  );
};

export default NaturalLanguageIndicator;
