/**
 * Test Utilities
 *
 * Common utilities and helpers for testing React components
 */

import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { vi } from 'vitest';
import { AppProvider } from '@/contexts/AppContext';
import { ToastProvider } from '@/components/ui/Toast';

// Mock API service
export const mockApiService = {
  healthCheck: vi.fn(),
  getModels: vi.fn(),
  chatCompletion: vi.fn(),
  agentChatCompletion: vi.fn(),
  streamChatCompletion: vi.fn(),
  processUserInput: vi.fn(),
  getAgentStatus: vi.fn(),
  updateAgentConfig: vi.fn(),
  getAvailableTools: vi.fn(),
  executeTool: vi.fn(),
  getLearningMetrics: vi.fn(),
  provideFeedback: vi.fn(),
  createSession: vi.fn(),
  getSession: vi.fn(),
  getConversationHistory: vi.fn(),
  clearConversationHistory: vi.fn(),
  exportConversation: vi.fn(),
  getSystemStats: vi.fn(),
  testConnection: vi.fn(),
  setApiKey: vi.fn(),
  getApiKey: vi.fn(),
};

// Mock natural language processor
export const mockNaturalLanguageProcessor = {
  processInput: vi.fn(),
  processStreamingInput: vi.fn(),
  clearConversationContext: vi.fn(),
  getConversationContext: vi.fn(),
};

// Mock session persistence
export const mockSessionPersistence = {
  createOrRestoreSession: vi.fn(),
  getActiveSessionForUser: vi.fn(),
  saveSession: vi.fn(),
  addMessageToSession: vi.fn(),
  updateSessionMessages: vi.fn(),
  getSession: vi.fn(),
  getAllSessions: vi.fn(),
  getSessionSummaries: vi.fn(),
  deleteSession: vi.fn(),
  clearAllSessions: vi.fn(),
  exportSession: vi.fn(),
  importSession: vi.fn(),
};

// Custom render function with providers
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialState?: any;
}

const AllTheProviders: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return (
    <AppProvider>
      <ToastProvider>{children}</ToastProvider>
    </AppProvider>
  );
};

export const renderWithProviders = (
  ui: ReactElement,
  options?: CustomRenderOptions
) => {
  return render(ui, { wrapper: AllTheProviders, ...options });
};

// Mock message data
export const mockMessage = {
  user: {
    id: 'msg-1',
    role: 'user' as const,
    content: 'Hello, this is a test message',
    timestamp: new Date('2024-01-01T10:00:00Z'),
  },
  assistant: {
    id: 'msg-2',
    role: 'assistant' as const,
    content: 'Hello! I understand your test message. How can I help you today?',
    timestamp: new Date('2024-01-01T10:00:01Z'),
    reasoning:
      'The user sent a greeting message, so I should respond politely and offer assistance.',
  },
  streaming: {
    id: 'msg-3',
    role: 'assistant' as const,
    content: 'This is a streaming response...',
    timestamp: new Date('2024-01-01T10:00:02Z'),
    isStreaming: true,
  },
  error: {
    id: 'msg-4',
    role: 'assistant' as const,
    content: '',
    timestamp: new Date('2024-01-01T10:00:03Z'),
    error: 'Connection failed',
  },
};

// Mock session data
export const mockSession = {
  session_id: 'test-session-1',
  created_at: '2024-01-01T10:00:00Z',
  user_id: 'test-user',
  session_name: 'Test Session',
  metadata: {
    created_at: '2024-01-01T10:00:00Z',
    lastActivity: '2024-01-01T10:00:00Z',
    messageCount: 2,
    language: 'ja' as const,
  },
};

// Mock persistent session
export const mockPersistentSession = {
  sessionId: 'test-session-1',
  userId: 'test-user',
  sessionName: 'Test Session',
  messages: [mockMessage.user, mockMessage.assistant],
  metadata: {
    createdAt: '2024-01-01T10:00:00Z',
    lastActivity: '2024-01-01T10:00:00Z',
    messageCount: 2,
    language: 'ja' as const,
  },
};

// Mock API responses
export const mockApiResponses = {
  healthCheck: {
    status: 'healthy',
    timestamp: new Date(),
    version: '1.0.0',
    system_info: {
      cpu_percent: 50.0,
      memory_percent: 60.0,
    },
  },
  models: {
    object: 'list',
    data: [
      {
        id: 'deepseek-r1:7b',
        object: 'model',
        created: 1640995200,
        owned_by: 'advanced-agent',
      },
      {
        id: 'qwen2.5:7b-instruct-q4_k_m',
        object: 'model',
        created: 1640995200,
        owned_by: 'advanced-agent',
      },
    ],
  },
  chatCompletion: {
    id: 'chatcmpl-123',
    object: 'chat.completion',
    created: 1640995200,
    model: 'deepseek-r1:7b',
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: 'Hello! How can I help you today?',
        },
        finish_reason: 'stop',
      },
    ],
    usage: {
      prompt_tokens: 10,
      completion_tokens: 8,
      total_tokens: 18,
    },
  },
  agentStatus: {
    status: 'active',
    capabilities: [
      'natural_language_processing',
      'web_search',
      'file_operations',
      'command_execution',
      'self_learning',
    ],
    active_tools: ['web_search', 'file_operations'],
    memory_usage: {
      gpu_memory_mb: 100,
      cpu_memory_mb: 50,
    },
    learning_metrics: {
      learning_epoch: 5,
      total_interactions: 100,
      reward_score: 0.85,
    },
  },
  learningMetrics: {
    total_interactions: 100,
    successful_responses: 85,
    average_response_time: 1.5,
    learning_rate: 0.01,
    recent_improvements: [
      {
        timestamp: '2024-01-01T10:00:00Z',
        improvement_type: 'response_quality',
        description: 'Improved natural language understanding',
      },
    ],
  },
};

// Test data generators
export const generateMockMessages = (count: number) => {
  const messages = [];
  for (let i = 0; i < count; i++) {
    messages.push({
      id: `msg-${i}`,
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `Test message ${i}`,
      timestamp: new Date(Date.now() + i * 1000),
    });
  }
  return messages;
};

// Wait for async operations
export const waitFor = (ms: number) =>
  new Promise(resolve => setTimeout(resolve, ms));

// Mock fetch responses
export const mockFetch = (response: any, ok = true, status = 200) => {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: () => Promise.resolve(response),
    text: () => Promise.resolve(JSON.stringify(response)),
  });
};

// Mock streaming response
export const mockStreamingResponse = (chunks: string[]) => {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      chunks.forEach((chunk, index) => {
        setTimeout(() => {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ content: chunk })}\n\n`)
          );
          if (index === chunks.length - 1) {
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
          }
        }, index * 100);
      });
    },
  });

  return {
    ok: true,
    status: 200,
    body: stream,
  };
};

// Custom matchers
export const customMatchers = {
  toBeInTheDocument: (received: any) => {
    const pass = received !== null && received !== undefined;
    return {
      message: () =>
        `expected element ${pass ? 'not ' : ''}to be in the document`,
      pass,
    };
  },
};

// Re-export everything from testing library
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
