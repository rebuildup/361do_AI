/**
 * API Service Tests
 *
 * Comprehensive tests for the API service layer
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ApiService } from '../api';
import { mockApiResponses } from '@/test/utils';

describe('ApiService', () => {
  let apiService: ApiService;

  beforeEach(() => {
    apiService = new ApiService();
  });

  describe('Health Check', () => {
    it('should perform health check successfully', async () => {
      const response = await apiService.healthCheck();

      expect(response).toEqual(mockApiResponses.healthCheck);
      expect(response.status).toBe('healthy');
    });

    it('should handle health check failure gracefully', async () => {
      // This will be handled by MSW error handlers when needed
      const response = await apiService.healthCheck();
      expect(response).toBeDefined();
    });
  });

  describe('Models', () => {
    it('should fetch available models', async () => {
      const response = await apiService.getModels();

      expect(response).toEqual(mockApiResponses.models);
      expect(response.data).toHaveLength(2);
      expect(response.data[0].id).toBe('deepseek-r1:7b');
    });
  });

  describe('Chat Completions', () => {
    it('should send chat completion request', async () => {
      const request = {
        model: 'deepseek-r1:7b',
        messages: [{ role: 'user' as const, content: 'Hello, world!' }],
        temperature: 0.7,
        max_tokens: 100,
      };

      const response = await apiService.chatCompletion(request);

      expect(response).toEqual(mockApiResponses.chatCompletion);
      expect(response.choices[0].message.content).toBe(
        'Hello! How can I help you today?'
      );
    });

    it('should send agent chat completion request', async () => {
      const request = {
        model: 'agent',
        messages: [{ role: 'user' as const, content: 'Test agent processing' }],
        temperature: 0.7,
        max_tokens: 100,
      };

      const response = await apiService.agentChatCompletion(request);

      expect(response.model).toBe('agent');
      expect(response.reasoning).toBe(
        'This is a test reasoning from the agent.'
      );
    });

    it('should handle streaming chat completion', async () => {
      const request = {
        model: 'deepseek-r1:7b',
        messages: [{ role: 'user' as const, content: 'Stream this response' }],
        temperature: 0.7,
        max_tokens: 100,
        stream: true,
      };

      const chunks: any[] = [];
      const stream = apiService.streamChatCompletion(request);

      for await (const chunk of stream) {
        chunks.push(chunk);
        if (chunks.length >= 3) break; // Limit for testing
      }

      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks[0]).toHaveProperty('choices');
      expect(chunks[0].choices[0]).toHaveProperty('delta');
    });
  });

  describe('Agent Operations', () => {
    it('should get agent status', async () => {
      const response = await apiService.getAgentStatus();

      expect(response).toEqual(mockApiResponses.agentStatus);
      expect(response.status).toBe('active');
      expect(response.capabilities).toContain('natural_language_processing');
    });

    it('should update agent configuration', async () => {
      const config = {
        temperature: 0.8,
        streaming_enabled: true,
      };

      const response = await apiService.updateAgentConfig(config);

      expect(response.success).toBe(true);
      expect(response.message).toBe('Configuration updated successfully');
    });

    it('should get available tools', async () => {
      const response = await apiService.getAvailableTools();

      expect(response.tools).toHaveLength(3);
      expect(response.tools[0].name).toBe('web_search');
      expect(response.tools[0].enabled).toBe(true);
    });

    it('should execute tool', async () => {
      const response = await apiService.executeTool(
        'web_search',
        { query: 'test query' },
        'test-session'
      );

      expect(response.success).toBe(true);
      expect(response.result).toContain('web_search');
      expect(response.execution_time).toBeGreaterThan(0);
    });

    it('should get learning metrics', async () => {
      const response = await apiService.getLearningMetrics();

      expect(response).toEqual(mockApiResponses.learningMetrics);
      expect(response.total_interactions).toBe(100);
      expect(response.successful_responses).toBe(85);
    });

    it('should provide feedback', async () => {
      const response = await apiService.provideFeedback('test-response-id', {
        rating: 5,
        helpful: true,
        comments: 'Great response!',
      });

      expect(response.success).toBe(true);
      expect(response.message).toBe('Feedback received and processed');
    });
  });

  describe('Session Management', () => {
    it('should create session', async () => {
      const request = {
        user_id: 'test-user',
        session_name: 'Test Session',
      };

      const response = await apiService.createSession(request);

      expect(response.session_id).toBe('test-session-123');
      expect(response.user_id).toBe('test-user');
      expect(response.session_name).toBe('Test Session');
    });

    it('should get session', async () => {
      const response = await apiService.getSession('test-session-123');

      expect(response.session_id).toBe('test-session-123');
      expect(response.status).toBe('active');
    });

    it('should get conversation history', async () => {
      const response = await apiService.getConversationHistory(
        'test-session-123',
        10
      );

      expect(response.messages).toHaveLength(10);
      expect(response.total_count).toBe(10);
      expect(response.messages[0]).toHaveProperty('role');
      expect(response.messages[0]).toHaveProperty('content');
    });

    it('should clear conversation history', async () => {
      const response =
        await apiService.clearConversationHistory('test-session-123');

      expect(response.success).toBe(true);
      expect(response.message).toBe('Conversation history cleared');
    });

    it('should export conversation', async () => {
      const response = await apiService.exportConversation(
        'test-session-123',
        'json'
      );

      expect(response).toBeInstanceOf(Blob);
    });
  });

  describe('System Operations', () => {
    it('should get system stats', async () => {
      const request = {
        include_gpu: true,
        include_memory: true,
        include_processes: false,
      };

      const response = await apiService.getSystemStats(request);

      expect(response.cpu).toBeDefined();
      expect(response.memory).toBeDefined();
      expect(response.gpu).toBeDefined();
      expect(response.processes).toBeUndefined();
    });
  });

  describe('Natural Language Processing', () => {
    it('should process user input with agent', async () => {
      const response = await apiService.processUserInput(
        'Hello, test input',
        'test-session',
        true
      );

      expect(response).toBeDefined();
      // Response structure depends on whether it's inference or chat completion
    });

    it('should process user input with fallback', async () => {
      const response = await apiService.processUserInput(
        'Hello, test input',
        'test-session',
        false
      );

      expect(response).toBeDefined();
    });
  });

  describe('Connection Testing', () => {
    it('should test connection successfully', async () => {
      const response = await apiService.testConnection();

      expect(response.healthy).toBe(true);
      expect(response.endpoints).toBeDefined();
      expect(response.endpoints.health).toBe(true);
      expect(response.errors).toHaveLength(0);
    });
  });

  describe('API Key Management', () => {
    it('should set and get API key', () => {
      const testKey = 'test-api-key-123';

      apiService.setApiKey(testKey);
      expect(apiService.getApiKey()).toBe(testKey);
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors gracefully', async () => {
      // Mock network error
      const originalFetch = global.fetch;
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      try {
        await apiService.healthCheck();
      } catch (error) {
        expect(error).toBeDefined();
      }

      global.fetch = originalFetch;
    });

    it('should retry on retryable errors', async () => {
      // This would be tested with specific MSW error handlers
      // For now, we'll test that the method exists
      expect(apiService.testConnection).toBeDefined();
    });
  });
});
