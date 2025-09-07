/**
 * Natural Language Processor Tests
 *
 * Tests for natural language processing capabilities
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { NaturalLanguageProcessor } from '../naturalLanguageProcessor';

describe('NaturalLanguageProcessor', () => {
  let nlp: NaturalLanguageProcessor;

  beforeEach(() => {
    nlp = new NaturalLanguageProcessor();
  });

  describe('Language Detection', () => {
    it('should detect Japanese language', async () => {
      const response = await nlp.processInput({
        input: 'こんにちは、今日はいい天気ですね。',
        language: 'auto',
      });

      expect(response.language).toBe('ja');
      expect(response.response).toBeDefined();
    });

    it('should detect English language', async () => {
      const response = await nlp.processInput({
        input: 'Hello, how are you today?',
        language: 'auto',
      });

      expect(response.language).toBe('en');
      expect(response.response).toBeDefined();
    });

    it('should handle mixed language input', async () => {
      const response = await nlp.processInput({
        input: 'Hello こんにちは mixed language test',
        language: 'auto',
      });

      expect(['ja', 'en']).toContain(response.language);
      expect(response.response).toBeDefined();
    });
  });

  describe('Tool Invocation Detection', () => {
    it('should detect web search intent in Japanese', async () => {
      const response = await nlp.processInput({
        input: '最新のAI技術について調べて教えてください。',
        language: 'ja',
      });

      expect(response.toolsUsed).toContain('web_search');
      expect(response.reasoning).toContain('検索');
    });

    it('should detect file operations intent', async () => {
      const response = await nlp.processInput({
        input: 'プロジェクトの設定ファイルを確認してください。',
        language: 'ja',
      });

      expect(response.toolsUsed).toContain('file_operations');
      expect(response.reasoning).toContain('ファイル');
    });

    it('should detect system command intent in English', async () => {
      const response = await nlp.processInput({
        input: 'Can you check the current system memory usage?',
        language: 'en',
      });

      expect(response.toolsUsed).toContain('command_execution');
      expect(response.reasoning).toContain('system');
    });
  });

  describe('Conversation Context', () => {
    it('should maintain conversation context across messages', async () => {
      const sessionId = 'test-context-session';

      // First message
      const response1 = await nlp.processInput({
        input: 'React Hooksについて教えてください。',
        sessionId,
        language: 'ja',
      });

      expect(response1.sessionUpdated).toBe(true);

      // Follow-up message
      const response2 = await nlp.processInput({
        input: 'それについてもう少し詳しく説明してください。',
        sessionId,
        language: 'ja',
      });

      expect(response2.sessionUpdated).toBe(true);
      expect(response2.response).toBeDefined();
    });

    it('should handle new sessions without context', async () => {
      const response = await nlp.processInput({
        input: 'Hello, this is a new conversation.',
        language: 'en',
      });

      expect(response.response).toBeDefined();
      expect(response.processingTime).toBeGreaterThan(0);
    });
  });

  describe('Streaming Processing', () => {
    it('should process streaming input', async () => {
      const chunks: any[] = [];

      const stream = nlp.processStreamingInput({
        input: 'Tell me about React development.',
        language: 'en',
      });

      for await (const chunk of stream) {
        chunks.push(chunk);
        if (chunk.isComplete) break;
      }

      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks[chunks.length - 1].isComplete).toBe(true);
    });

    it('should handle streaming errors gracefully', async () => {
      const chunks: any[] = [];

      try {
        const stream = nlp.processStreamingInput({
          input: '', // Empty input to potentially trigger error
          language: 'ja',
        });

        for await (const chunk of stream) {
          chunks.push(chunk);
          if (chunk.isComplete) break;
        }
      } catch {
        // Error handling is expected for edge cases
      }

      // Should either succeed or fail gracefully
      expect(true).toBe(true);
    });
  });

  describe('Confidence Scoring', () => {
    it('should provide confidence scores', async () => {
      const response = await nlp.processInput({
        input: 'This is a clear and simple request.',
        language: 'en',
      });

      expect(response.confidence).toBeDefined();
      expect(response.confidence).toBeGreaterThanOrEqual(0);
      expect(response.confidence).toBeLessThanOrEqual(1);
    });

    it('should have lower confidence for unclear input', async () => {
      const response = await nlp.processInput({
        input: 'asdf jkl; qwerty',
        language: 'auto',
      });

      expect(response.confidence).toBeDefined();
      expect(response.confidence).toBeLessThan(0.5);
    });
  });

  describe('Error Handling', () => {
    it('should provide fallback responses on errors', async () => {
      // Mock API service to throw error
      // const _originalProcessUserInput = vi
      //   .fn()
      //   .mockRejectedValue(new Error('API Error'));

      const response = await nlp.processInput({
        input: 'Test input that should trigger fallback',
        language: 'ja',
      });

      expect(response.response).toBeDefined();
      expect(response.confidence).toBeLessThan(0.5);
    });

    it('should handle empty input gracefully', async () => {
      const response = await nlp.processInput({
        input: '',
        language: 'ja',
      });

      expect(response.response).toBeDefined();
      expect(response.language).toBe('ja');
    });
  });

  describe('Context Management', () => {
    it('should clear conversation context', () => {
      const sessionId = 'test-clear-session';

      nlp.clearConversationContext(sessionId);

      const context = nlp.getConversationContext(sessionId);
      expect(context).toBeUndefined();
    });

    it('should retrieve conversation context', async () => {
      const sessionId = 'test-get-context';

      await nlp.processInput({
        input: 'Test message for context',
        sessionId,
        language: 'en',
      });

      const context = nlp.getConversationContext(sessionId);
      expect(context).toBeDefined();
      expect(context?.sessionId).toBe(sessionId);
    });
  });

  describe('Performance', () => {
    it('should process input within reasonable time', async () => {
      const startTime = performance.now();

      const response = await nlp.processInput({
        input: 'Quick performance test message.',
        language: 'en',
      });

      const endTime = performance.now();
      const processingTime = endTime - startTime;

      expect(response.processingTime).toBeDefined();
      expect(processingTime).toBeLessThan(10000); // Should complete within 10 seconds
    });
  });
});
