import { useState, useCallback, useRef } from 'react';
import { generateId } from '@/utils';
import { naturalLanguageProcessor } from '@/services/naturalLanguageProcessor';

interface StreamingOptions {
  onMessageStart?: (messageId: string) => void;
  onMessageUpdate?: (messageId: string, content: string) => void;
  onMessageComplete?: (
    messageId: string,
    content: string,
    reasoning?: string
  ) => void;
  onError?: (error: string) => void;
  speed?: number; // Characters per second
}

interface StreamingState {
  isStreaming: boolean;
  currentMessageId: string | null;
  error: string | null;
}

export const useStreaming = (options: StreamingOptions = {}) => {
  const [state, setState] = useState<StreamingState>({
    isStreaming: false,
    currentMessageId: null,
    error: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);
  const streamTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const {
    speed = 30,
    onMessageStart,
    onMessageUpdate,
    onMessageComplete,
    onError,
  } = options;

  // Start streaming a message with natural language processing
  const startStreaming = useCallback(
    async (prompt: string, sessionId?: string): Promise<string> => {
      // Cancel any existing stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const messageId = generateId();
      abortControllerRef.current = new AbortController();

      setState({
        isStreaming: true,
        currentMessageId: messageId,
        error: null,
      });

      onMessageStart?.(messageId);

      try {
        // Use natural language processor for enhanced streaming
        let fullContent = '';
        let reasoning = '';
        // let _toolsUsed: string[] = [];

        const streamGenerator = naturalLanguageProcessor.processStreamingInput({
          input: prompt,
          sessionId,
          language: 'auto',
        });

        for await (const chunk of streamGenerator) {
          // Check if streaming was cancelled
          if (abortControllerRef.current?.signal.aborted) {
            break;
          }

          if (chunk.content) {
            fullContent += chunk.content;
            onMessageUpdate?.(messageId, fullContent);
          }

          if (chunk.reasoning) {
            reasoning = chunk.reasoning;
          }

          // if (chunk.toolsUsed) {
          //   _toolsUsed = chunk.toolsUsed;
          // }

          if (chunk.isComplete) {
            onMessageComplete?.(messageId, fullContent, reasoning);
            setState({
              isStreaming: false,
              currentMessageId: null,
              error: null,
            });
            return messageId;
          }
        }

        // Fallback completion if loop exits without completion
        onMessageComplete?.(messageId, fullContent, reasoning);
        setState({
          isStreaming: false,
          currentMessageId: null,
          error: null,
        });

        return messageId;
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          // Stream was cancelled
          setState({
            isStreaming: false,
            currentMessageId: null,
            error: null,
          });
          return messageId;
        }

        const errorMessage =
          error instanceof Error ? error.message : 'Unknown streaming error';
        setState({
          isStreaming: false,
          currentMessageId: null,
          error: errorMessage,
        });
        onError?.(errorMessage);
        throw error;
      }
    },
    [speed, onMessageStart, onMessageUpdate, onMessageComplete, onError]
  );

  // Simulate streaming for development/testing
  const simulateStreaming = useCallback(
    (content: string, reasoning?: string): Promise<string> => {
      return new Promise(resolve => {
        const messageId = generateId();

        setState({
          isStreaming: true,
          currentMessageId: messageId,
          error: null,
        });

        onMessageStart?.(messageId);

        let currentIndex = 0;
        let currentContent = '';

        const streamCharacter = () => {
          if (currentIndex < content.length) {
            currentContent += content[currentIndex];
            currentIndex++;
            onMessageUpdate?.(messageId, currentContent);

            streamTimeoutRef.current = setTimeout(
              streamCharacter,
              1000 / speed
            );
          } else {
            onMessageComplete?.(messageId, content, reasoning);
            setState({
              isStreaming: false,
              currentMessageId: null,
              error: null,
            });
            resolve(messageId);
          }
        };

        streamCharacter();
      });
    },
    [speed, onMessageStart, onMessageUpdate, onMessageComplete]
  );

  // Stop streaming
  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    if (streamTimeoutRef.current) {
      clearTimeout(streamTimeoutRef.current);
      streamTimeoutRef.current = null;
    }

    setState({
      isStreaming: false,
      currentMessageId: null,
      error: null,
    });
  }, []);

  // Retry streaming with exponential backoff
  const retryStreaming = useCallback(
    async (
      prompt: string,
      sessionId?: string,
      maxRetries = 3
    ): Promise<string | null> => {
      let retries = 0;

      while (retries < maxRetries) {
        try {
          return await startStreaming(prompt, sessionId);
        } catch (error) {
          retries++;
          if (retries >= maxRetries) {
            throw error;
          }

          // Exponential backoff: 1s, 2s, 4s
          const delay = Math.pow(2, retries - 1) * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }

      return null;
    },
    [startStreaming]
  );

  return {
    ...state,
    startStreaming,
    simulateStreaming,
    stopStreaming,
    retryStreaming,
  };
};
