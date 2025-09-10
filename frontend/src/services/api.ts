/**
 * API Service Layer
 *
 * Handles all communication with the FastAPI backend
 */

import axios from 'axios';
import type { AxiosInstance, AxiosResponse } from 'axios';
import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionStreamResponse,
  ModelsResponse,
  HealthResponse,
  SystemStatsRequest,
  SystemStatsResponse,
  InferenceRequest,
  InferenceResponse,
  MemorySearchRequest,
  MemorySearchResponse,
  SessionRequest,
  SessionResponse,
  ApiError,
  MessageRole,
} from '@/types';

// API configuration
// Development: use same-origin and Vite proxy for /v1
// Production: prefer VITE_API_BASE; fallback to local backend to avoid 404s from static servers
export const API_BASE_URL =
  (typeof import.meta !== 'undefined' &&
    (import.meta as any).env?.VITE_API_BASE) ||
  (typeof window !== 'undefined' && (window as any).__API_BASE__) ||
  // Default to localhost:8000 for production builds
  'http://localhost:8000';
export const API_VERSION = 'v1';
export const API_TIMEOUT = 60000; // 60 seconds for agent processing
export const isBackendConfigured: boolean = !!API_BASE_URL;
export function buildApiUrl(path: string): string {
  const trimmed = path.replace(/^\/+/, '');
  const base = API_BASE_URL ? API_BASE_URL.replace(/\/$/, '') : '';
  const versioned = `${base ? base + '/' : '/'}${API_VERSION}`;
  return `${versioned}/${trimmed}`;
}

/**
 * Custom error class for API errors
 */
export class ApiServiceError extends Error {
  public readonly code: string;
  public readonly details?: Record<string, unknown>;
  public readonly retryable: boolean;

  constructor(error: ApiError) {
    super(error.message);
    this.name = 'ApiServiceError';
    this.code = error.code;
    this.details = error.details;
    this.retryable = error.retryable;
  }
}

/**
 * Main API service class
 */
export class ApiService {
  private client: AxiosInstance;
  private apiKey?: string;

  constructor(apiKey?: string) {
    this.apiKey = apiKey;
    this.client = this.createAxiosInstance();
  }

  /**
   * Create and configure axios instance
   */
  private createAxiosInstance(): AxiosInstance {
    const client = axios.create({
      baseURL: `${API_BASE_URL ? API_BASE_URL.replace(/\/$/, '') + '/' : '/'}${API_VERSION}`,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for authentication
    client.interceptors.request.use(
      config => {
        if (this.apiKey) {
          (config.headers as any).Authorization = `Bearer ${this.apiKey}`;
        }
        return config;
      },
      error => Promise.reject(error)
    );

    // Response interceptor for error handling and retries
    client.interceptors.response.use(
      response => response,
      async error => {
        const apiError = this.handleApiError(error);

        // Implement retry logic for retryable errors
        if (
          apiError.retryable &&
          error.config &&
          !(error.config as any)._retry
        ) {
          (error.config as any)._retry = true;
          (error.config as any)._retryCount =
            ((error.config as any)._retryCount || 0) + 1;

          if ((error.config as any)._retryCount <= 3) {
            // Exponential backoff
            const delay = Math.pow(2, (error.config as any)._retryCount) * 1000;
            await new Promise(resolve => setTimeout(resolve, delay));

            return client.request(error.config);
          }
        }

        return Promise.reject(new ApiServiceError(apiError));
      }
    );

    return client;
  }

  /**
   * Handle API errors and convert to standardized format
   */
  private handleApiError(error: unknown): ApiError {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      const data = error.response?.data;

      // Handle different error types
      switch (status) {
        case 401:
          return {
            code: 'UNAUTHORIZED',
            message: '認証が必要です',
            retryable: false,
          };
        case 403:
          return {
            code: 'FORBIDDEN',
            message: 'アクセスが拒否されました',
            retryable: false,
          };
        case 404:
          return {
            code: 'NOT_FOUND',
            message: 'リソースが見つかりません',
            retryable: false,
          };
        case 422:
          return {
            code: 'VALIDATION_ERROR',
            message: (data as any)?.detail || '入力データに問題があります',
            details: data as any,
            retryable: false,
          };
        case 429:
          return {
            code: 'RATE_LIMITED',
            message:
              'リクエストが多すぎます。しばらく時間をおいてから再度お試しください',
            retryable: true,
          };
        case 500:
          return {
            code: 'INTERNAL_ERROR',
            message:
              (data as any)?.detail || 'サーバー内部エラーが発生しました',
            retryable: true,
          };
        case 502:
        case 503:
        case 504:
          return {
            code: 'SERVICE_UNAVAILABLE',
            message: 'サービスが一時的に利用できません',
            retryable: true,
          };
        default:
          return {
            code: 'NETWORK_ERROR',
            message:
              (error as any).message || 'ネットワークエラーが発生しました',
            details: { status, data } as any,
            retryable: true,
          };
      }
    }

    // Handle network errors
    if (error instanceof Error) {
      if (error.name === 'NetworkError' || error.message.includes('fetch')) {
        return {
          code: 'NETWORK_ERROR',
          message: 'ネットワーク接続に失敗しました',
          retryable: true,
        };
      }

      if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
        return {
          code: 'TIMEOUT_ERROR',
          message: 'リクエストがタイムアウトしました',
          retryable: true,
        };
      }
    }

    return {
      code: 'UNKNOWN_ERROR',
      message: '予期しないエラーが発生しました',
      retryable: false,
    };
  }

  /**
   * Health check endpoint with fallback
   */
  async healthCheck(): Promise<HealthResponse> {
    try {
      const response: AxiosResponse<HealthResponse> =
        await this.client.get('/health');
      return response.data;
    } catch (error) {
      console.warn('Health check failed, returning fallback status:', error);

      // Return fallback health status
      return {
        status: 'degraded',
        timestamp: new Date().toISOString(),
        version: 'unknown',
        system_info: {
          cpu_percent: 0,
          memory_percent: 0,
          note: 'Backend unavailable',
        },
      };
    }
  }

  /**
   * Get available models
   */
  async getModels(): Promise<ModelsResponse> {
    const response: AxiosResponse<ModelsResponse> =
      await this.client.get('/models');
    return response.data;
  }

  /**
   * Send chat completion request
   */
  async chatCompletion(
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    const response: AxiosResponse<ChatCompletionResponse> =
      await this.client.post('/chat/completions', request);
    return response.data;
  }

  /**
   * Send enhanced chat completion request through agent
   */
  async agentChatCompletion(
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    const response: AxiosResponse<ChatCompletionResponse> =
      await this.client.post('/chat/completions/agent', request);
    return response.data;
  }

  /**
   * Send streaming chat completion request
   */
  async *streamChatCompletion(
    request: ChatCompletionRequest,
    useAgent: boolean = false
  ): AsyncGenerator<ChatCompletionStreamResponse, void, unknown> {
    const streamRequest = { ...request, stream: true } as any;
    const endpoint = useAgent ? '/chat/completions/agent' : '/chat/completions';

    try {
      const response = await fetch(buildApiUrl(endpoint), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
        },
        body: JSON.stringify(streamRequest),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is not readable');
      }

      const decoder = new TextDecoder();

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6).trim();
              if (data === '[DONE]') return;

              try {
                const parsed: ChatCompletionStreamResponse = JSON.parse(data);
                yield parsed;
              } catch (error) {
                console.warn('Failed to parse streaming response:', error);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      console.error('Streaming error:', error);
      throw error;
    }
  }

  /**
   * Send inference request
   */
  async inference(request: InferenceRequest): Promise<InferenceResponse> {
    const response: AxiosResponse<InferenceResponse> = await this.client.post(
      '/inference',
      request
    );
    return response.data;
  }

  /**
   * Search memory
   */
  async searchMemory(
    request: MemorySearchRequest
  ): Promise<MemorySearchResponse> {
    const response: AxiosResponse<MemorySearchResponse> =
      await this.client.post('/memory/search', request);
    return response.data;
  }

  /**
   * Create new session
   */
  async createSession(request: SessionRequest): Promise<SessionResponse> {
    const response: AxiosResponse<SessionResponse> = await this.client.post(
      '/sessions',
      request
    );
    return response.data;
  }

  /**
   * Get session by ID
   */
  async getSession(sessionId: string): Promise<SessionResponse> {
    const response: AxiosResponse<SessionResponse> = await this.client.get(
      `/sessions/${sessionId}`
    );
    return response.data;
  }

  /**
   * Get system statistics
   */
  async getSystemStats(
    request: SystemStatsRequest = {}
  ): Promise<SystemStatsResponse> {
    const response: AxiosResponse<SystemStatsResponse> = await this.client.post(
      '/system/stats',
      request
    );
    return response.data;
  }

  /**
   * Update API key
   */
  setApiKey(apiKey: string): void {
    this.apiKey = apiKey;
  }

  /**
   * Get current API key
   */
  getApiKey(): string | undefined {
    return this.apiKey;
  }

  /**
   * Test backend connection and verify all endpoints
   */
  async testConnection(): Promise<{
    healthy: boolean;
    endpoints: Record<string, boolean>;
    errors: string[];
  }> {
    const results = {
      healthy: true,
      endpoints: {} as Record<string, boolean>,
      errors: [] as string[],
    };

    // Test health endpoint
    try {
      await this.healthCheck();
      (results.endpoints as any).health = true;
    } catch (error) {
      (results.endpoints as any).health = false;
      results.errors.push(`Health check failed: ${error}`);
      results.healthy = false;
    }

    // Test models endpoint
    try {
      await this.getModels();
      (results.endpoints as any).models = true;
    } catch (error) {
      (results.endpoints as any).models = false;
      results.errors.push(`Models endpoint failed: ${error}`);
    }

    // Test agent status endpoint
    try {
      await this.getAgentStatus();
      (results.endpoints as any).agentStatus = true;
    } catch (error) {
      (results.endpoints as any).agentStatus = false;
      results.errors.push(`Agent status failed: ${error}`);
    }

    // Test session creation
    try {
      await this.createSession({ user_id: 'test' });
      (results.endpoints as any).sessions = true;
    } catch (error) {
      (results.endpoints as any).sessions = false;
      results.errors.push(`Session creation failed: ${error}`);
    }

    return results;
  }

  /**
   * Agent-specific methods for enhanced functionality
   */

  /**
   * Process user input through the agent (natural language)
   */
  async processUserInput(
    input: string,
    sessionId?: string,
    useAgent: boolean = true
  ): Promise<InferenceResponse | ChatCompletionResponse> {
    try {
      if (useAgent) {
        // Try enhanced agent endpoint first
        const chatRequest: ChatCompletionRequest = {
          model: 'agent',
          messages: [{ role: 'user', content: input }],
          temperature: 0.7,
          max_tokens: 2000,
        };

        return await this.agentChatCompletion(chatRequest);
      } else {
        // Fallback to inference endpoint
        const request: InferenceRequest = {
          prompt: input,
          session_id: sessionId,
          use_cot: true, // Enable Chain-of-Thought reasoning
        };

        return this.inference(request);
      }
    } catch (error) {
      console.warn(
        'Agent processing failed, falling back to inference:',
        error
      );

      // Fallback to basic inference
      const request: InferenceRequest = {
        prompt: input,
        session_id: sessionId,
        use_cot: true,
      };

      return this.inference(request);
    }
  }

  /**
   * Get agent status and capabilities
   */
  async getAgentStatus(): Promise<{
    status: string;
    capabilities: string[];
    active_tools: string[];
    memory_usage: Record<string, unknown>;
  }> {
    const response = await this.client.get('/agent/status');
    return response.data;
  }

  /**
   * Update agent configuration
   */
  async updateAgentConfig(config: {
    temperature?: number;
    max_tokens?: number;
    streaming_enabled?: boolean;
    tools_enabled?: string[];
  }): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post('/agent/config', config);
    return response.data;
  }

  /**
   * Get available tools and their status
   */
  async getAvailableTools(): Promise<{
    tools: Array<{
      name: string;
      description: string;
      enabled: boolean;
      parameters?: Record<string, unknown>;
    }>;
  }> {
    const response = await this.client.get('/agent/tools');
    return response.data;
  }

  /**
   * Execute a specific tool with parameters
   */
  async executeTool(
    toolName: string,
    parameters: Record<string, unknown>,
    sessionId?: string
  ): Promise<{
    result: unknown;
    execution_time: number;
    success: boolean;
    error?: string;
  }> {
    const response = await this.client.post('/agent/tools/execute', {
      tool_name: toolName,
      parameters,
      session_id: sessionId,
    });
    return response.data;
  }

  /**
   * Get learning progress and metrics
   */
  async getLearningMetrics(): Promise<{
    total_interactions: number;
    successful_responses: number;
    average_response_time: number;
    learning_rate: number;
    recent_improvements: Array<{
      timestamp: string;
      improvement_type: string;
      description: string;
    }>;
  }> {
    const response = await this.client.get('/agent/learning/metrics');
    return response.data;
  }

  /**
   * Provide feedback on agent response
   */
  async provideFeedback(
    responseId: string,
    feedback: {
      rating: number; // 1-5
      helpful: boolean;
      comments?: string;
    }
  ): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post('/agent/feedback', {
      response_id: responseId,
      ...feedback,
    });
    return response.data;
  }

  /**
   * Get conversation history with enhanced metadata
   */
  async getConversationHistory(
    sessionId: string,
    limit = 50
  ): Promise<{
    messages: Array<{
      id: string;
      role: MessageRole;
      content: string;
      timestamp: string;
      reasoning?: string;
      tools_used?: string[];
      processing_time?: number;
    }>;
    total_count: number;
  }> {
    const response = await this.client.get(`/sessions/${sessionId}/messages`, {
      params: { limit },
    });
    return response.data;
  }

  /**
   * Clear conversation history
   */
  async clearConversationHistory(sessionId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    const response = await this.client.delete(
      `/sessions/${sessionId}/messages`
    );
    return response.data;
  }

  /**
   * Export conversation data
   */
  async exportConversation(
    sessionId: string,
    format: 'json' | 'markdown' | 'txt' = 'json'
  ): Promise<Blob> {
    const response = await this.client.get(`/sessions/${sessionId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * Fallback methods for degraded mode
   */

  /**
   * Get cached models when backend is unavailable
   */
  getCachedModels(): ModelsResponse {
    const cached = localStorage.getItem('cached_models');
    if (cached) {
      try {
        return JSON.parse(cached);
      } catch (error) {
        console.warn('Failed to parse cached models:', error);
      }
    }

    // Return default models as fallback
    return {
      object: 'list',
      data: [
        {
          id: 'agent',
          object: 'model',
          created: Date.now(),
          owned_by: 'system',
        },
        {
          id: 'fallback',
          object: 'model',
          created: Date.now(),
          owned_by: 'system',
        },
      ],
    };
  }

  /**
   * Cache models for offline use
   */
  async cacheModels(): Promise<void> {
    try {
      const models = await this.getModels();
      localStorage.setItem('cached_models', JSON.stringify(models));
    } catch (error) {
      console.warn('Failed to cache models:', error);
    }
  }

  /**
   * Get offline chat response (mock for demonstration)
   */
  getOfflineChatResponse(message: string): ChatCompletionResponse {
    return {
      id: `offline_${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: 'offline',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: `申し訳ございませんが、現在オフラインモードです。サーバーに接続できないため、完全な応答を提供できません。ネットワーク接続を確認してから再度お試しください。\n\nあなたのメッセージ: "${message}"`,
          },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: message.length,
        completion_tokens: 50,
        total_tokens: message.length + 50,
      },
    };
  }

  /**
   * Store message for later processing when connection is restored
   */
  storeOfflineMessage(message: string, sessionId?: string): void {
    const offlineMessages = JSON.parse(
      localStorage.getItem('offline_messages') || '[]'
    );

    offlineMessages.push({
      id: `offline_${Date.now()}`,
      content: message,
      sessionId,
      timestamp: new Date().toISOString(),
    });

    // Keep only last 50 offline messages
    if (offlineMessages.length > 50) {
      offlineMessages.splice(0, offlineMessages.length - 50);
    }

    localStorage.setItem('offline_messages', JSON.stringify(offlineMessages));
  }

  /**
   * Get stored offline messages
   */
  getOfflineMessages(): Array<{
    id: string;
    content: string;
    sessionId?: string;
    timestamp: string;
  }> {
    return JSON.parse(localStorage.getItem('offline_messages') || '[]');
  }

  /**
   * Clear offline messages (after successful sync)
   */
  clearOfflineMessages(): void {
    localStorage.removeItem('offline_messages');
  }

  /**
   * Sync offline messages when connection is restored
   */
  async syncOfflineMessages(): Promise<void> {
    const offlineMessages = this.getOfflineMessages();

    if (offlineMessages.length === 0) return;

    try {
      // Process each offline message
      for (const message of offlineMessages) {
        try {
          await this.processUserInput(message.content, message.sessionId);
        } catch (error) {
          console.warn(`Failed to sync offline message ${message.id}:`, error);
        }
      }

      // Clear successfully synced messages
      this.clearOfflineMessages();
    } catch (error) {
      console.error('Failed to sync offline messages:', error);
    }
  }

  /**
   * Check if we're in degraded mode
   */
  async isDegradedMode(): Promise<boolean> {
    try {
      const health = await this.healthCheck();
      return health.status !== 'healthy';
    } catch {
      return true;
    }
  }

  /**
   * Get fallback system stats
   */
  getFallbackSystemStats(): SystemStatsResponse {
    return {
      timestamp: new Date().toISOString(),
      system: {
        cpu_percent: 0,
        memory_percent: 0,
        disk_usage: 0,
        uptime: 0,
      },
      agent: {
        status: 'offline',
        active_sessions: 0,
        total_requests: 0,
        average_response_time: 0,
      },
      note: 'Backend unavailable - showing fallback data',
    };
  }

  /**
   * Initialize fallback data and caching
   */
  async initializeFallbacks(): Promise<void> {
    try {
      // Cache models for offline use
      await this.cacheModels();

      // Cache other essential data
      const health = await this.healthCheck();
      localStorage.setItem(
        'last_health_check',
        JSON.stringify({
          ...health,
          timestamp: new Date().toISOString(),
        })
      );
    } catch (error) {
      console.warn('Failed to initialize fallbacks:', error);
    }
  }
}

// Default API service instance
export const apiService = new ApiService();

// Initialize fallbacks on startup
apiService.initializeFallbacks().catch(console.warn);

// Export for testing and custom instances
export default ApiService;
