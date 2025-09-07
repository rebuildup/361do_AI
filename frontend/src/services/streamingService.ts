import { generateId } from '@/utils';

export interface StreamingMessage {
  id: string;
  type: 'start' | 'content' | 'reasoning' | 'complete' | 'error';
  content?: string;
  reasoning?: string;
  error?: string;
  metadata?: Record<string, any>;
}

export interface StreamingOptions {
  onMessage?: (message: StreamingMessage) => void;
  onError?: (error: Error) => void;
  onClose?: () => void;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

export class StreamingService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private reconnectDelay: number;
  private isConnecting = false;
  private messageQueue: any[] = [];
  private options: StreamingOptions;

  constructor(options: StreamingOptions = {}) {
    this.options = options;
    this.maxReconnectAttempts = options.reconnectAttempts ?? 3;
    this.reconnectDelay = options.reconnectDelay ?? 1000;
  }

  // Connect to WebSocket
  async connect(url?: string): Promise<void> {
    if (
      this.isConnecting ||
      (this.ws && this.ws.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    this.isConnecting = true;
    const wsUrl = url || this.getWebSocketUrl();

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        // Send queued messages
        while (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift();
          this.send(message);
        }
      };

      this.ws.onmessage = event => {
        try {
          const message: StreamingMessage = JSON.parse(event.data);
          this.options.onMessage?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = event => {
        console.log('WebSocket closed:', event.code, event.reason);
        this.isConnecting = false;
        this.ws = null;

        if (
          !event.wasClean &&
          this.reconnectAttempts < this.maxReconnectAttempts
        ) {
          this.scheduleReconnect();
        } else {
          this.options.onClose?.();
        }
      };

      this.ws.onerror = error => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
        this.options.onError?.(new Error('WebSocket connection error'));
      };
    } catch (error) {
      this.isConnecting = false;
      throw error;
    }
  }

  // Send message through WebSocket
  send(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Queue message for when connection is established
      this.messageQueue.push(message);
      if (!this.isConnecting) {
        this.connect();
      }
    }
  }

  // Start streaming chat
  startChat(prompt: string, sessionId?: string): string {
    const messageId = generateId();

    this.send({
      type: 'chat',
      id: messageId,
      prompt,
      session_id: sessionId,
      stream: true,
    });

    return messageId;
  }

  // Stop streaming
  stopStreaming(messageId: string): void {
    this.send({
      type: 'stop',
      id: messageId,
    });
  }

  // Disconnect WebSocket
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.messageQueue = [];
    this.reconnectAttempts = 0;
  }

  // Get WebSocket URL
  private getWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/v1/ws/chat`;
  }

  // Schedule reconnection with exponential backoff
  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(
      `Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`
    );

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  // Get connection status
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get connectionState(): string {
    if (!this.ws) return 'disconnected';

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'closed';
      default:
        return 'unknown';
    }
  }
}

// Server-Sent Events alternative for streaming
export class SSEStreamingService {
  private eventSource: EventSource | null = null;
  private options: StreamingOptions;

  constructor(options: StreamingOptions = {}) {
    this.options = options;
  }

  // Start SSE streaming
  startStream(prompt: string, sessionId?: string): string {
    const messageId = generateId();

    // Close existing connection
    this.disconnect();

    const url = new URL('/v1/chat/stream', window.location.origin);
    url.searchParams.set('prompt', prompt);
    url.searchParams.set('message_id', messageId);
    if (sessionId) {
      url.searchParams.set('session_id', sessionId);
    }

    this.eventSource = new EventSource(url.toString());

    this.eventSource.onopen = () => {
      console.log('SSE connection opened');
    };

    this.eventSource.onmessage = event => {
      try {
        const message: StreamingMessage = JSON.parse(event.data);
        this.options.onMessage?.(message);
      } catch (error) {
        console.error('Failed to parse SSE message:', error);
      }
    };

    this.eventSource.onerror = error => {
      console.error('SSE error:', error);
      this.options.onError?.(new Error('SSE connection error'));
    };

    return messageId;
  }

  // Disconnect SSE
  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  get isConnected(): boolean {
    return this.eventSource?.readyState === EventSource.OPEN;
  }
}

// Factory function to create appropriate streaming service
export function createStreamingService(
  type: 'websocket' | 'sse' = 'websocket',
  options: StreamingOptions = {}
): StreamingService | SSEStreamingService {
  if (type === 'sse') {
    return new SSEStreamingService(options);
  }
  return new StreamingService(options);
}
