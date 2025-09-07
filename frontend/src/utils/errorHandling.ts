/**
 * Error Handling Utilities
 * エラーハンドリングユーティリティ
 */

import { useToast } from '@/components/ui/Toast';

export interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  backoffFactor: number;
}

export interface NetworkError extends Error {
  status?: number;
  code?: string;
  isNetworkError: boolean;
  isRetryable: boolean;
}

export const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 10000,
  backoffFactor: 2,
};

/**
 * Create a network error with additional metadata
 */
export function createNetworkError(
  message: string,
  status?: number,
  code?: string
): NetworkError {
  const error = new Error(message) as NetworkError;
  error.name = 'NetworkError';
  error.status = status;
  error.code = code;
  error.isNetworkError = true;
  error.isRetryable = isRetryableError(status, code);
  return error;
}

/**
 * Determine if an error is retryable
 */
export function isRetryableError(status?: number, code?: string): boolean {
  if (!status) return true; // Network errors without status are retryable

  // Retryable HTTP status codes
  const retryableStatuses = [408, 429, 500, 502, 503, 504];
  if (retryableStatuses.includes(status)) return true;

  // Retryable error codes
  const retryableCodes = ['NETWORK_ERROR', 'TIMEOUT', 'CONNECTION_ERROR'];
  if (code && retryableCodes.includes(code)) return true;

  return false;
}

/**
 * Calculate delay for exponential backoff
 */
export function calculateDelay(
  attempt: number,
  config: RetryConfig = DEFAULT_RETRY_CONFIG
): number {
  const delay = config.baseDelay * Math.pow(config.backoffFactor, attempt - 1);
  return Math.min(delay, config.maxDelay);
}

/**
 * Retry function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig = DEFAULT_RETRY_CONFIG,
  onRetry?: (attempt: number, error: Error) => void
): Promise<T> {
  let lastError: Error;

  for (let attempt = 1; attempt <= config.maxRetries + 1; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      // Don't retry if it's the last attempt or error is not retryable
      if (
        attempt > config.maxRetries ||
        !isRetryableError(
          (error as NetworkError).status,
          (error as NetworkError).code
        )
      ) {
        throw error;
      }

      // Call retry callback
      if (onRetry) {
        onRetry(attempt, error as Error);
      }

      // Wait before retrying
      const delay = calculateDelay(attempt, config);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

/**
 * Wrap fetch with retry logic
 */
export async function fetchWithRetry(
  url: string,
  options?: RequestInit,
  config?: RetryConfig
): Promise<Response> {
  return retryWithBackoff(async () => {
    try {
      const response = await fetch(url, options);

      if (!response.ok) {
        throw createNetworkError(
          `HTTP ${response.status}: ${response.statusText}`,
          response.status
        );
      }

      return response;
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw createNetworkError(
          'ネットワーク接続エラー',
          undefined,
          'NETWORK_ERROR'
        );
      }
      throw error;
    }
  }, config);
}

/**
 * Error notification hook
 */
export function useErrorHandler() {
  const { addToast } = useToast();

  const handleError = (error: Error, context?: string) => {
    console.error('Error occurred:', error, context);

    let message = error.message;
    let title = 'エラーが発生しました';

    if (error instanceof NetworkError) {
      if (error.status === 404) {
        title = 'リソースが見つかりません';
        message = '要求されたリソースが見つかりませんでした。';
      } else if (error.status === 500) {
        title = 'サーバーエラー';
        message =
          'サーバーで問題が発生しました。しばらく時間をおいてから再度お試しください。';
      } else if (error.status === 429) {
        title = 'リクエスト制限';
        message =
          'リクエストが多すぎます。しばらく時間をおいてから再度お試しください。';
      } else if (error.code === 'NETWORK_ERROR') {
        title = 'ネットワークエラー';
        message = 'ネットワーク接続を確認してください。';
      }
    }

    addToast({
      type: 'error',
      title,
      message: context ? `${context}: ${message}` : message,
      duration: 5000,
    });
  };

  const handleNetworkError = (error: NetworkError, context?: string) => {
    if (error.isRetryable) {
      addToast({
        type: 'warning',
        title: '接続の問題',
        message: '接続に問題があります。自動的に再試行しています...',
        duration: 3000,
      });
    } else {
      handleError(error, context);
    }
  };

  return { handleError, handleNetworkError };
}

/**
 * Global error handler for unhandled promise rejections
 */
export function setupGlobalErrorHandlers() {
  // Handle unhandled promise rejections
  window.addEventListener('unhandledrejection', event => {
    console.error('Unhandled promise rejection:', event.reason);

    // Prevent the default browser error handling
    event.preventDefault();

    // Store error for debugging
    const errorReport = {
      type: 'unhandledrejection',
      reason: event.reason?.toString() || 'Unknown error',
      timestamp: new Date().toISOString(),
      url: window.location.href,
    };

    const existingErrors = JSON.parse(
      localStorage.getItem('errorReports') || '[]'
    );
    existingErrors.push(errorReport);
    localStorage.setItem(
      'errorReports',
      JSON.stringify(existingErrors.slice(-10))
    );
  });

  // Handle global errors
  window.addEventListener('error', event => {
    console.error('Global error:', event.error);

    const errorReport = {
      type: 'error',
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      stack: event.error?.stack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
    };

    const existingErrors = JSON.parse(
      localStorage.getItem('errorReports') || '[]'
    );
    existingErrors.push(errorReport);
    localStorage.setItem(
      'errorReports',
      JSON.stringify(existingErrors.slice(-10))
    );
  });
}

/**
 * Check if the backend is available
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch('/health', {
      method: 'GET',
      timeout: 5000,
    } as RequestInit);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Graceful degradation manager
 */
export class GracefulDegradationManager {
  private static instance: GracefulDegradationManager;
  private backendAvailable = true;
  private offlineMode = false;
  private listeners: Array<
    (status: { backendAvailable: boolean; offlineMode: boolean }) => void
  > = [];

  static getInstance(): GracefulDegradationManager {
    if (!GracefulDegradationManager.instance) {
      GracefulDegradationManager.instance = new GracefulDegradationManager();
    }
    return GracefulDegradationManager.instance;
  }

  private constructor() {
    this.setupNetworkMonitoring();
    this.setupBackendMonitoring();
  }

  private setupNetworkMonitoring() {
    window.addEventListener('online', () => {
      this.offlineMode = false;
      this.notifyListeners();
    });

    window.addEventListener('offline', () => {
      this.offlineMode = true;
      this.notifyListeners();
    });
  }

  private setupBackendMonitoring() {
    // Check backend health periodically
    setInterval(async () => {
      const isAvailable = await checkBackendHealth();
      if (isAvailable !== this.backendAvailable) {
        this.backendAvailable = isAvailable;
        this.notifyListeners();
      }
    }, 30000); // Check every 30 seconds
  }

  private notifyListeners() {
    const status = {
      backendAvailable: this.backendAvailable,
      offlineMode: this.offlineMode,
    };

    this.listeners.forEach(listener => listener(status));
  }

  public subscribe(
    listener: (status: {
      backendAvailable: boolean;
      offlineMode: boolean;
    }) => void
  ) {
    this.listeners.push(listener);

    // Return unsubscribe function
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  public getStatus() {
    return {
      backendAvailable: this.backendAvailable,
      offlineMode: this.offlineMode,
    };
  }

  public isFullyOperational(): boolean {
    return this.backendAvailable && !this.offlineMode;
  }

  public canUseBasicFeatures(): boolean {
    return !this.offlineMode; // Can use cached data and local features
  }
}

export default GracefulDegradationManager;
