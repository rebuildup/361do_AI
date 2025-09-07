/**
 * Application Resilience Hook
 * アプリケーション復旧性フック
 */

import { useState, useEffect, useCallback } from 'react';
import {
  GracefulDegradationManager,
  useErrorHandler,
} from '@/utils/errorHandling';
import { useToast } from '@/components/ui/Toast';

export interface AppStatus {
  isOnline: boolean;
  backendAvailable: boolean;
  degradedMode: boolean;
  lastError: Error | null;
  retryCount: number;
}

export interface ResilienceActions {
  retry: () => void;
  reportError: (error: Error, context?: string) => void;
  clearError: () => void;
  checkBackendHealth: () => Promise<boolean>;
}

/**
 * Hook for managing application resilience and graceful degradation
 */
export function useAppResilience(): [AppStatus, ResilienceActions] {
  const [status, setStatus] = useState<AppStatus>({
    isOnline: navigator.onLine,
    backendAvailable: true,
    degradedMode: false,
    lastError: null,
    retryCount: 0,
  });

  const { addToast } = useToast();
  const { handleError } = useErrorHandler();
  const degradationManager = GracefulDegradationManager.getInstance();

  // Subscribe to network and backend status changes
  useEffect(() => {
    const unsubscribe = degradationManager.subscribe(degradationStatus => {
      setStatus(prev => ({
        ...prev,
        backendAvailable: degradationStatus.backendAvailable,
        degradedMode:
          !degradationStatus.backendAvailable || degradationStatus.offlineMode,
        isOnline: !degradationStatus.offlineMode,
      }));

      // Show notifications for status changes
      if (!degradationStatus.backendAvailable && prev.backendAvailable) {
        addToast({
          type: 'warning',
          title: 'サーバー接続エラー',
          message: '一部の機能が制限されます。自動的に再接続を試行しています。',
          duration: 5000,
        });
      } else if (degradationStatus.backendAvailable && !prev.backendAvailable) {
        addToast({
          type: 'success',
          title: '接続復旧',
          message: 'サーバーへの接続が復旧しました。',
          duration: 3000,
        });
      }

      if (degradationStatus.offlineMode && !prev.degradedMode) {
        addToast({
          type: 'warning',
          title: 'オフラインモード',
          message: 'ネットワーク接続が失われました。一部の機能が制限されます。',
          duration: 5000,
        });
      } else if (!degradationStatus.offlineMode && prev.degradedMode) {
        addToast({
          type: 'success',
          title: 'オンライン復旧',
          message: 'ネットワーク接続が復旧しました。',
          duration: 3000,
        });
      }
    });

    return unsubscribe;
  }, [addToast]);

  // Monitor online/offline status
  useEffect(() => {
    const handleOnline = () => {
      setStatus(prev => ({ ...prev, isOnline: true }));
    };

    const handleOffline = () => {
      setStatus(prev => ({ ...prev, isOnline: false, degradedMode: true }));
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const retry = useCallback(() => {
    setStatus(prev => ({
      ...prev,
      retryCount: prev.retryCount + 1,
      lastError: null,
    }));

    // Force a backend health check
    degradationManager.getInstance();
  }, []);

  const reportError = useCallback(
    (error: Error, context?: string) => {
      setStatus(prev => ({ ...prev, lastError: error }));
      handleError(error, context);
    },
    [handleError]
  );

  const clearError = useCallback(() => {
    setStatus(prev => ({ ...prev, lastError: null }));
  }, []);

  const checkBackendHealth = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch('/health', {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      const isHealthy = response.ok;

      setStatus(prev => ({
        ...prev,
        backendAvailable: isHealthy,
        degradedMode: !isHealthy || !prev.isOnline,
      }));

      return isHealthy;
    } catch {
      setStatus(prev => ({
        ...prev,
        backendAvailable: false,
        degradedMode: true,
      }));
      return false;
    }
  }, []);

  const actions: ResilienceActions = {
    retry,
    reportError,
    clearError,
    checkBackendHealth,
  };

  return [status, actions];
}

/**
 * Hook for component-level error handling
 */
export function useComponentResilience(componentName: string) {
  const [error, setError] = useState<Error | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [isRetrying, setIsRetrying] = useState(false);
  const { handleError } = useErrorHandler();

  const reportError = useCallback(
    (error: Error) => {
      console.error(`Error in ${componentName}:`, error);
      setError(error);
      handleError(error, componentName);
    },
    [componentName, handleError]
  );

  const retry = useCallback(
    async (retryFn?: () => Promise<void> | void) => {
      setIsRetrying(true);
      setRetryCount(prev => prev + 1);

      try {
        if (retryFn) {
          await retryFn();
        }
        setError(null);
      } catch (retryError) {
        reportError(retryError as Error);
      } finally {
        setIsRetrying(false);
      }
    },
    [reportError]
  );

  const clearError = useCallback(() => {
    setError(null);
    setRetryCount(0);
  }, []);

  return {
    error,
    retryCount,
    isRetrying,
    reportError,
    retry,
    clearError,
  };
}

/**
 * Hook for API call resilience
 */
export function useApiResilience() {
  const [, { reportError }] = useAppResilience();

  const callWithResilience = useCallback(
    async <T>(
      apiCall: () => Promise<T>,
      options?: {
        maxRetries?: number;
        onRetry?: (attempt: number) => void;
        fallbackValue?: T;
      }
    ): Promise<T> => {
      const { maxRetries = 3, onRetry, fallbackValue } = options || {};

      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          return await apiCall();
        } catch (error) {
          const isLastAttempt = attempt === maxRetries;

          if (isLastAttempt) {
            reportError(error as Error, 'API Call');

            if (fallbackValue !== undefined) {
              return fallbackValue;
            }

            throw error;
          }

          if (onRetry) {
            onRetry(attempt);
          }

          // Wait before retrying (exponential backoff)
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }

      throw new Error('Max retries exceeded');
    },
    [reportError]
  );

  return { callWithResilience };
}

export default useAppResilience;
