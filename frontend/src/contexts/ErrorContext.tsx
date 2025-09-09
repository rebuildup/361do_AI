/**
 * Error Context Provider
 * エラーコンテキストプロバイダー
 */

import React, {
  createContext,
  useContext,
  useReducer,
  useEffect,
  ReactNode,
} from 'react';
import {
  setupGlobalErrorHandlers,
  GracefulDegradationManager,
} from '@/utils/errorHandling';
import { useToast } from '@/components/ui/Toast';

interface ErrorState {
  globalError: Error | null;
  networkStatus: 'online' | 'offline' | 'degraded';
  backendStatus: 'available' | 'unavailable' | 'degraded';
  retryCount: number;
  lastErrorTime: number | null;
  errorHistory: Array<{
    id: string;
    error: Error;
    timestamp: number;
    context?: string;
    resolved: boolean;
  }>;
}

type ErrorAction =
  | { type: 'SET_GLOBAL_ERROR'; payload: { error: Error; context?: string } }
  | { type: 'CLEAR_GLOBAL_ERROR' }
  | { type: 'SET_NETWORK_STATUS'; payload: 'online' | 'offline' | 'degraded' }
  | {
      type: 'SET_BACKEND_STATUS';
      payload: 'available' | 'unavailable' | 'degraded';
    }
  | { type: 'INCREMENT_RETRY' }
  | { type: 'RESET_RETRY' }
  | { type: 'RESOLVE_ERROR'; payload: string };

interface ErrorContextType {
  state: ErrorState;
  reportError: (error: Error, context?: string) => void;
  clearError: () => void;
  retry: () => void;
  resolveError: (errorId: string) => void;
  isInDegradedMode: () => boolean;
  canRetry: () => boolean;
}

const initialState: ErrorState = {
  globalError: null,
  networkStatus: 'online',
  backendStatus: 'available',
  retryCount: 0,
  lastErrorTime: null,
  errorHistory: [],
};

function errorReducer(state: ErrorState, action: ErrorAction): ErrorState {
  switch (action.type) {
    case 'SET_GLOBAL_ERROR': {
      const errorId = `error_${Date.now()}_${Math.random()
        .toString(36)
        .slice(2, 11)}`;
      return {
        ...state,
        globalError: action.payload.error,
        lastErrorTime: Date.now(),
        errorHistory: [
          ...state.errorHistory.slice(-9), // Keep only last 10 errors
          {
            id: errorId,
            error: action.payload.error,
            timestamp: Date.now(),
            context: action.payload.context,
            resolved: false,
          },
        ],
      };
    }

    case 'CLEAR_GLOBAL_ERROR':
      return {
        ...state,
        globalError: null,
      };

    case 'SET_NETWORK_STATUS':
      return {
        ...state,
        networkStatus: action.payload,
      };

    case 'SET_BACKEND_STATUS':
      return {
        ...state,
        backendStatus: action.payload,
      };

    case 'INCREMENT_RETRY':
      return {
        ...state,
        retryCount: state.retryCount + 1,
      };

    case 'RESET_RETRY':
      return {
        ...state,
        retryCount: 0,
      };

    case 'RESOLVE_ERROR':
      return {
        ...state,
        errorHistory: state.errorHistory.map(error =>
          error.id === action.payload ? { ...error, resolved: true } : error
        ),
      };

    default:
      return state;
  }
}

const ErrorContext = createContext<ErrorContextType | undefined>(undefined);

export const useErrorContext = () => {
  const context = useContext(ErrorContext);
  if (!context) {
    throw new Error('useErrorContext must be used within an ErrorProvider');
  }
  return context;
};

interface ErrorProviderProps {
  children: ReactNode;
}

export const ErrorProvider: React.FC<ErrorProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(errorReducer, initialState);
  const { addToast } = useToast();

  // Setup global error handlers on mount
  useEffect(() => {
    setupGlobalErrorHandlers();
  }, []);

  // Monitor network and backend status
  useEffect(() => {
    const degradationManager = GracefulDegradationManager.getInstance();

    const unsubscribe = degradationManager.subscribe(status => {
      dispatch({
        type: 'SET_NETWORK_STATUS',
        payload: status.offlineMode ? 'offline' : 'online',
      });

      dispatch({
        type: 'SET_BACKEND_STATUS',
        payload: status.backendAvailable ? 'available' : 'unavailable',
      });
    });

    return unsubscribe;
  }, []);

  // Auto-clear errors after a certain time
  useEffect(() => {
    if (state.globalError && state.lastErrorTime) {
      const timeout = setTimeout(() => {
        dispatch({ type: 'CLEAR_GLOBAL_ERROR' });
      }, 30000); // Clear after 30 seconds

      return () => clearTimeout(timeout);
    }
  }, [state.globalError, state.lastErrorTime]);

  const reportError = (error: Error, context?: string) => {
    console.error('Error reported:', error, context);

    dispatch({
      type: 'SET_GLOBAL_ERROR',
      payload: { error, context },
    });

    // Show toast notification
    let message = error.message;
    let title = 'エラーが発生しました';

    // Customize message based on error type
    if (error.name === 'NetworkError') {
      title = 'ネットワークエラー';
      message = 'ネットワーク接続を確認してください';
    } else if (error.name === 'ApiServiceError') {
      title = 'サーバーエラー';
    } else if (context) {
      title = `${context}でエラー`;
    }

    addToast({
      type: 'error',
      title,
      message,
      duration: 5000,
    });

    // Report to external service (if configured)
    reportToExternalService(error, context);
  };

  const clearError = () => {
    dispatch({ type: 'CLEAR_GLOBAL_ERROR' });
  };

  const retry = () => {
    dispatch({ type: 'INCREMENT_RETRY' });
    dispatch({ type: 'CLEAR_GLOBAL_ERROR' });

    // Show retry notification
    addToast({
      type: 'info',
      title: '再試行中',
      message: '操作を再試行しています...',
      duration: 2000,
    });
  };

  const resolveError = (errorId: string) => {
    dispatch({ type: 'RESOLVE_ERROR', payload: errorId });
  };

  const isInDegradedMode = () => {
    return (
      state.networkStatus !== 'online' || state.backendStatus !== 'available'
    );
  };

  const canRetry = () => {
    return state.retryCount < 3;
  };

  const contextValue: ErrorContextType = {
    state,
    reportError,
    clearError,
    retry,
    resolveError,
    isInDegradedMode,
    canRetry,
  };

  return (
    <ErrorContext.Provider value={contextValue}>
      {children}
    </ErrorContext.Provider>
  );
};

/**
 * Report error to external monitoring service
 */
function reportToExternalService(error: Error, context?: string) {
  try {
    // In a real application, you would send this to services like:
    // - Sentry: Sentry.captureException(error, { tags: { context } });
    // - LogRocket: LogRocket.captureException(error);
    // - Bugsnag: Bugsnag.notify(error, { context });

    // For now, we'll store it locally for debugging
    const errorReport = {
      message: error.message,
      stack: error.stack,
      name: error.name,
      context,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      userId: localStorage.getItem('userId') || 'anonymous',
    };

    // Store in localStorage for debugging
    const existingReports = JSON.parse(
      localStorage.getItem('errorReports') || '[]'
    );
    existingReports.push(errorReport);

    // Keep only last 50 reports
    if (existingReports.length > 50) {
      existingReports.splice(0, existingReports.length - 50);
    }

    localStorage.setItem('errorReports', JSON.stringify(existingReports));
  } catch (reportingError) {
    console.error(
      'Failed to report error to external service:',
      reportingError
    );
  }
}

/**
 * Hook for component-level error reporting
 */
export const useErrorReporting = () => {
  const { reportError, clearError } = useErrorContext();

  const reportComponentError = (error: Error, componentName: string) => {
    reportError(error, `Component: ${componentName}`);
  };

  const reportApiError = (error: Error, endpoint: string) => {
    reportError(error, `API: ${endpoint}`);
  };

  const reportUserActionError = (error: Error, action: string) => {
    reportError(error, `User Action: ${action}`);
  };

  return {
    reportError,
    reportComponentError,
    reportApiError,
    reportUserActionError,
    clearError,
  };
};

export default ErrorProvider;
