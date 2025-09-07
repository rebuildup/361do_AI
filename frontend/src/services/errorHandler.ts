/**
 * Error handling utilities
 */

import { ApiServiceError } from './api';

/**
 * API Error interface
 */
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  retryable: boolean;
}

/**
 * Error severity levels
 */
export const ErrorSeverity = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
} as const;

export type ErrorSeverity = (typeof ErrorSeverity)[keyof typeof ErrorSeverity];

/**
 * Error categories
 */
export const ErrorCategory = {
  NETWORK: 'network',
  AUTHENTICATION: 'authentication',
  VALIDATION: 'validation',
  SERVER: 'server',
  CLIENT: 'client',
  UNKNOWN: 'unknown',
} as const;

export type ErrorCategory = (typeof ErrorCategory)[keyof typeof ErrorCategory];

/**
 * Enhanced error information
 */
export interface ErrorInfo {
  id: string;
  message: string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  retryable: boolean;
  timestamp: Date;
  context?: Record<string, unknown>;
  originalError?: Error;
}

/**
 * Error handler class
 */
export class ErrorHandler {
  private static instance: ErrorHandler;
  private errorLog: ErrorInfo[] = [];
  private maxLogSize = 100;

  private constructor() {}

  /**
   * Get singleton instance
   */
  static getInstance(): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler();
    }
    return ErrorHandler.instance;
  }

  /**
   * Handle different types of errors
   */
  handleError(error: unknown, context?: Record<string, unknown>): ErrorInfo {
    const errorInfo = this.createErrorInfo(error, context);
    this.logError(errorInfo);
    return errorInfo;
  }

  /**
   * Create standardized error information
   */
  private createErrorInfo(
    error: unknown,
    context?: Record<string, unknown>
  ): ErrorInfo {
    const id = this.generateErrorId();
    const timestamp = new Date();

    if (error instanceof ApiServiceError) {
      return {
        id,
        message: error.message,
        category: this.categorizeApiError(error.code),
        severity: this.getSeverityForApiError(error.code),
        retryable: error.retryable,
        timestamp,
        context: { ...context, code: error.code, details: error.details },
        originalError: error,
      };
    }

    if (error instanceof Error) {
      return {
        id,
        message: error.message,
        category: ErrorCategory.CLIENT,
        severity: ErrorSeverity.MEDIUM,
        retryable: false,
        timestamp,
        context,
        originalError: error,
      };
    }

    return {
      id,
      message: 'An unknown error occurred',
      category: ErrorCategory.UNKNOWN,
      severity: ErrorSeverity.LOW,
      retryable: false,
      timestamp,
      context: { ...context, originalError: error },
    };
  }

  /**
   * Categorize API errors
   */
  private categorizeApiError(code: string): ErrorCategory {
    switch (code) {
      case 'UNAUTHORIZED':
      case 'FORBIDDEN':
        return ErrorCategory.AUTHENTICATION;
      case 'NOT_FOUND':
      case 'VALIDATION_ERROR':
        return ErrorCategory.VALIDATION;
      case 'INTERNAL_ERROR':
      case 'SERVICE_UNAVAILABLE':
        return ErrorCategory.SERVER;
      case 'NETWORK_ERROR':
      case 'TIMEOUT':
      case 'RATE_LIMITED':
        return ErrorCategory.NETWORK;
      default:
        return ErrorCategory.UNKNOWN;
    }
  }

  /**
   * Get severity for API errors
   */
  private getSeverityForApiError(code: string): ErrorSeverity {
    switch (code) {
      case 'INTERNAL_ERROR':
      case 'SERVICE_UNAVAILABLE':
        return ErrorSeverity.CRITICAL;
      case 'UNAUTHORIZED':
      case 'FORBIDDEN':
        return ErrorSeverity.HIGH;
      case 'NETWORK_ERROR':
      case 'TIMEOUT':
      case 'RATE_LIMITED':
        return ErrorSeverity.MEDIUM;
      default:
        return ErrorSeverity.LOW;
    }
  }

  /**
   * Generate unique error ID
   */
  private generateErrorId(): string {
    return `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Log error to internal storage
   */
  private logError(errorInfo: ErrorInfo): void {
    this.errorLog.unshift(errorInfo);

    // Keep log size manageable
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog = this.errorLog.slice(0, this.maxLogSize);
    }

    // Log to console based on severity
    this.logToConsole(errorInfo);
  }

  /**
   * Log to console with appropriate level
   */
  private logToConsole(errorInfo: ErrorInfo): void {
    const logData = {
      id: errorInfo.id,
      message: errorInfo.message,
      category: errorInfo.category,
      severity: errorInfo.severity,
      context: errorInfo.context,
    };

    switch (errorInfo.severity) {
      case ErrorSeverity.CRITICAL:
      case ErrorSeverity.HIGH:
        console.error('Error:', logData);
        break;
      case ErrorSeverity.MEDIUM:
        console.warn('Warning:', logData);
        break;
      case ErrorSeverity.LOW:
      default:
        console.info('Info:', logData);
        break;
    }
  }

  /**
   * Get user-friendly error message
   */
  getUserMessage(errorInfo: ErrorInfo): string {
    switch (errorInfo.category) {
      case ErrorCategory.NETWORK:
        if (errorInfo.retryable) {
          return 'ネットワークエラーが発生しました。しばらく待ってから再試行してください。';
        }
        return 'ネットワークに接続できません。インターネット接続を確認してください。';

      case ErrorCategory.AUTHENTICATION:
        return '認証に失敗しました。ログインし直してください。';

      case ErrorCategory.VALIDATION:
        return '入力内容に問題があります。内容を確認してください。';

      case ErrorCategory.SERVER:
        return 'サーバーエラーが発生しました。しばらく待ってから再試行してください。';

      default:
        return errorInfo.message || '予期しないエラーが発生しました。';
    }
  }

  /**
   * Get retry suggestion
   */
  getRetryAction(errorInfo: ErrorInfo): string | null {
    if (!errorInfo.retryable) {
      return null;
    }

    switch (errorInfo.category) {
      case ErrorCategory.NETWORK:
        return '再試行';
      case ErrorCategory.SERVER:
        return 'しばらく待ってから再試行';
      default:
        return '再試行';
    }
  }

  /**
   * Get recent errors
   */
  getRecentErrors(limit = 10): ErrorInfo[] {
    return this.errorLog.slice(0, limit);
  }

  /**
   * Clear error log
   */
  clearErrorLog(): void {
    this.errorLog = [];
  }

  /**
   * Get error statistics
   */
  getErrorStats(): {
    total: number;
    byCategory: Record<ErrorCategory, number>;
    bySeverity: Record<ErrorSeverity, number>;
  } {
    const stats = {
      total: this.errorLog.length,
      byCategory: {} as Record<ErrorCategory, number>,
      bySeverity: {} as Record<ErrorSeverity, number>,
    };

    // Initialize counters
    Object.values(ErrorCategory).forEach(category => {
      stats.byCategory[category] = 0;
    });
    Object.values(ErrorSeverity).forEach(severity => {
      stats.bySeverity[severity] = 0;
    });

    // Count errors
    this.errorLog.forEach(error => {
      stats.byCategory[error.category]++;
      stats.bySeverity[error.severity]++;
    });

    return stats;
  }
}

// Export singleton instance
export const errorHandler = ErrorHandler.getInstance();

// Utility functions
export const handleError = (
  error: unknown,
  context?: Record<string, unknown>
): ErrorInfo => {
  return errorHandler.handleError(error, context);
};

export const getUserErrorMessage = (error: unknown): string => {
  const errorInfo = handleError(error);
  return errorHandler.getUserMessage(errorInfo);
};

export const getRetryAction = (error: unknown): string | null => {
  const errorInfo = handleError(error);
  return errorHandler.getRetryAction(errorInfo);
};
