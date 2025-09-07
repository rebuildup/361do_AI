/**
 * Services index
 *
 * Centralized exports for all service modules
 */

// API Service
export { ApiService, ApiServiceError, apiService } from "./api";

// Error Handler
export {
  ErrorHandler,
  ErrorSeverity,
  ErrorCategory,
  errorHandler,
  handleError,
  getUserErrorMessage,
  getRetryAction,
} from "./errorHandler";

// Types
export type { ErrorInfo } from "./errorHandler";
