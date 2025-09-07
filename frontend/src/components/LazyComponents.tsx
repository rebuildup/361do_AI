/**
 * Lazy Component Definitions
 *
 * Lazy-loaded components for code splitting and performance optimization
 */

import { lazy } from 'react';

// Debug components (loaded only when needed)
export const BackendTester = lazy(() => import('./debug/BackendTester'));
export const NLPTester = lazy(() => import('./debug/NLPTester'));

// Chat components (core functionality - loaded immediately)
export const ChatInterface = lazy(() => import('./chat/ChatInterface'));
export const MessageBubble = lazy(() => import('./chat/MessageBubble'));
export const NaturalLanguageIndicator = lazy(
  () => import('./chat/NaturalLanguageIndicator')
);

// Layout components (loaded immediately for core UI)
export const SimpleLayout = lazy(() => import('./layout/SimpleLayout'));

// UI components (loaded on demand)
export const ShortcutsModal = lazy(() => import('./ui/ShortcutsModal'));
export const Toast = lazy(() =>
  import('./ui/Toast').then(module => ({ default: module.ToastProvider }))
);

// Utility components (loaded on demand)
export const LoadingSpinner = lazy(() =>
  import('./ui/LoadingSpinner').catch(() => ({
    default: () => (
      <div className="flex items-center justify-center p-4">
        <div className="w-6 h-6 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin" />
      </div>
    ),
  }))
);

// Error boundary fallback
export const ErrorFallback = lazy(() =>
  Promise.resolve({
    default: ({
      error,
      resetErrorBoundary,
    }: {
      error: Error;
      resetErrorBoundary: () => void;
    }) => (
      <div className="flex flex-col items-center justify-center p-8 bg-red-50 border border-red-200 rounded-lg">
        <h2 className="text-lg font-semibold text-red-800 mb-2">
          Something went wrong
        </h2>
        <p className="text-red-600 mb-4">{error.message}</p>
        <button
          onClick={resetErrorBoundary}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Try again
        </button>
      </div>
    ),
  })
);
