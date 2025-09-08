import React, { Suspense, lazy, memo, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { ErrorProvider, useErrorContext } from '@/contexts/ErrorContext';
import { AppProvider } from '@/contexts/AppContext';
import { ToastProvider } from '@/components/ui/Toast';
import {
  OfflineModeBanner,
  BackendDisconnectedBanner,
} from '@/components/ui/FallbackComponents';
import useAppResilience from '@/hooks/useAppResilience';
import { setupGlobalErrorHandlers } from '@/utils/errorHandling';
import './index.css';

// Lazy load components for better performance
const SimpleLayout = lazy(() => import('@/components/layout/SimpleLayout'));
const SimpleMainContent = lazy(() => import('@/components/SimpleMainContent'));
const BackendTester = lazy(() => import('@/components/debug/BackendTester'));

// Loading fallback component
const LoadingFallback = memo(() => (
  <div
    className="flex items-center justify-center min-h-screen"
    style={{ backgroundColor: '#000000' }}
  >
    <div className="flex flex-col items-center gap-4">
      <div
        className="w-8 h-8 border-2 rounded-full animate-spin"
        style={{ borderColor: '#181818', borderTopColor: '#1d9bf0' }}
      />
      <p className="text-sm" style={{ color: '#e7e9ea' }}>
        Loading...
      </p>
    </div>
  </div>
));

LoadingFallback.displayName = 'LoadingFallback';

// Error fallback component
const ErrorFallback = memo(
  ({
    error,
    resetErrorBoundary,
  }: {
    error: Error;
    resetErrorBoundary: () => void;
  }) => (
    <div
      className="flex items-center justify-center min-h-screen"
      style={{ backgroundColor: '#000000' }}
    >
      <div
        className="flex flex-col items-center gap-4 p-8 rounded-lg max-w-md border"
        style={{ backgroundColor: '#181818', borderColor: '#181818' }}
      >
        <AlertTriangle size={24} color="#1d9bf0" aria-label="Alert" />
        <h2 className="text-lg font-semibold" style={{ color: '#e7e9ea' }}>
          Something went wrong
        </h2>
        <p className="text-center text-sm" style={{ color: '#e7e9ea' }}>
          {error.message}
        </p>
        <button
          onClick={resetErrorBoundary}
          className="px-4 py-2 rounded transition-colors"
          style={{ backgroundColor: '#1d9bf0', color: '#000000' }}
        >
          Try again
        </button>
      </div>
    </div>
  )
);

ErrorFallback.displayName = 'ErrorFallback';

// Debug mode wrapper component
const DebugWrapper = memo(() => {
  const isDebugMode =
    import.meta.env.VITE_DEBUG_MODE === 'true' || import.meta.env.DEV;

  if (!isDebugMode) return null;

  return (
    <>
      <div className="mb-4">
        <Suspense
          fallback={<div className="h-32 bg-gray-900 rounded animate-pulse" />}
        >
          <BackendTester />
        </Suspense>
      </div>

      {/* Error testing component */}
      <Suspense fallback={null}>
        {React.createElement(
          lazy(() => import('@/components/debug/ErrorTester'))
        )}
      </Suspense>
    </>
  );
});

DebugWrapper.displayName = 'DebugWrapper';

// App content with resilience features
const AppContent = memo(() => {
  const [appStatus] = useAppResilience();
  const { reportError } = useErrorContext();

  // Setup global error handlers
  useEffect(() => {
    setupGlobalErrorHandlers();
  }, []);

  // Handle unhandled promise rejections
  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      reportError(
        new Error(event.reason?.toString() || 'Unhandled promise rejection'),
        'Global Promise Rejection'
      );
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    return () => {
      window.removeEventListener(
        'unhandledrejection',
        handleUnhandledRejection
      );
    };
  }, [reportError]);

  return (
    <AppProvider>
      <div
        className="min-h-screen"
        style={{ backgroundColor: '#000000', color: '#e7e9ea' }}
      >
        {/* Status banners */}
        {!appStatus.isOnline && <OfflineModeBanner />}
        {appStatus.isOnline && !appStatus.backendAvailable && (
          <BackendDisconnectedBanner onRetry={() => window.location.reload()} />
        )}

        {/* Main application */}
        <ErrorBoundary
          level="page"
          showDetails={import.meta.env.DEV}
          onError={(error, errorInfo) => {
            reportError(error, 'App Level Error');
            console.error('App Error:', error, errorInfo);
          }}
        >
          <Suspense fallback={<LoadingFallback />}>
            <SimpleLayout>
              <DebugWrapper />

              <ErrorBoundary
                level="component"
                onError={error => reportError(error, 'Main Content')}
              >
                <Suspense fallback={<LoadingFallback />}>
                  <SimpleMainContent />
                </Suspense>
              </ErrorBoundary>
            </SimpleLayout>
          </Suspense>
        </ErrorBoundary>
      </div>
    </AppProvider>
  );
});

AppContent.displayName = 'AppContent';

// Main App component with comprehensive error handling
const App = memo(() => {
  return (
    <ToastProvider>
      <ErrorProvider>
        <AppContent />
      </ErrorProvider>
    </ToastProvider>
  );
});

App.displayName = 'App';

export default App;
