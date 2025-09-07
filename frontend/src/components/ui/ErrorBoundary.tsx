import React, { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: (error: Error, resetErrorBoundary: () => void) => ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  level?: 'page' | 'component' | 'feature';
  showDetails?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  retryCount: number;
}

export class ErrorBoundary extends Component<Props, State> {
  private retryTimeout: NodeJS.Timeout | null = null;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    this.setState({ errorInfo });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Report error to monitoring service (if available)
    this.reportError(error, errorInfo);
  }

  componentWillUnmount() {
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
    }
  }

  reportError = (error: Error, errorInfo: React.ErrorInfo) => {
    // In a real application, you would send this to an error reporting service
    // like Sentry, LogRocket, or Bugsnag
    try {
      const errorReport = {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        level: this.props.level || 'component',
      };

      // Store in localStorage for debugging
      const existingErrors = JSON.parse(
        localStorage.getItem('errorReports') || '[]'
      );
      existingErrors.push(errorReport);

      // Keep only last 10 errors
      if (existingErrors.length > 10) {
        existingErrors.splice(0, existingErrors.length - 10);
      }

      localStorage.setItem('errorReports', JSON.stringify(existingErrors));
    } catch (reportingError) {
      console.error('Failed to report error:', reportingError);
    }
  };

  resetErrorBoundary = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: this.state.retryCount + 1,
    });
  };

  autoRetry = () => {
    if (this.state.retryCount < 3) {
      this.retryTimeout = setTimeout(() => {
        this.resetErrorBoundary();
      }, 2000);
    }
  };

  reloadPage = () => {
    window.location.reload();
  };

  goHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError && this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.resetErrorBoundary);
      }

      const { level = 'component', showDetails = false } = this.props;
      const isPageLevel = level === 'page';
      const canAutoRetry = this.state.retryCount < 3;

      return (
        <div
          className={`flex flex-col items-center justify-center ${
            isPageLevel
              ? 'min-h-screen bg-black'
              : 'min-h-64 bg-gray-900/50 rounded-lg border border-gray-700'
          } text-white p-8`}
        >
          <div className="max-w-md text-center">
            <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />

            <h2 className="text-xl font-semibold mb-2 text-red-400">
              {isPageLevel ? 'アプリケーションエラー' : 'コンポーネントエラー'}
            </h2>

            <p className="text-gray-300 mb-4">
              {isPageLevel
                ? 'アプリケーションで予期しないエラーが発生しました。'
                : 'この機能で問題が発生しました。'}
            </p>

            {showDetails && (
              <details className="mb-4 text-left">
                <summary className="cursor-pointer text-sm text-gray-400 hover:text-gray-300">
                  エラー詳細を表示
                </summary>
                <div className="mt-2 p-3 bg-gray-800 rounded text-xs font-mono text-red-300 overflow-auto max-h-32">
                  <div className="mb-2">
                    <strong>エラー:</strong> {this.state.error.message}
                  </div>
                  {this.state.error.stack && (
                    <div>
                      <strong>スタックトレース:</strong>
                      <pre className="whitespace-pre-wrap text-xs">
                        {this.state.error.stack}
                      </pre>
                    </div>
                  )}
                </div>
              </details>
            )}

            <div className="flex flex-col gap-3">
              <button
                onClick={this.resetErrorBoundary}
                className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                再試行{' '}
                {this.state.retryCount > 0 &&
                  `(${this.state.retryCount + 1}回目)`}
              </button>

              {isPageLevel && (
                <>
                  <button
                    onClick={this.reloadPage}
                    className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
                  >
                    <RefreshCw className="w-4 h-4" />
                    ページを再読み込み
                  </button>

                  <button
                    onClick={this.goHome}
                    className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
                  >
                    <Home className="w-4 h-4" />
                    ホームに戻る
                  </button>
                </>
              )}

              <button
                onClick={() => {
                  const errorReport = {
                    error: this.state.error?.message,
                    stack: this.state.error?.stack,
                    timestamp: new Date().toISOString(),
                  };
                  navigator.clipboard?.writeText(
                    JSON.stringify(errorReport, null, 2)
                  );
                }}
                className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors text-sm"
              >
                <Bug className="w-4 h-4" />
                エラー情報をコピー
              </button>
            </div>

            {canAutoRetry && (
              <div className="mt-4 text-sm text-gray-400">
                <p>2秒後に自動的に再試行します...</p>
                {this.autoRetry()}
              </div>
            )}

            {this.state.retryCount >= 3 && (
              <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-700 rounded text-sm text-yellow-300">
                <p>
                  複数回の再試行が失敗しました。ページを再読み込みするか、しばらく時間をおいてから再度お試しください。
                </p>
              </div>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
