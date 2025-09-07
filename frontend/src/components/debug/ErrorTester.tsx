/**
 * Error Testing Component
 * エラーテストコンポーネント
 */

import React, { useState } from 'react';
import {
  AlertTriangle,
  WifiOff,
  Server,
  Bug,
  RefreshCw,
  Zap,
} from 'lucide-react';
import { useErrorReporting } from '@/contexts/ErrorContext';
import { useToast } from '@/components/ui/Toast';
import { apiService } from '@/services/api';

const ErrorTester: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false);
  const { reportError, reportApiError, reportComponentError } =
    useErrorReporting();
  const { addToast } = useToast();

  // Only show in development mode
  if (!import.meta.env.DEV) {
    return null;
  }

  const testScenarios = [
    {
      name: 'JavaScript Error',
      icon: Bug,
      color: 'text-red-400',
      action: () => {
        throw new Error('Test JavaScript error for error boundary testing');
      },
    },
    {
      name: 'Network Error',
      icon: WifiOff,
      color: 'text-orange-400',
      action: () => {
        const networkError = new Error('Network connection failed');
        networkError.name = 'NetworkError';
        reportError(networkError, 'Network Test');
      },
    },
    {
      name: 'API Error',
      icon: Server,
      color: 'text-yellow-400',
      action: async () => {
        try {
          // Try to call a non-existent endpoint
          await fetch('/api/non-existent-endpoint');
        } catch (error) {
          reportApiError(error as Error, '/api/non-existent-endpoint');
        }
      },
    },
    {
      name: 'Component Error',
      icon: AlertTriangle,
      color: 'text-purple-400',
      action: () => {
        const componentError = new Error('Component rendering failed');
        reportComponentError(componentError, 'ErrorTester');
      },
    },
    {
      name: 'Async Error',
      icon: Zap,
      color: 'text-blue-400',
      action: async () => {
        // Simulate an async operation that fails
        await new Promise((_, reject) => {
          setTimeout(() => {
            reject(new Error('Async operation failed'));
          }, 1000);
        });
      },
    },
    {
      name: 'Backend Health Check',
      icon: RefreshCw,
      color: 'text-green-400',
      action: async () => {
        try {
          const health = await apiService.healthCheck();
          addToast({
            type: 'success',
            title: 'Health Check',
            message: `Backend status: ${health.status}`,
          });
        } catch (error) {
          reportApiError(error as Error, 'Health Check');
        }
      },
    },
  ];

  if (!isVisible) {
    return (
      <button
        onClick={() => setIsVisible(true)}
        className="fixed bottom-4 right-4 p-2 bg-red-600 text-white rounded-full shadow-lg hover:bg-red-700 transition-colors z-50"
        title="Error Testing (Dev Only)"
      >
        <Bug className="w-5 h-5" />
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 bg-gray-900 border border-gray-700 rounded-lg p-4 shadow-xl z-50 max-w-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Bug className="w-5 h-5 text-red-400" />
          Error Tester
        </h3>
        <button
          onClick={() => setIsVisible(false)}
          className="text-gray-400 hover:text-white"
        >
          ×
        </button>
      </div>

      <div className="space-y-2">
        {testScenarios.map((scenario, index) => (
          <button
            key={index}
            onClick={async () => {
              try {
                await scenario.action();
              } catch (error) {
                reportError(error as Error, `Test: ${scenario.name}`);
              }
            }}
            className="w-full flex items-center gap-3 p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-left"
          >
            <scenario.icon className={`w-5 h-5 ${scenario.color}`} />
            <span className="text-white text-sm">{scenario.name}</span>
          </button>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="text-xs text-gray-400 space-y-1">
          <p>• Test various error scenarios</p>
          <p>• Verify error boundaries work</p>
          <p>• Check fallback mechanisms</p>
          <p>• Validate retry logic</p>
        </div>
      </div>

      <div className="mt-3 flex gap-2">
        <button
          onClick={() => {
            localStorage.clear();
            addToast({
              type: 'info',
              title: 'Storage Cleared',
              message: 'All cached data has been cleared',
            });
          }}
          className="flex-1 px-3 py-2 bg-yellow-600 text-white text-xs rounded hover:bg-yellow-700 transition-colors"
        >
          Clear Cache
        </button>
        <button
          onClick={() => {
            const errorReports = localStorage.getItem('errorReports');
            if (errorReports) {
              navigator.clipboard?.writeText(errorReports);
              addToast({
                type: 'success',
                title: 'Copied',
                message: 'Error reports copied to clipboard',
              });
            }
          }}
          className="flex-1 px-3 py-2 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 transition-colors"
        >
          Copy Errors
        </button>
      </div>
    </div>
  );
};

export default ErrorTester;
