/**
 * Backend Integration Tester Component
 *
 * Provides a UI for testing backend integration and API endpoints
 */

import React, { useState } from 'react';
import { Play, CheckCircle, XCircle, Clock, AlertTriangle } from 'lucide-react';
import {
  runComprehensiveTests,
  quickConnectionTest,
  type TestSuite,
} from '@/utils/testBackend';
import {
  runComprehensiveAgentTests,
  type AgentTestSuite,
} from '@/utils/agentIntegrationTest';
import NLPTester from './NLPTester';

interface BackendTesterProps {
  className?: string;
}

export const BackendTester: React.FC<BackendTesterProps> = ({
  className = '',
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [testResults, setTestResults] = useState<{
    suites: TestSuite[];
    summary: any;
  } | null>(null);
  const [agentTestResults, setAgentTestResults] = useState<{
    suites: AgentTestSuite[];
    overallSummary: any;
  } | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<boolean | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  const handleQuickTest = async () => {
    setError(null);
    try {
      const isConnected = await quickConnectionTest();
      setConnectionStatus(isConnected);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection test failed');
      setConnectionStatus(false);
    }
  };

  const handleComprehensiveTest = async () => {
    setIsRunning(true);
    setError(null);
    setTestResults(null);
    setAgentTestResults(null);

    try {
      const results = await runComprehensiveTests();
      setTestResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test execution failed');
    } finally {
      setIsRunning(false);
    }
  };

  const handleAgentTest = async () => {
    setIsRunning(true);
    setError(null);
    setTestResults(null);
    setAgentTestResults(null);

    try {
      const results = await runComprehensiveAgentTests();
      setAgentTestResults(results);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Agent test execution failed'
      );
    } finally {
      setIsRunning(false);
    }
  };

  const renderConnectionStatus = () => {
    if (connectionStatus === null) return null;

    return (
      <div
        className={`flex items-center gap-2 p-3 rounded-lg ${
          connectionStatus
            ? 'bg-green-50 text-green-700 border border-green-200'
            : 'bg-red-50 text-red-700 border border-red-200'
        }`}
      >
        {connectionStatus ? (
          <>
            <CheckCircle className="w-5 h-5" />
            <span>Backend connection successful</span>
          </>
        ) : (
          <>
            <XCircle className="w-5 h-5" />
            <span>Backend connection failed</span>
          </>
        )}
      </div>
    );
  };

  const renderTestSuite = (suite: TestSuite) => {
    const successRate = (suite.passedTests / suite.totalTests) * 100;

    return (
      <div
        key={suite.name}
        className="border border-gray-200 rounded-lg p-4 mb-4"
      >
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-gray-900">{suite.name}</h3>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">
              {suite.passedTests}/{suite.totalTests} passed
            </span>
            <div
              className={`px-2 py-1 rounded text-xs font-medium ${
                successRate === 100
                  ? 'bg-green-100 text-green-800'
                  : successRate >= 50
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-red-100 text-red-800'
              }`}
            >
              {successRate.toFixed(0)}%
            </div>
          </div>
        </div>

        <div className="space-y-2">
          {suite.results.map((result, index) => (
            <div
              key={index}
              className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded"
            >
              <div className="flex items-center gap-2">
                {result.success ? (
                  <CheckCircle className="w-4 h-4 text-green-600" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-600" />
                )}
                <span className="text-sm font-medium">{result.name}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <Clock className="w-3 h-3" />
                <span>{result.duration.toFixed(0)}ms</span>
              </div>
            </div>
          ))}
        </div>

        {suite.results.some(r => !r.success) && (
          <div className="mt-3 p-3 bg-red-50 rounded border border-red-200">
            <h4 className="text-sm font-medium text-red-800 mb-2">Errors:</h4>
            <div className="space-y-1">
              {suite.results
                .filter(r => !r.success)
                .map((result, index) => (
                  <div key={index} className="text-xs text-red-700">
                    <strong>{result.name}:</strong> {result.error}
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}
    >
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Backend Integration Tester
        </h2>
        <p className="text-gray-600 text-sm">
          Test the connection and functionality of the FastAPI backend
          integration.
        </p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
          <AlertTriangle className="w-5 h-5" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      <div className="space-y-6">
        {/* Quick Connection Test */}
        <div>
          <button
            onClick={handleQuickTest}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <CheckCircle className="w-4 h-4" />
            Quick Connection Test
          </button>

          <div className="mt-2">{renderConnectionStatus()}</div>
        </div>

        {/* Natural Language Processing Tester */}
        <div>
          <NLPTester />
        </div>

        {/* Comprehensive Test */}
        <div className="flex gap-2">
          <button
            onClick={handleComprehensiveTest}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Running Tests...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run API Tests
              </>
            )}
          </button>

          <button
            onClick={handleAgentTest}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Testing Agent...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Test Agent Features
              </>
            )}
          </button>
        </div>

        {/* Test Results */}
        {testResults && (
          <div className="mt-6">
            <div className="mb-4 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Test Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Total Tests:</span>
                  <div className="font-semibold">
                    {testResults.summary.totalTests}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Passed:</span>
                  <div className="font-semibold text-green-600">
                    {testResults.summary.passedTests}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Failed:</span>
                  <div className="font-semibold text-red-600">
                    {testResults.summary.failedTests}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Success Rate:</span>
                  <div className="font-semibold">
                    {testResults.summary.successRate.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              {testResults.suites.map(renderTestSuite)}
            </div>
          </div>
        )}

        {agentTestResults && (
          <div className="mt-6">
            <div className="mb-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
              <h3 className="font-semibold text-gray-900 mb-2">
                Agent Integration Test Summary
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Total Tests:</span>
                  <div className="font-semibold">
                    {agentTestResults.overallSummary.totalTests}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Passed:</span>
                  <div className="font-semibold text-green-600">
                    {agentTestResults.overallSummary.passedTests}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Failed:</span>
                  <div className="font-semibold text-red-600">
                    {agentTestResults.overallSummary.failedTests}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Success Rate:</span>
                  <div className="font-semibold">
                    {agentTestResults.overallSummary.overallSuccessRate.toFixed(
                      1
                    )}
                    %
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              {agentTestResults.suites.map(renderAgentTestSuite)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BackendTester;

const renderAgentTestSuite = (suite: AgentTestSuite) => {
  const successRate = suite.summary.successRate;

  return (
    <div
      key={suite.name}
      className="border border-purple-200 rounded-lg p-4 mb-4 bg-purple-50"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-900">{suite.name}</h3>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">
            {suite.summary.passed}/{suite.summary.total} passed
          </span>
          <div
            className={`px-2 py-1 rounded text-xs font-medium ${
              successRate === 100
                ? 'bg-green-100 text-green-800'
                : successRate >= 50
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-red-100 text-red-800'
            }`}
          >
            {successRate.toFixed(0)}%
          </div>
        </div>
      </div>

      <div className="space-y-2">
        {suite.results.map((result, index) => (
          <div
            key={index}
            className="flex items-center justify-between py-2 px-3 bg-white rounded border border-purple-100"
          >
            <div className="flex items-center gap-2">
              {result.success ? (
                <CheckCircle className="w-4 h-4 text-green-600" />
              ) : (
                <XCircle className="w-4 h-4 text-red-600" />
              )}
              <span className="text-sm font-medium">{result.testName}</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Clock className="w-3 h-3" />
              <span>{result.duration.toFixed(0)}ms</span>
            </div>
          </div>
        ))}
      </div>

      {suite.results.some(r => !r.success) && (
        <div className="mt-3 p-3 bg-red-50 rounded border border-red-200">
          <h4 className="text-sm font-medium text-red-800 mb-2">Errors:</h4>
          <div className="space-y-1">
            {suite.results
              .filter(r => !r.success)
              .map((result, index) => (
                <div key={index} className="text-xs text-red-700">
                  <strong>{result.testName}:</strong> {result.error}
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

//
