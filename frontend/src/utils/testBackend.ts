/**
 * Backend Integration Test Utility
 *
 * Tests all API endpoints to ensure proper integration with FastAPI backend
 */

import { apiService } from '@/services/api';

export interface TestResult {
  name: string;
  success: boolean;
  error?: string;
  duration: number;
  data?: any;
}

export interface TestSuite {
  name: string;
  results: TestResult[];
  totalTests: number;
  passedTests: number;
  failedTests: number;
  totalDuration: number;
}

/**
 * Run a single test with timing
 */
async function runTest(
  name: string,
  testFn: () => Promise<any>
): Promise<TestResult> {
  const startTime = performance.now();

  try {
    const data = await testFn();
    const duration = performance.now() - startTime;

    return {
      name,
      success: true,
      duration,
      data,
    };
  } catch (error) {
    const duration = performance.now() - startTime;

    return {
      name,
      success: false,
      error: error instanceof Error ? error.message : String(error),
      duration,
    };
  }
}

/**
 * Test basic API endpoints
 */
export async function testBasicEndpoints(): Promise<TestSuite> {
  const results: TestResult[] = [];

  // Test health check
  results.push(
    await runTest('Health Check', async () => {
      return await apiService.healthCheck();
    })
  );

  // Test models endpoint
  results.push(
    await runTest('Get Models', async () => {
      return await apiService.getModels();
    })
  );

  // Test system stats
  results.push(
    await runTest('System Stats', async () => {
      return await apiService.getSystemStats();
    })
  );

  return createTestSuite('Basic API Endpoints', results);
}

/**
 * Test agent-specific endpoints
 */
export async function testAgentEndpoints(): Promise<TestSuite> {
  const results: TestResult[] = [];

  // Test agent status
  results.push(
    await runTest('Agent Status', async () => {
      return await apiService.getAgentStatus();
    })
  );

  // Test available tools
  results.push(
    await runTest('Available Tools', async () => {
      return await apiService.getAvailableTools();
    })
  );

  // Test learning metrics
  results.push(
    await runTest('Learning Metrics', async () => {
      return await apiService.getLearningMetrics();
    })
  );

  // Test agent configuration
  results.push(
    await runTest('Update Agent Config', async () => {
      return await apiService.updateAgentConfig({
        temperature: 0.7,
        streaming_enabled: true,
      });
    })
  );

  return createTestSuite('Agent Endpoints', results);
}

/**
 * Test session management
 */
export async function testSessionManagement(): Promise<TestSuite> {
  const results: TestResult[] = [];
  let sessionId: string | undefined;

  // Test session creation
  results.push(
    await runTest('Create Session', async () => {
      const response = await apiService.createSession({
        user_id: 'test_user',
        session_name: 'Test Session',
      });
      sessionId = response.session_id;
      return response;
    })
  );

  // Test session retrieval
  if (sessionId) {
    results.push(
      await runTest('Get Session', async () => {
        return await apiService.getSession(sessionId!);
      })
    );

    // Test conversation history
    results.push(
      await runTest('Get Conversation History', async () => {
        return await apiService.getConversationHistory(sessionId!);
      })
    );
  }

  return createTestSuite('Session Management', results);
}

/**
 * Test chat functionality
 */
export async function testChatFunctionality(): Promise<TestSuite> {
  const results: TestResult[] = [];

  // Test basic chat completion
  results.push(
    await runTest('Basic Chat Completion', async () => {
      return await apiService.chatCompletion({
        model: 'deepseek-r1:7b',
        messages: [{ role: 'user', content: 'Hello, this is a test message.' }],
        temperature: 0.7,
        max_tokens: 100,
      });
    })
  );

  // Test agent chat completion
  results.push(
    await runTest('Agent Chat Completion', async () => {
      return await apiService.agentChatCompletion({
        model: 'agent',
        messages: [
          { role: 'user', content: 'Test agent processing capabilities.' },
        ],
        temperature: 0.7,
        max_tokens: 100,
      });
    })
  );

  // Test natural language processing
  results.push(
    await runTest('Natural Language Processing', async () => {
      return await apiService.processUserInput(
        'こんにちは、テストメッセージです。',
        undefined,
        true
      );
    })
  );

  return createTestSuite('Chat Functionality', results);
}

/**
 * Test streaming functionality
 */
export async function testStreamingFunctionality(): Promise<TestSuite> {
  const results: TestResult[] = [];

  // Test streaming chat
  results.push(
    await runTest('Streaming Chat', async () => {
      const chunks: any[] = [];
      const stream = apiService.streamChatCompletion({
        model: 'deepseek-r1:7b',
        messages: [
          {
            role: 'user',
            content: 'Generate a short test response for streaming.',
          },
        ],
        temperature: 0.7,
        max_tokens: 50,
        stream: true,
      });

      for await (const chunk of stream) {
        chunks.push(chunk);
        if (chunks.length >= 5) break; // Limit for testing
      }

      return { chunks: chunks.length, sample: chunks[0] };
    })
  );

  return createTestSuite('Streaming Functionality', results);
}

/**
 * Run comprehensive backend integration tests
 */
export async function runComprehensiveTests(): Promise<{
  suites: TestSuite[];
  summary: {
    totalSuites: number;
    totalTests: number;
    passedTests: number;
    failedTests: number;
    totalDuration: number;
    successRate: number;
  };
}> {
  const suites: TestSuite[] = [];

  console.log('🧪 Starting comprehensive backend integration tests...');

  // Run all test suites
  suites.push(await testBasicEndpoints());
  suites.push(await testAgentEndpoints());
  suites.push(await testSessionManagement());
  suites.push(await testChatFunctionality());
  suites.push(await testStreamingFunctionality());

  // Calculate summary
  const summary = {
    totalSuites: suites.length,
    totalTests: suites.reduce((sum, suite) => sum + suite.totalTests, 0),
    passedTests: suites.reduce((sum, suite) => sum + suite.passedTests, 0),
    failedTests: suites.reduce((sum, suite) => sum + suite.failedTests, 0),
    totalDuration: suites.reduce((sum, suite) => sum + suite.totalDuration, 0),
    successRate: 0,
  };

  summary.successRate =
    summary.totalTests > 0
      ? (summary.passedTests / summary.totalTests) * 100
      : 0;

  console.log('✅ Backend integration tests completed');
  console.log(
    `📊 Results: ${summary.passedTests}/${summary.totalTests} tests passed (${summary.successRate.toFixed(1)}%)`
  );

  return { suites, summary };
}

/**
 * Create test suite summary
 */
function createTestSuite(name: string, results: TestResult[]): TestSuite {
  const passedTests = results.filter(r => r.success).length;
  const failedTests = results.filter(r => !r.success).length;
  const totalDuration = results.reduce((sum, r) => sum + r.duration, 0);

  return {
    name,
    results,
    totalTests: results.length,
    passedTests,
    failedTests,
    totalDuration,
  };
}

/**
 * Format test results for console output
 */
export function formatTestResults(suites: TestSuite[]): string {
  let output = '\n🧪 Backend Integration Test Results\n';
  output += '='.repeat(50) + '\n\n';

  for (const suite of suites) {
    output += `📋 ${suite.name}\n`;
    output += `-`.repeat(30) + '\n';

    for (const result of suite.results) {
      const status = result.success ? '✅' : '❌';
      const duration = `${result.duration.toFixed(0)}ms`;
      output += `${status} ${result.name} (${duration})\n`;

      if (!result.success && result.error) {
        output += `   Error: ${result.error}\n`;
      }
    }

    const successRate = (suite.passedTests / suite.totalTests) * 100;
    output += `\n📊 ${suite.passedTests}/${suite.totalTests} passed (${successRate.toFixed(1)}%)\n\n`;
  }

  return output;
}

/**
 * Quick connection test
 */
export async function quickConnectionTest(): Promise<boolean> {
  try {
    const result = await apiService.testConnection();
    return result.healthy;
  } catch (error) {
    console.error('Quick connection test failed:', error);
    return false;
  }
}
