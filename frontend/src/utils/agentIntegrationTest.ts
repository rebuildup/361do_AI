/**
 * Agent Integration Test
 *
 * Comprehensive test for agent functionality including natural language processing,
 * tool usage, self-learning capabilities, and session management.
 */

import { apiService } from '@/services/api';

export interface AgentTestResult {
  testName: string;
  success: boolean;
  duration: number;
  details: any;
  error?: string;
}

export interface AgentTestSuite {
  name: string;
  results: AgentTestResult[];
  summary: {
    total: number;
    passed: number;
    failed: number;
    successRate: number;
    totalDuration: number;
  };
}

/**
 * Test natural language processing capabilities
 */
export async function testNaturalLanguageProcessing(): Promise<AgentTestSuite> {
  const results: AgentTestResult[] = [];

  // Test Japanese input processing
  results.push(
    await runAgentTest('Japanese Natural Language Input', async () => {
      const response = await apiService.processUserInput(
        'こんにちは。今日の天気について教えてください。',
        undefined,
        true
      );

      return {
        hasResponse: !!response,
        responseLength:
          typeof response === 'object' && 'response' in response
            ? response.response.length
            : 0,
        processingTime:
          typeof response === 'object' && 'processing_time' in response
            ? response.processing_time
            : 0,
      };
    })
  );

  // Test English input processing
  results.push(
    await runAgentTest('English Natural Language Input', async () => {
      const response = await apiService.processUserInput(
        'Hello, can you help me understand how to use this system?',
        undefined,
        true
      );

      return {
        hasResponse: !!response,
        responseLength:
          typeof response === 'object' && 'response' in response
            ? response.response.length
            : 0,
      };
    })
  );

  // Test complex instruction processing
  results.push(
    await runAgentTest('Complex Instruction Processing', async () => {
      const response = await apiService.processUserInput(
        'Please analyze the current system status and provide recommendations for optimization.',
        undefined,
        true
      );

      return {
        hasResponse: !!response,
        hasReasoningSteps:
          typeof response === 'object' && 'reasoning_steps' in response
            ? !!response.reasoning_steps
            : false,
      };
    })
  );

  return createAgentTestSuite('Natural Language Processing', results);
}

/**
 * Test agent tool integration
 */
export async function testAgentToolIntegration(): Promise<AgentTestSuite> {
  const results: AgentTestResult[] = [];

  // Test tool availability
  results.push(
    await runAgentTest('Tool Availability Check', async () => {
      const toolsResponse = await apiService.getAvailableTools();
      return {
        toolCount: toolsResponse.tools.length,
        tools: toolsResponse.tools.map(t => t.name),
        enabledTools: toolsResponse.tools.filter(t => t.enabled).length,
      };
    })
  );

  // Test tool execution (if tools are available)
  results.push(
    await runAgentTest('Tool Execution Test', async () => {
      const toolsResponse = await apiService.getAvailableTools();

      if (toolsResponse.tools.length === 0) {
        return { message: 'No tools available for testing' };
      }

      // Try to execute the first available tool with minimal parameters
      const firstTool = toolsResponse.tools[0];
      const executionResult = await apiService.executeTool(
        firstTool.name,
        {},
        'test-session'
      );

      return {
        toolName: firstTool.name,
        executionSuccess: executionResult.success,
        executionTime: executionResult.execution_time,
        hasResult: !!executionResult.result,
      };
    })
  );

  return createAgentTestSuite('Agent Tool Integration', results);
}

/**
 * Test self-learning capabilities
 */
export async function testSelfLearningCapabilities(): Promise<AgentTestSuite> {
  const results: AgentTestResult[] = [];

  // Test learning metrics retrieval
  results.push(
    await runAgentTest('Learning Metrics Retrieval', async () => {
      const metrics = await apiService.getLearningMetrics();
      return {
        totalInteractions: metrics.total_interactions,
        successfulResponses: metrics.successful_responses,
        averageResponseTime: metrics.average_response_time,
        learningRate: metrics.learning_rate,
        hasRecentImprovements: metrics.recent_improvements.length > 0,
      };
    })
  );

  // Test feedback submission
  results.push(
    await runAgentTest('Feedback Submission', async () => {
      const feedbackResult = await apiService.provideFeedback(
        'test-response-id',
        {
          rating: 5,
          helpful: true,
          comments: 'Test feedback for integration testing',
        }
      );

      return {
        feedbackAccepted: feedbackResult.success,
        message: feedbackResult.message,
      };
    })
  );

  return createAgentTestSuite('Self-Learning Capabilities', results);
}

/**
 * Test session management and persistence
 */
export async function testSessionManagement(): Promise<AgentTestSuite> {
  const results: AgentTestResult[] = [];
  let testSessionId: string | undefined;

  // Test session creation
  results.push(
    await runAgentTest('Session Creation', async () => {
      const session = await apiService.createSession({
        user_id: 'integration-test-user',
        session_name: 'Integration Test Session',
        metadata: { test: true, timestamp: Date.now() },
      });

      testSessionId = session.session_id;

      return {
        sessionId: session.session_id,
        userId: session.user_id,
        sessionName: session.session_name,
        hasMetadata: !!session.metadata,
      };
    })
  );

  // Test session retrieval
  if (testSessionId) {
    results.push(
      await runAgentTest('Session Retrieval', async () => {
        const session = await apiService.getSession(testSessionId!);
        return {
          sessionFound: !!session,
          sessionId: session.session_id,
          status: session.status || 'unknown',
        };
      })
    );

    // Test conversation history
    results.push(
      await runAgentTest('Conversation History', async () => {
        const history = await apiService.getConversationHistory(testSessionId!);
        return {
          messageCount: history.messages.length,
          totalCount: history.total_count,
          hasMessages: history.messages.length > 0,
        };
      })
    );
  }

  return createAgentTestSuite('Session Management', results);
}

/**
 * Test streaming functionality
 */
export async function testStreamingFunctionality(): Promise<AgentTestSuite> {
  const results: AgentTestResult[] = [];

  // Test basic streaming
  results.push(
    await runAgentTest('Basic Streaming Response', async () => {
      const chunks: any[] = [];
      let totalChars = 0;

      const stream = apiService.streamChatCompletion({
        model: 'deepseek-r1:7b',
        messages: [
          {
            role: 'user',
            content: 'Generate a short response about AI capabilities.',
          },
        ],
        temperature: 0.7,
        max_tokens: 100,
        stream: true,
      });

      for await (const chunk of stream) {
        chunks.push(chunk);
        const content = (chunk as any)?.choices?.[0]?.delta?.content;
        if (typeof content === 'string') {
          totalChars += content.length;
        }

        // Limit chunks for testing
        if (chunks.length >= 10) break;
      }

      return {
        chunkCount: chunks.length,
        totalCharacters: totalChars,
        hasContent: totalChars > 0,
        sampleChunk: chunks[0],
      };
    })
  );

  // Test agent streaming
  results.push(
    await runAgentTest('Agent Streaming Response', async () => {
      const chunks: any[] = [];

      const stream = apiService.streamChatCompletion(
        {
          model: 'agent',
          messages: [
            {
              role: 'user',
              content: 'Explain the benefits of AI in simple terms.',
            },
          ],
          temperature: 0.7,
          max_tokens: 100,
          stream: true,
        },
        true
      );

      for await (const chunk of stream) {
        chunks.push(chunk);
        if (chunks.length >= 5) break; // Limit for testing
      }

      return {
        chunkCount: chunks.length,
        hasAgentProcessing: chunks.some(c => c.model === 'agent'),
        sampleChunk: chunks[0],
      };
    })
  );

  return createAgentTestSuite('Streaming Functionality', results);
}

/**
 * Run comprehensive agent integration tests
 */
export async function runComprehensiveAgentTests(): Promise<{
  suites: AgentTestSuite[];
  overallSummary: {
    totalSuites: number;
    totalTests: number;
    passedTests: number;
    failedTests: number;
    overallSuccessRate: number;
    totalDuration: number;
  };
}> {
  console.log('🤖 Starting comprehensive agent integration tests...');

  const suites: AgentTestSuite[] = [];

  // Run all test suites
  suites.push(await testNaturalLanguageProcessing());
  suites.push(await testAgentToolIntegration());
  suites.push(await testSelfLearningCapabilities());
  suites.push(await testSessionManagement());
  suites.push(await testStreamingFunctionality());

  // Calculate overall summary
  const overallSummary = {
    totalSuites: suites.length,
    totalTests: suites.reduce((sum, suite) => sum + suite.summary.total, 0),
    passedTests: suites.reduce((sum, suite) => sum + suite.summary.passed, 0),
    failedTests: suites.reduce((sum, suite) => sum + suite.summary.failed, 0),
    overallSuccessRate: 0,
    totalDuration: suites.reduce(
      (sum, suite) => sum + suite.summary.totalDuration,
      0
    ),
  };

  overallSummary.overallSuccessRate =
    overallSummary.totalTests > 0
      ? (overallSummary.passedTests / overallSummary.totalTests) * 100
      : 0;

  console.log('✅ Agent integration tests completed');
  console.log(
    `🎯 Overall Results: ${overallSummary.passedTests}/${overallSummary.totalTests} tests passed (${overallSummary.overallSuccessRate.toFixed(1)}%)`
  );

  return { suites, overallSummary };
}

/**
 * Helper function to run a single agent test
 */
async function runAgentTest(
  testName: string,
  testFn: () => Promise<any>
): Promise<AgentTestResult> {
  const startTime = performance.now();

  try {
    const details = await testFn();
    const duration = performance.now() - startTime;

    return {
      testName,
      success: true,
      duration,
      details,
    };
  } catch (error) {
    const duration = performance.now() - startTime;

    return {
      testName,
      success: false,
      duration,
      details: {},
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Create agent test suite summary
 */
function createAgentTestSuite(
  name: string,
  results: AgentTestResult[]
): AgentTestSuite {
  const passed = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  const totalDuration = results.reduce((sum, r) => sum + r.duration, 0);
  const successRate = results.length > 0 ? (passed / results.length) * 100 : 0;

  return {
    name,
    results,
    summary: {
      total: results.length,
      passed,
      failed,
      successRate,
      totalDuration,
    },
  };
}
