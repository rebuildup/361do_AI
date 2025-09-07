/**
 * Natural Language Processing Integration Test
 *
 * Comprehensive test suite for verifying NLP capabilities including
 * Japanese language support, tool invocation, conversation continuation,
 * and agent self-improvement features.
 */

import { naturalLanguageProcessor } from '@/services/naturalLanguageProcessor';
import { sessionPersistence } from '@/services/sessionPersistence';
import { apiService } from '@/services/api';

export interface NLPTestResult {
  testName: string;
  success: boolean;
  duration: number;
  details: any;
  error?: string;
}

export interface NLPTestSuite {
  name: string;
  results: NLPTestResult[];
  summary: {
    total: number;
    passed: number;
    failed: number;
    successRate: number;
    totalDuration: number;
  };
}

/**
 * Test Japanese language processing
 */
export async function testJapaneseLanguageProcessing(): Promise<NLPTestSuite> {
  const results: NLPTestResult[] = [];

  // Test basic Japanese input
  results.push(
    await runNLPTest('Basic Japanese Input', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input: 'こんにちは。今日はいい天気ですね。',
        language: 'ja',
      });

      return {
        hasResponse: !!response.response,
        detectedLanguage: response.language,
        responseLength: response.response.length,
        confidence: response.confidence,
      };
    })
  );

  // Test complex Japanese instructions
  results.push(
    await runNLPTest('Complex Japanese Instructions', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input:
          'システムの現在の状態を確認して、パフォーマンスの問題があれば詳細な分析を行ってください。',
        language: 'ja',
      });

      return {
        hasResponse: !!response.response,
        hasReasoning: !!response.reasoning,
        toolsDetected: response.toolsUsed?.length || 0,
        processingTime: response.processingTime,
      };
    })
  );

  // Test Japanese with technical terms
  results.push(
    await runNLPTest('Japanese Technical Terms', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input:
          'React Hooksの使い方について教えてください。useStateとuseEffectの違いも説明してください。',
        language: 'ja',
      });

      return {
        hasResponse: !!response.response,
        containsTechnicalContent:
          response.response.includes('React') ||
          response.response.includes('Hook'),
        confidence: response.confidence,
      };
    })
  );

  return createNLPTestSuite('Japanese Language Processing', results);
}

/**
 * Test tool invocation through natural language
 */
export async function testToolInvocation(): Promise<NLPTestSuite> {
  const results: NLPTestResult[] = [];

  // Test web search invocation
  results.push(
    await runNLPTest('Web Search Tool Invocation', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input: '最新のAI技術のトレンドについて調べて教えてください。',
        language: 'ja',
      });

      return {
        toolsUsed: response.toolsUsed || [],
        hasWebSearch: response.toolsUsed?.includes('web_search') || false,
        hasReasoning: !!response.reasoning,
        reasoningMentionsSearch:
          response.reasoning?.includes('検索') ||
          response.reasoning?.includes('search') ||
          false,
      };
    })
  );

  // Test file operations invocation
  results.push(
    await runNLPTest('File Operations Tool Invocation', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input:
          'プロジェクトの設定ファイルを確認して、必要があれば修正してください。',
        language: 'ja',
      });

      return {
        toolsUsed: response.toolsUsed || [],
        hasFileOps: response.toolsUsed?.includes('file_operations') || false,
        hasReasoning: !!response.reasoning,
        reasoningMentionsFiles:
          response.reasoning?.includes('ファイル') ||
          response.reasoning?.includes('file') ||
          false,
      };
    })
  );

  // Test system command invocation
  results.push(
    await runNLPTest('System Command Tool Invocation', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input:
          'Can you check the current system memory usage and CPU performance?',
        language: 'en',
      });

      return {
        toolsUsed: response.toolsUsed || [],
        hasCommandExecution:
          response.toolsUsed?.includes('command_execution') || false,
        hasReasoning: !!response.reasoning,
        reasoningMentionsSystem:
          response.reasoning?.includes('system') ||
          response.reasoning?.includes('システム') ||
          false,
      };
    })
  );

  return createNLPTestSuite('Tool Invocation', results);
}

/**
 * Test conversation continuation without session clearing
 */
export async function testConversationContinuation(): Promise<NLPTestSuite> {
  const results: NLPTestResult[] = [];

  // Create a test session
  const testSession =
    await sessionPersistence.createOrRestoreSession('nlp_test_user');

  // Test initial conversation
  results.push(
    await runNLPTest('Initial Conversation Setup', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input: 'React Hooksについて教えてください。',
        sessionId: testSession.sessionId,
        language: 'ja',
      });

      return {
        hasResponse: !!response.response,
        sessionUpdated: response.sessionUpdated,
        responseLength: response.response.length,
      };
    })
  );

  // Test follow-up question (context-dependent)
  results.push(
    await runNLPTest('Context-Dependent Follow-up', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input: 'それについてもう少し詳しく説明してもらえますか？',
        sessionId: testSession.sessionId,
        language: 'ja',
      });

      return {
        hasResponse: !!response.response,
        sessionUpdated: response.sessionUpdated,
        hasContextualResponse:
          response.response.includes('Hook') ||
          response.response.includes('React'),
        confidence: response.confidence,
      };
    })
  );

  // Test conversation history persistence
  results.push(
    await runNLPTest('Conversation History Persistence', async () => {
      const session = await sessionPersistence.getSession(
        testSession.sessionId
      );

      return {
        sessionExists: !!session,
        messageCount: session?.messages.length || 0,
        hasUserMessages:
          session?.messages.some(m => m.role === 'user') || false,
        hasAssistantMessages:
          session?.messages.some(m => m.role === 'assistant') || false,
      };
    })
  );

  return createNLPTestSuite('Conversation Continuation', results);
}

/**
 * Test agent self-improvement capabilities
 */
export async function testAgentSelfImprovement(): Promise<NLPTestSuite> {
  const results: NLPTestResult[] = [];

  // Test prompt rewriting capability
  results.push(
    await runNLPTest('Prompt Rewriting Detection', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input:
          'あなたの回答の質を向上させるために、プロンプトを改善してください。',
        language: 'ja',
      });

      return {
        hasResponse: !!response.response,
        promptRewritten: response.promptRewritten || false,
        hasReasoning: !!response.reasoning,
        confidence: response.confidence,
      };
    })
  );

  // Test tuning data manipulation
  results.push(
    await runNLPTest('Tuning Data Manipulation', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input:
          'この会話を学習データとして保存し、今後の回答品質向上に活用してください。',
        language: 'ja',
      });

      return {
        hasResponse: !!response.response,
        tuningDataUpdated: response.tuningDataUpdated || false,
        hasReasoning: !!response.reasoning,
        confidence: response.confidence,
      };
    })
  );

  // Test learning metrics access
  results.push(
    await runNLPTest('Learning Metrics Access', async () => {
      const metrics = await apiService.getLearningMetrics();

      return {
        hasMetrics: !!metrics,
        totalInteractions: metrics.total_interactions,
        successfulResponses: metrics.successful_responses,
        learningRate: metrics.learning_rate,
        hasRecentImprovements: metrics.recent_improvements.length > 0,
      };
    })
  );

  return createNLPTestSuite('Agent Self-Improvement', results);
}

/**
 * Test mixed language and edge cases
 */
export async function testMixedLanguageAndEdgeCases(): Promise<NLPTestSuite> {
  const results: NLPTestResult[] = [];

  // Test mixed Japanese and English
  results.push(
    await runNLPTest('Mixed Japanese and English', async () => {
      const response = await naturalLanguageProcessor.processInput({
        input:
          'Hello! こんにちは。Can you explain React Hooks in Japanese? Reactフックについて日本語で説明してください。',
        language: 'auto',
      });

      return {
        hasResponse: !!response.response,
        detectedLanguage: response.language,
        handlesMultipleLanguages: response.response.length > 50,
        confidence: response.confidence,
      };
    })
  );

  // Test empty input handling
  results.push(
    await runNLPTest('Empty Input Handling', async () => {
      try {
        const response = await naturalLanguageProcessor.processInput({
          input: '',
          language: 'ja',
        });

        return {
          hasResponse: !!response.response,
          handlesEmptyInput: true,
          confidence: response.confidence,
        };
      } catch (error) {
        return {
          hasResponse: false,
          handlesEmptyInput: false,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    })
  );

  // Test very long input
  results.push(
    await runNLPTest('Long Input Handling', async () => {
      const longInput = 'これは非常に長い入力テストです。'.repeat(50);

      const response = await naturalLanguageProcessor.processInput({
        input: longInput,
        language: 'ja',
      });

      return {
        hasResponse: !!response.response,
        handlesLongInput: response.response.length > 0,
        processingTime: response.processingTime,
        confidence: response.confidence,
      };
    })
  );

  return createNLPTestSuite('Mixed Language and Edge Cases', results);
}

/**
 * Run comprehensive NLP integration tests
 */
export async function runComprehensiveNLPTests(): Promise<{
  suites: NLPTestSuite[];
  overallSummary: {
    totalSuites: number;
    totalTests: number;
    passedTests: number;
    failedTests: number;
    overallSuccessRate: number;
    totalDuration: number;
  };
}> {
  console.log('🧠 Starting comprehensive NLP integration tests...');

  const suites: NLPTestSuite[] = [];

  // Run all test suites
  suites.push(await testJapaneseLanguageProcessing());
  suites.push(await testToolInvocation());
  suites.push(await testConversationContinuation());
  suites.push(await testAgentSelfImprovement());
  suites.push(await testMixedLanguageAndEdgeCases());

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

  console.log('✅ NLP integration tests completed');
  console.log(
    `🎯 Overall Results: ${overallSummary.passedTests}/${overallSummary.totalTests} tests passed (${overallSummary.overallSuccessRate.toFixed(1)}%)`
  );

  return { suites, overallSummary };
}

/**
 * Helper function to run a single NLP test
 */
async function runNLPTest(
  testName: string,
  testFn: () => Promise<any>
): Promise<NLPTestResult> {
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
 * Create NLP test suite summary
 */
function createNLPTestSuite(
  name: string,
  results: NLPTestResult[]
): NLPTestSuite {
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
