/**
 * Natural Language Processing Tester
 *
 * Component for testing and demonstrating NLP capabilities
 */

import React, { useState } from 'react';
import {
  Play,
  Brain,
  MessageSquare,
  Zap,
  CheckCircle,
  XCircle,
  Clock,
} from 'lucide-react';
import { naturalLanguageProcessor } from '@/services/naturalLanguageProcessor';
import { cn } from '@/utils';

interface NLPTestCase {
  id: string;
  name: string;
  input: string;
  language: 'ja' | 'en';
  expectedTools?: string[];
  description: string;
}

const testCases: NLPTestCase[] = [
  {
    id: 'japanese_web_search',
    name: '日本語Web検索',
    input: '最新のAI技術について調べて教えてください',
    language: 'ja',
    expectedTools: ['web_search'],
    description: '日本語での自然言語によるWeb検索要求',
  },
  {
    id: 'english_system_info',
    name: 'English System Check',
    input: 'Can you check the current system status and memory usage?',
    language: 'en',
    expectedTools: ['command_execution'],
    description: 'English natural language system information request',
  },
  {
    id: 'japanese_file_ops',
    name: '日本語ファイル操作',
    input: 'プロジェクトの設定ファイルを確認して、必要があれば修正してください',
    language: 'ja',
    expectedTools: ['file_operations'],
    description: '日本語でのファイル操作要求',
  },
  {
    id: 'mixed_language',
    name: 'Mixed Language',
    input:
      'Hello! こんにちは。Can you search for information about React hooks?',
    language: 'ja',
    expectedTools: ['web_search'],
    description: '混合言語での自然言語処理',
  },
  {
    id: 'complex_reasoning',
    name: '複雑な推論',
    input:
      'システムの性能を分析して、最適化の提案をしてください。必要に応じて外部ツールも使用してください。',
    language: 'ja',
    expectedTools: ['command_execution', 'web_search'],
    description: '複雑な推論と複数ツール使用',
  },
  {
    id: 'conversation_continuation',
    name: '会話継続',
    input: 'それについてもう少し詳しく説明してもらえますか？',
    language: 'ja',
    expectedTools: [],
    description: '文脈を考慮した会話継続',
  },
];

interface TestResult {
  testId: string;
  success: boolean;
  response: string;
  reasoning?: string;
  toolsUsed: string[];
  confidence: number;
  processingTime: number;
  error?: string;
}

export const NLPTester: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<TestResult[]>([]);
  // const [_selectedTest, _setSelectedTest] = useState<string | null>(null);
  const [customInput, setCustomInput] = useState('');
  const [customLanguage, setCustomLanguage] = useState<'ja' | 'en'>('ja');

  const runSingleTest = async (testCase: NLPTestCase): Promise<TestResult> => {
    const startTime = performance.now();

    try {
      const response = await naturalLanguageProcessor.processInput({
        input: testCase.input,
        language: testCase.language,
        sessionId: 'nlp-test-session',
      });

      const processingTime = performance.now() - startTime;

      return {
        testId: testCase.id,
        success: true,
        response: response.response,
        reasoning: response.reasoning,
        toolsUsed: response.toolsUsed || [],
        confidence: response.confidence || 0,
        processingTime,
      };
    } catch (error) {
      const processingTime = performance.now() - startTime;

      return {
        testId: testCase.id,
        success: false,
        response: '',
        toolsUsed: [],
        confidence: 0,
        processingTime,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  };

  const runAllTests = async () => {
    setIsRunning(true);
    setResults([]);

    try {
      const testResults: TestResult[] = [];

      for (const testCase of testCases) {
        const result = await runSingleTest(testCase);
        testResults.push(result);
        setResults([...testResults]); // Update UI progressively

        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } finally {
      setIsRunning(false);
    }
  };

  const runCustomTest = async () => {
    if (!customInput.trim()) return;

    setIsRunning(true);

    try {
      const customTestCase: NLPTestCase = {
        id: 'custom',
        name: 'Custom Test',
        input: customInput,
        language: customLanguage,
        description: 'Custom user input test',
      };

      const result = await runSingleTest(customTestCase);
      setResults([result]);
    } finally {
      setIsRunning(false);
    }
  };

  const getTestCaseById = (id: string) => testCases.find(t => t.id === id);

  const renderTestResult = (result: TestResult) => {
    const testCase = getTestCaseById(result.testId);
    if (!testCase) return null;

    const expectedToolsMatch = testCase.expectedTools
      ? testCase.expectedTools.every(tool => result.toolsUsed.includes(tool))
      : true;

    return (
      <div
        key={result.testId}
        className="border border-gray-700 rounded-lg p-4 bg-gray-900/50"
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {result.success ? (
              <CheckCircle className="w-5 h-5 text-green-400" />
            ) : (
              <XCircle className="w-5 h-5 text-red-400" />
            )}
            <h3 className="font-medium text-white">{testCase.name}</h3>
            <span className="text-xs px-2 py-1 bg-gray-800 rounded text-gray-300">
              {testCase.language.toUpperCase()}
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Clock className="w-3 h-3" />
            <span>{result.processingTime.toFixed(0)}ms</span>
          </div>
        </div>

        <div className="space-y-2 text-sm">
          <div>
            <span className="text-gray-400">Input:</span>
            <div className="text-gray-300 bg-gray-800 rounded p-2 mt-1">
              {testCase.input}
            </div>
          </div>

          {result.success && (
            <>
              <div>
                <span className="text-gray-400">Response:</span>
                <div className="text-gray-300 bg-gray-800 rounded p-2 mt-1">
                  {result.response}
                </div>
              </div>

              {result.reasoning && (
                <div>
                  <span className="text-gray-400">Reasoning:</span>
                  <div className="text-gray-300 bg-gray-800 rounded p-2 mt-1 text-xs">
                    {result.reasoning}
                  </div>
                </div>
              )}

              <div className="flex items-center gap-4">
                <div>
                  <span className="text-gray-400">Tools Used:</span>
                  <div className="flex gap-1 mt-1">
                    {result.toolsUsed.length > 0 ? (
                      result.toolsUsed.map((tool, index) => (
                        <span
                          key={index}
                          className={cn(
                            'px-2 py-1 rounded text-xs',
                            expectedToolsMatch
                              ? 'bg-green-800 text-green-200'
                              : 'bg-yellow-800 text-yellow-200'
                          )}
                        >
                          {tool}
                        </span>
                      ))
                    ) : (
                      <span className="text-gray-500 text-xs">None</span>
                    )}
                  </div>
                </div>

                <div>
                  <span className="text-gray-400">Confidence:</span>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-16 bg-gray-800 rounded-full h-2">
                      <div
                        className={cn(
                          'h-full rounded-full',
                          result.confidence >= 0.8
                            ? 'bg-green-400'
                            : result.confidence >= 0.6
                              ? 'bg-yellow-400'
                              : 'bg-red-400'
                        )}
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-300">
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </>
          )}

          {result.error && (
            <div className="text-red-400 bg-red-900/20 rounded p-2 text-xs">
              Error: {result.error}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-2 flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-600" />
          Natural Language Processing Tester
        </h2>
        <p className="text-gray-600 text-sm">
          Test the agent's natural language processing capabilities including
          Japanese support, tool invocation, and conversation continuation.
        </p>
      </div>

      {/* Test Controls */}
      <div className="space-y-4 mb-6">
        <div>
          <button
            onClick={runAllTests}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Running Tests...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run All NLP Tests
              </>
            )}
          </button>
        </div>

        {/* Custom Test Input */}
        <div className="border border-gray-300 rounded-lg p-4">
          <h3 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
            <MessageSquare className="w-4 h-4" />
            Custom Test
          </h3>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Language
              </label>
              <select
                value={customLanguage}
                onChange={e => setCustomLanguage(e.target.value as 'ja' | 'en')}
                className="w-32 px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                <option value="ja">Japanese</option>
                <option value="en">English</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Input Text
              </label>
              <textarea
                value={customInput}
                onChange={e => setCustomInput(e.target.value)}
                placeholder="Enter your natural language input..."
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                rows={3}
              />
            </div>
            <button
              onClick={runCustomTest}
              disabled={isRunning || !customInput.trim()}
              className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
              <Zap className="w-4 h-4" />
              Test Custom Input
            </button>
          </div>
        </div>
      </div>

      {/* Test Results */}
      {results.length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-900 mb-4">Test Results</h3>
          <div className="space-y-4">{results.map(renderTestResult)}</div>

          {/* Summary */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">Summary</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Total Tests:</span>
                <div className="font-semibold">{results.length}</div>
              </div>
              <div>
                <span className="text-gray-600">Passed:</span>
                <div className="font-semibold text-green-600">
                  {results.filter(r => r.success).length}
                </div>
              </div>
              <div>
                <span className="text-gray-600">Failed:</span>
                <div className="font-semibold text-red-600">
                  {results.filter(r => !r.success).length}
                </div>
              </div>
              <div>
                <span className="text-gray-600">Avg Confidence:</span>
                <div className="font-semibold">
                  {results.length > 0
                    ? (
                        (results.reduce((sum, r) => sum + r.confidence, 0) /
                          results.length) *
                        100
                      ).toFixed(0)
                    : 0}
                  %
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NLPTester;
