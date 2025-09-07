/**
 * MSW Request Handlers
 *
 * Mock API handlers for testing
 */

import { http, HttpResponse } from 'msw';
import { mockApiResponses } from '../utils';

const API_BASE_URL = 'http://localhost:8000/v1';

export const handlers = [
  // Health check
  http.get(`${API_BASE_URL}/health`, () => {
    return HttpResponse.json(mockApiResponses.healthCheck);
  }),

  // Models
  http.get(`${API_BASE_URL}/models`, () => {
    return HttpResponse.json(mockApiResponses.models);
  }),

  // Chat completions
  http.post(`${API_BASE_URL}/chat/completions`, async ({ request }) => {
    const body = (await request.json()) as any;

    if (body.stream) {
      // Return streaming response
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          const chunks = [
            'Hello',
            ' there',
            '! How',
            ' can I',
            ' help you',
            ' today?',
          ];

          chunks.forEach((chunk, index) => {
            setTimeout(() => {
              const data = {
                id: 'chatcmpl-123',
                object: 'chat.completion.chunk',
                created: Date.now(),
                model: body.model,
                choices: [
                  {
                    index: 0,
                    delta: { content: chunk },
                    finish_reason: null,
                  },
                ],
              };

              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
              );

              if (index === chunks.length - 1) {
                const endData = {
                  id: 'chatcmpl-123',
                  object: 'chat.completion.chunk',
                  created: Date.now(),
                  model: body.model,
                  choices: [
                    {
                      index: 0,
                      delta: {},
                      finish_reason: 'stop',
                    },
                  ],
                };
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify(endData)}\n\n`)
                );
                controller.enqueue(encoder.encode('data: [DONE]\n\n'));
                controller.close();
              }
            }, index * 100);
          });
        },
      });

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/plain',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
      });
    }

    return HttpResponse.json(mockApiResponses.chatCompletion);
  }),

  // Agent chat completions
  http.post(`${API_BASE_URL}/chat/completions/agent`, async ({ request }) => {
    await request.json(); // Consume request body

    const response = {
      ...mockApiResponses.chatCompletion,
      reasoning: 'This is a test reasoning from the agent.',
      model: 'agent',
    };

    return HttpResponse.json(response);
  }),

  // Agent status
  http.get(`${API_BASE_URL}/agent/status`, () => {
    return HttpResponse.json(mockApiResponses.agentStatus);
  }),

  // Agent configuration
  http.post(`${API_BASE_URL}/agent/config`, async ({ request }) => {
    await request.json(); // Consume request body

    return HttpResponse.json({
      success: true,
      message: 'Configuration updated successfully',
    });
  }),

  // Available tools
  http.get(`${API_BASE_URL}/agent/tools`, () => {
    return HttpResponse.json({
      tools: [
        {
          name: 'web_search',
          description: 'Search the web for information',
          enabled: true,
          parameters: {
            query: 'string',
            max_results: 'number',
          },
        },
        {
          name: 'file_operations',
          description: 'Perform file operations',
          enabled: true,
          parameters: {
            action: 'string',
            path: 'string',
          },
        },
        {
          name: 'command_execution',
          description: 'Execute system commands',
          enabled: false,
          parameters: {
            command: 'string',
          },
        },
      ],
    });
  }),

  // Tool execution
  http.post(`${API_BASE_URL}/agent/tools/execute`, async ({ request }) => {
    const body = (await request.json()) as any;

    return HttpResponse.json({
      result: `Executed tool ${body.tool_name} with parameters: ${JSON.stringify(body.parameters)}`,
      execution_time: 1.5,
      success: true,
    });
  }),

  // Learning metrics
  http.get(`${API_BASE_URL}/agent/learning/metrics`, () => {
    return HttpResponse.json(mockApiResponses.learningMetrics);
  }),

  // Feedback
  http.post(`${API_BASE_URL}/agent/feedback`, async ({ request }) => {
    await request.json(); // Consume request body

    return HttpResponse.json({
      success: true,
      message: 'Feedback received and processed',
    });
  }),

  // Sessions
  http.post(`${API_BASE_URL}/sessions`, async ({ request }) => {
    const body = (await request.json()) as any;

    return HttpResponse.json({
      session_id: 'test-session-123',
      created_at: new Date().toISOString(),
      user_id: body.user_id || 'test-user',
      session_name: body.session_name || 'Test Session',
      metadata: body.metadata || {},
    });
  }),

  http.get(`${API_BASE_URL}/sessions/:sessionId`, ({ params: _params }) => {
    return HttpResponse.json({
      session_id: _params.sessionId,
      created_at: '2024-01-01T10:00:00Z',
      user_id: 'test-user',
      session_name: 'Test Session',
      metadata: {},
      status: 'active',
    });
  }),

  // Conversation history
  http.get(
    `${API_BASE_URL}/sessions/:sessionId/messages`,
    ({ params: _params, request }) => {
      const url = new URL(request.url);
      const limit = parseInt(url.searchParams.get('limit') || '50');

      const messages = Array.from({ length: Math.min(limit, 10) }, (_, i) => ({
        id: `msg-${i}`,
        role: i % 2 === 0 ? 'user' : 'assistant',
        content: `Test message ${i}`,
        timestamp: new Date(Date.now() - (10 - i) * 60000).toISOString(),
        reasoning: i % 2 === 1 ? 'Test reasoning' : undefined,
        tools_used: i % 4 === 1 ? ['web_search'] : [],
        processing_time: i % 2 === 1 ? 1.5 : undefined,
      }));

      return HttpResponse.json({
        messages,
        total_count: messages.length,
      });
    }
  ),

  // Clear conversation history
  http.delete(`${API_BASE_URL}/sessions/:sessionId/messages`, () => {
    return HttpResponse.json({
      success: true,
      message: 'Conversation history cleared',
    });
  }),

  // Export conversation
  http.get(
    `${API_BASE_URL}/sessions/:sessionId/export`,
    ({ params, request }) => {
      const url = new URL(request.url);
      const format = url.searchParams.get('format') || 'json';

      const data = {
        session_id: params.sessionId,
        exported_at: new Date().toISOString(),
        messages: [
          {
            id: 'msg-1',
            role: 'user',
            content: 'Hello',
            timestamp: '2024-01-01T10:00:00Z',
          },
          {
            id: 'msg-2',
            role: 'assistant',
            content: 'Hello! How can I help you?',
            timestamp: '2024-01-01T10:00:01Z',
          },
        ],
      };

      let content: string;
      let contentType: string;

      switch (format) {
        case 'markdown':
          content = `# Conversation Export\n\n**Session ID:** ${params.sessionId}\n\n## Messages\n\n### User\nHello\n\n### Assistant\nHello! How can I help you?`;
          contentType = 'text/markdown';
          break;
        case 'txt':
          content = `Conversation Export\nSession ID: ${params.sessionId}\n\nUser: Hello\nAssistant: Hello! How can I help you?`;
          contentType = 'text/plain';
          break;
        default:
          content = JSON.stringify(data, null, 2);
          contentType = 'application/json';
      }

      return new Response(content, {
        headers: {
          'Content-Type': contentType,
          'Content-Disposition': `attachment; filename=conversation_${params.sessionId}.${format}`,
        },
      });
    }
  ),

  // System stats
  http.post(`${API_BASE_URL}/system/stats`, async ({ request }) => {
    const body = (await request.json()) as any;

    const response: any = {
      timestamp: new Date().toISOString(),
      cpu: {
        usage_percent: 45.5,
        temperature: 65.0,
      },
      memory: {
        usage_percent: 60.2,
        total_gb: 32.0,
        used_gb: 19.2,
      },
    };

    if (body.include_gpu) {
      response.gpu = {
        usage_percent: 70.0,
        memory_percent: 75.0,
        temperature: 75.0,
      };
    }

    if (body.include_processes) {
      response.processes = {
        count: 150,
        top_cpu: 'python.exe',
      };
    }

    return HttpResponse.json(response);
  }),

  // Inference endpoint
  http.post(`${API_BASE_URL}/inference`, async ({ request }) => {
    const body = (await request.json()) as any;

    return HttpResponse.json({
      id: 'inference-123',
      response: `Response to: ${body.prompt}`,
      reasoning_steps: [
        {
          step: 1,
          description: 'Analyzing user input',
          confidence: 0.9,
        },
        {
          step: 2,
          description: 'Generating response',
          confidence: 0.85,
        },
      ],
      confidence_score: 0.87,
      processing_time: 1.2,
      memory_usage: {
        gpu_memory_mb: 100,
        cpu_memory_mb: 50,
      },
      model_info: {
        model: body.model || 'deepseek-r1:7b',
        version: '1.0',
      },
    });
  }),

  // Memory search
  http.post(`${API_BASE_URL}/memory/search`, async ({ request }) => {
    const body = (await request.json()) as any;

    return HttpResponse.json({
      results: [
        {
          id: 'memory_1',
          content: `Memory result for query: ${body.query}`,
          relevance_score: 0.9,
          timestamp: new Date().toISOString(),
        },
        {
          id: 'memory_2',
          content: `Another memory result for: ${body.query}`,
          relevance_score: 0.8,
          timestamp: new Date().toISOString(),
        },
      ],
      total_found: 2,
      query: body.query,
      search_time: 0.1,
    });
  }),
];

// Error handlers for testing error scenarios
export const errorHandlers = [
  http.get(`${API_BASE_URL}/health`, () => {
    return HttpResponse.json({ error: 'Service unavailable' }, { status: 503 });
  }),

  http.post(`${API_BASE_URL}/chat/completions`, () => {
    return HttpResponse.json({ error: 'Model not available' }, { status: 503 });
  }),

  http.get(`${API_BASE_URL}/agent/status`, () => {
    return HttpResponse.json(
      { error: 'Agent not responding' },
      { status: 500 }
    );
  }),
];
