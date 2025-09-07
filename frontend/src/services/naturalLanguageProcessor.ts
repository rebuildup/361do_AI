/**
 * Natural Language Processor
 *
 * Handles natural language processing for agent interactions including
 * Japanese language support, tool invocation, and conversation management.
 */

import { apiService } from './api';
import type { ChatCompletionRequest, ChatMessage } from '@/types';

export interface NLPRequest {
  input: string;
  sessionId?: string;
  language?: 'ja' | 'en' | 'auto';
  context?: {
    previousMessages?: ChatMessage[];
    userPreferences?: Record<string, any>;
    sessionMetadata?: Record<string, any>;
  };
}

export interface NLPResponse {
  response: string;
  reasoning?: string;
  toolsUsed?: string[];
  confidence?: number;
  language: 'ja' | 'en';
  processingTime: number;
  sessionUpdated?: boolean;
  promptRewritten?: boolean;
  tuningDataUpdated?: boolean;
}

export interface ToolInvocation {
  toolName: string;
  parameters: Record<string, any>;
  reasoning: string;
}

export interface ConversationContext {
  sessionId: string;
  messageHistory: ChatMessage[];
  userProfile?: {
    language: 'ja' | 'en';
    preferences: Record<string, any>;
  };
  agentState?: {
    learningEpoch: number;
    totalInteractions: number;
    rewardScore: number;
  };
}

/**
 * Natural Language Processor class
 */
export class NaturalLanguageProcessor {
  private conversationContexts: Map<string, ConversationContext> = new Map();

  /**
   * Process natural language input through the agent
   */
  async processInput(request: NLPRequest): Promise<NLPResponse> {
    const startTime = performance.now();

    try {
      // Detect language if not specified
      const detectedLanguage =
        request.language === 'auto'
          ? this.detectLanguage(request.input)
          : request.language || 'ja';

      // Build conversation context
      const context = await this.buildConversationContext(request);

      // Enhance input with context and instructions
      const enhancedInput = this.enhanceInputWithContext(
        request.input,
        context,
        detectedLanguage
      );

      // Process through agent with natural language understanding
      const agentResponse = await this.processWithAgent(
        enhancedInput,
        request.sessionId
      );

      // Extract tool usage and reasoning
      const toolsUsed = this.extractToolUsage(agentResponse);
      const reasoning = this.extractReasoning(agentResponse);

      // Update conversation context
      if (request.sessionId) {
        await this.updateConversationContext(
          request.sessionId,
          request.input,
          agentResponse.response
        );
      }

      const processingTime = performance.now() - startTime;

      return {
        response: agentResponse.response,
        reasoning,
        toolsUsed,
        confidence: this.calculateConfidence(agentResponse),
        language: detectedLanguage,
        processingTime,
        sessionUpdated: !!request.sessionId,
        promptRewritten: agentResponse.promptRewritten || false,
        tuningDataUpdated: agentResponse.tuningDataUpdated || false,
      };
    } catch (error) {
      console.error('Natural language processing error:', error);

      // Fallback response
      return {
        response: this.generateFallbackResponse(
          request.input,
          request.language || 'ja'
        ),
        language: request.language || 'ja',
        processingTime: performance.now() - startTime,
        confidence: 0.1,
      };
    }
  }

  /**
   * Process streaming natural language input
   */
  async *processStreamingInput(request: NLPRequest): AsyncGenerator<{
    content: string;
    reasoning?: string;
    toolsUsed?: string[];
    isComplete: boolean;
  }> {
    try {
      // Build context
      const context = await this.buildConversationContext(request);
      const detectedLanguage =
        request.language === 'auto'
          ? this.detectLanguage(request.input)
          : request.language || 'ja';

      // Enhanced input for streaming
      const enhancedInput = this.enhanceInputWithContext(
        request.input,
        context,
        detectedLanguage
      );

      // Create chat completion request for streaming
      const chatRequest: ChatCompletionRequest = {
        model: 'agent',
        messages: [{ role: 'user', content: enhancedInput }],
        temperature: 0.7,
        max_tokens: 2000,
        stream: true,
      };

      let accumulatedContent = '';
      let reasoning = '';
      let toolsUsed: string[] = [];

      // Stream through agent
      for await (const chunk of apiService.streamChatCompletion(
        chatRequest,
        true
      )) {
        if (chunk.choices?.[0]?.delta?.content) {
          accumulatedContent += chunk.choices[0].delta.content;

          // Extract tools and reasoning as they come in
          toolsUsed = this.extractToolUsage({ response: accumulatedContent });
          reasoning = this.extractReasoning({ response: accumulatedContent });

          yield {
            content: chunk.choices[0].delta.content,
            reasoning: reasoning || undefined,
            toolsUsed: toolsUsed.length > 0 ? toolsUsed : undefined,
            isComplete: false,
          };
        }

        if (chunk.choices?.[0]?.finish_reason === 'stop') {
          // Update conversation context
          if (request.sessionId) {
            await this.updateConversationContext(
              request.sessionId,
              request.input,
              accumulatedContent
            );
          }

          yield {
            content: '',
            reasoning,
            toolsUsed,
            isComplete: true,
          };
          break;
        }
      }
    } catch (error) {
      console.error('Streaming natural language processing error:', error);

      // Fallback streaming response
      const fallbackResponse = this.generateFallbackResponse(
        request.input,
        request.language || 'ja'
      );

      for (let i = 0; i < fallbackResponse.length; i++) {
        yield {
          content: fallbackResponse[i],
          isComplete: i === fallbackResponse.length - 1,
        };

        // Small delay for streaming effect
        await new Promise(resolve => setTimeout(resolve, 30));
      }
    }
  }

  /**
   * Detect language from input text
   */
  private detectLanguage(input: string): 'ja' | 'en' {
    // Simple language detection based on character patterns
    const japanesePattern = /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]/;
    const englishPattern = /^[a-zA-Z0-9\s.,!?'"()-]+$/;

    if (japanesePattern.test(input)) {
      return 'ja';
    } else if (englishPattern.test(input)) {
      return 'en';
    }

    // Default to Japanese for mixed or unclear content
    return 'ja';
  }

  /**
   * Build conversation context for enhanced processing
   */
  private async buildConversationContext(
    request: NLPRequest
  ): Promise<ConversationContext> {
    if (!request.sessionId) {
      return {
        sessionId: 'temp',
        messageHistory: [],
      };
    }

    // Get existing context or create new one
    let context = this.conversationContexts.get(request.sessionId);

    if (!context) {
      try {
        // Fetch conversation history from API
        const history = await apiService.getConversationHistory(
          request.sessionId
        );

        // Get agent status for context
        const agentStatus = await apiService.getAgentStatus();

        context = {
          sessionId: request.sessionId,
          messageHistory: history.messages.map(msg => ({
            role: msg.role,
            content: msg.content,
          })),
          agentState: {
            learningEpoch: agentStatus.learning_metrics?.learning_epoch || 0,
            totalInteractions:
              agentStatus.learning_metrics?.total_interactions || 0,
            rewardScore: agentStatus.learning_metrics?.reward_score || 0,
          },
        };

        this.conversationContexts.set(request.sessionId, context);
      } catch (error) {
        console.warn('Failed to build conversation context:', error);
        context = {
          sessionId: request.sessionId,
          messageHistory: [],
        };
      }
    }

    return context;
  }

  /**
   * Enhance input with conversation context and natural language instructions
   */
  private enhanceInputWithContext(
    input: string,
    context: ConversationContext,
    language: 'ja' | 'en'
  ): string {
    const instructions =
      language === 'ja'
        ? {
            systemPrompt: `あなたは高度なAIエージェントです。以下の能力を持っています：

1. **自然言語理解**: ユーザーの意図を正確に理解し、適切なツールを自発的に使用
2. **ツール使用**: Web検索、コマンド実行、ファイル操作、MCP連携を自然言語で判断
3. **継続学習**: 会話を通じて学習し、プロンプトやチューニングデータを自己改善
4. **推論表示**: 思考過程を明確に示し、なぜそのツールを使用するかを説明

現在のセッション情報：
- セッションID: ${context.sessionId}
- 学習エポック: ${context.agentState?.learningEpoch || 0}
- 総インタラクション数: ${context.agentState?.totalInteractions || 0}
- 報酬スコア: ${context.agentState?.rewardScore || 0}

会話履歴（最新5件）：
${context.messageHistory
  .slice(-5)
  .map(
    msg =>
      `${msg.role === 'user' ? 'ユーザー' : 'エージェント'}: ${msg.content}`
  )
  .join('\n')}

重要な指示：
- ユーザーの要求に応じて、必要なツールを**自発的に判断して使用**してください
- Web検索が必要な場合は「検索して」と推論で示してください
- ファイル操作が必要な場合は「ファイルを確認」と推論で示してください
- コマンド実行が必要な場合は「システム情報を確認」と推論で示してください
- 推論過程を明確に示し、なぜそのアクションを取るかを説明してください

ユーザーの質問：`,

            contextNote: `\n\n上記の会話履歴を参考にして、継続的で自然な対話を心がけてください。`,
          }
        : {
            systemPrompt: `You are an advanced AI agent with the following capabilities:

1. **Natural Language Understanding**: Accurately understand user intent and proactively use appropriate tools
2. **Tool Usage**: Web search, command execution, file operations, MCP integration based on natural language judgment
3. **Continuous Learning**: Learn through conversations and self-improve prompts and tuning data
4. **Reasoning Display**: Clearly show thought processes and explain why specific tools are used

Current session information:
- Session ID: ${context.sessionId}
- Learning Epoch: ${context.agentState?.learningEpoch || 0}
- Total Interactions: ${context.agentState?.totalInteractions || 0}
- Reward Score: ${context.agentState?.rewardScore || 0}

Conversation History (last 5):
${context.messageHistory
  .slice(-5)
  .map(msg => `${msg.role === 'user' ? 'User' : 'Agent'}: ${msg.content}`)
  .join('\n')}

Important Instructions:
- **Proactively judge and use** necessary tools based on user requests
- When web search is needed, indicate "searching for" in your reasoning
- When file operations are needed, indicate "checking files" in your reasoning  
- When command execution is needed, indicate "checking system info" in your reasoning
- Clearly show your reasoning process and explain why you take specific actions

User's question:`,

            contextNote: `\n\nPlease refer to the above conversation history for continuous and natural dialogue.`,
          };

    return `${instructions.systemPrompt}\n${input}${instructions.contextNote}`;
  }

  /**
   * Process input through the agent with enhanced natural language understanding
   */
  private async processWithAgent(
    input: string,
    sessionId?: string
  ): Promise<{
    response: string;
    promptRewritten?: boolean;
    tuningDataUpdated?: boolean;
  }> {
    try {
      // Try agent-enhanced processing first
      const response = await apiService.processUserInput(
        input,
        sessionId,
        true
      );

      return {
        response:
          typeof response === 'object' && 'response' in response
            ? response.response
            : response.choices?.[0]?.message?.content ||
              'No response generated',
        promptRewritten:
          typeof response === 'object' && 'agent_state' in response,
        tuningDataUpdated:
          typeof response === 'object' && 'agent_state' in response,
      };
    } catch (error) {
      console.warn('Agent processing failed, using fallback:', error);

      // Fallback to basic chat completion
      const chatRequest: ChatCompletionRequest = {
        model: 'deepseek-r1:7b',
        messages: [{ role: 'user', content: input }],
        temperature: 0.7,
        max_tokens: 1000,
      };

      const response = await apiService.chatCompletion(chatRequest);
      return {
        response: response.choices[0].message.content,
      };
    }
  }

  /**
   * Extract tool usage from agent response
   */
  private extractToolUsage(response: any): string[] {
    const tools: string[] = [];

    if (typeof response === 'object' && 'tool_usage' in response) {
      return response.tool_usage.tools_used || [];
    }

    // Extract from response text patterns
    const responseText =
      typeof response === 'object' ? response.response : response;
    if (typeof responseText === 'string') {
      // Look for tool usage indicators in Japanese and English
      if (
        responseText.includes('検索して') ||
        responseText.includes('searching for')
      ) {
        tools.push('web_search');
      }
      if (
        responseText.includes('ファイルを確認') ||
        responseText.includes('checking files')
      ) {
        tools.push('file_operations');
      }
      if (
        responseText.includes('システム情報を確認') ||
        responseText.includes('checking system')
      ) {
        tools.push('command_execution');
      }
      if (
        responseText.includes('外部ツール') ||
        responseText.includes('external tool')
      ) {
        tools.push('mcp_integration');
      }
    }

    return tools;
  }

  /**
   * Extract reasoning from agent response
   */
  private extractReasoning(response: any): string {
    if (typeof response === 'object') {
      if ('reasoning' in response) {
        return response.reasoning;
      }
      if (
        'reasoning_steps' in response &&
        Array.isArray(response.reasoning_steps)
      ) {
        return response.reasoning_steps
          .map((step: any) => step.description || step)
          .join('\n');
      }
    }

    return '';
  }

  /**
   * Calculate confidence score for the response
   */
  private calculateConfidence(response: any): number {
    // Simple confidence calculation based on response characteristics
    let confidence = 0.5; // Base confidence

    if (typeof response === 'object') {
      // Higher confidence if reasoning is provided
      if ('reasoning' in response && response.reasoning) {
        confidence += 0.2;
      }

      // Higher confidence if tools were used
      if (
        'tool_usage' in response &&
        response.tool_usage?.tools_used?.length > 0
      ) {
        confidence += 0.2;
      }

      // Higher confidence if processing time is reasonable
      if ('processing_time' in response && response.processing_time < 5000) {
        confidence += 0.1;
      }
    }

    return Math.min(confidence, 1.0);
  }

  /**
   * Update conversation context with new messages
   */
  private async updateConversationContext(
    sessionId: string,
    userInput: string,
    agentResponse: string
  ): Promise<void> {
    const context = this.conversationContexts.get(sessionId);
    if (context) {
      // Add new messages to history
      context.messageHistory.push(
        { role: 'user', content: userInput },
        { role: 'assistant', content: agentResponse }
      );

      // Keep only last 20 messages for performance
      if (context.messageHistory.length > 20) {
        context.messageHistory = context.messageHistory.slice(-20);
      }

      this.conversationContexts.set(sessionId, context);
    }
  }

  /**
   * Generate fallback response when agent processing fails
   */
  private generateFallbackResponse(
    input: string,
    language: 'ja' | 'en'
  ): string {
    if (language === 'ja') {
      return `申し訳ございません。「${input}」についてのご質問を理解しましたが、現在システムに一時的な問題が発生しています。\n\n基本的な機能は動作していますので、もう一度お試しいただくか、より具体的な質問をしていただけますでしょうか。\n\nご不便をおかけして申し訳ありません。`;
    } else {
      return `I apologize, but I understand your question about "${input}" however there's currently a temporary system issue.\n\nBasic functions are working, so please try again or ask a more specific question.\n\nI apologize for the inconvenience.`;
    }
  }

  /**
   * Clear conversation context for a session
   */
  clearConversationContext(sessionId: string): void {
    this.conversationContexts.delete(sessionId);
  }

  /**
   * Get conversation context for debugging
   */
  getConversationContext(sessionId: string): ConversationContext | undefined {
    return this.conversationContexts.get(sessionId);
  }
}

// Export singleton instance
export const naturalLanguageProcessor = new NaturalLanguageProcessor();

// Export for testing and custom instances
export default NaturalLanguageProcessor;
