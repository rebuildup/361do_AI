/**
 * Natural Language Processing E2E Tests
 *
 * End-to-end tests for natural language processing features
 */

import { test, expect } from '@playwright/test';

test.describe('Natural Language Processing', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Wait for chat interface to be ready
    await page.waitForSelector('textarea[placeholder*="メッセージを入力"]');
  });

  test('should process Japanese natural language input', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send Japanese message
    await messageInput.fill('こんにちは、今日の天気について教えてください。');
    await sendButton.click();

    // Should show NLP indicator
    await expect(page.locator('text=自然言語処理')).toBeVisible({
      timeout: 5000,
    });

    // Should show language indicator
    await expect(page.locator('text=JA')).toBeVisible();

    // Wait for response
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });
  });

  test('should process English natural language input', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send English message
    await messageInput.fill(
      'Hello, can you help me understand how this system works?'
    );
    await sendButton.click();

    // Should show NLP indicator
    await expect(page.locator('text=Natural Language Processing')).toBeVisible({
      timeout: 5000,
    });

    // Should show language indicator
    await expect(page.locator('text=EN')).toBeVisible();

    // Wait for response
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });
  });

  test('should detect and invoke web search tool', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send message that should trigger web search
    await messageInput.fill(
      '最新のAI技術のトレンドについて調べて教えてください。'
    );
    await sendButton.click();

    // Should show web search tool usage
    await expect(page.locator('text=Web検索')).toBeVisible({ timeout: 10000 });

    // Should show reasoning that mentions search
    await expect(page.locator('text=検索')).toBeVisible({ timeout: 10000 });
  });

  test('should detect and invoke file operations tool', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send message that should trigger file operations
    await messageInput.fill(
      'プロジェクトの設定ファイルを確認して、必要があれば修正してください。'
    );
    await sendButton.click();

    // Should show file operations tool usage
    await expect(page.locator('text=ファイル操作')).toBeVisible({
      timeout: 10000,
    });

    // Should show reasoning that mentions files
    await expect(page.locator('text=ファイル')).toBeVisible({ timeout: 10000 });
  });

  test('should detect and invoke system command tool', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send message that should trigger system commands
    await messageInput.fill(
      'Can you check the current system memory usage and CPU performance?'
    );
    await sendButton.click();

    // Should show command execution tool usage
    await expect(page.locator('text=Command Execution')).toBeVisible({
      timeout: 10000,
    });

    // Should show reasoning that mentions system
    await expect(page.locator('text=system')).toBeVisible({ timeout: 10000 });
  });

  test('should handle mixed language input', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send mixed language message
    await messageInput.fill(
      'Hello! こんにちは。Can you explain React Hooks in Japanese? Reactフックについて日本語で説明してください。'
    );
    await sendButton.click();

    // Should detect and process mixed language
    await expect(page.locator('text=自然言語処理')).toBeVisible({
      timeout: 5000,
    });

    // Wait for response
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });
  });

  test('should show confidence scores', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send clear, unambiguous message
    await messageInput.fill(
      'This is a clear and simple request for information.'
    );
    await sendButton.click();

    // Should show confidence indicator
    await expect(page.locator('text=Confidence:')).toBeVisible({
      timeout: 10000,
    });

    // Should show percentage
    await expect(page.locator('text=/%/')).toBeVisible();
  });

  test('should display reasoning process', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send message that should generate reasoning
    await messageInput.fill(
      'システムの性能を分析して、最適化の提案をしてください。'
    );
    await sendButton.click();

    // Should show reasoning section
    await expect(page.locator('text=推論過程:')).toBeVisible({
      timeout: 10000,
    });
  });

  test('should maintain conversation context', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // First message about React
    await messageInput.fill('React Hooksについて教えてください。');
    await sendButton.click();

    // Wait for response
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });

    // Follow-up question that depends on context
    await messageInput.fill('それについてもう少し詳しく説明してもらえますか？');
    await sendButton.click();

    // Should understand context and provide relevant response
    await expect(page.locator('[data-role="assistant"]')).toHaveCount(2, {
      timeout: 15000,
    });
  });

  test('should handle tool invocation through natural language', async ({
    page,
  }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send complex request that should use multiple tools
    await messageInput.fill(
      'Please analyze the current system performance and search for optimization techniques online.'
    );
    await sendButton.click();

    // Should show multiple tools being used
    await expect(page.locator('text=Tools Used:')).toBeVisible({
      timeout: 10000,
    });

    // Should show reasoning for tool selection
    await expect(page.locator('text=Reasoning:')).toBeVisible();
  });

  test('should process streaming natural language responses', async ({
    page,
  }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send message that should stream
    await messageInput.fill(
      'Generate a detailed explanation about artificial intelligence.'
    );
    await sendButton.click();

    // Should show NLP processing indicator
    await expect(page.locator('text=Natural Language Processing')).toBeVisible({
      timeout: 5000,
    });

    // Should show streaming indicator
    await expect(
      page.locator('text=Analyzing natural language...')
    ).toBeVisible();

    // Should eventually complete
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 20000,
    });
  });

  test('should handle agent self-improvement requests', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send message about self-improvement
    await messageInput.fill(
      'あなたの回答の質を向上させるために、プロンプトを改善してください。'
    );
    await sendButton.click();

    // Should process the self-improvement request
    await expect(page.locator('text=自然言語処理')).toBeVisible({
      timeout: 5000,
    });

    // Wait for response
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });
  });

  test('should handle indefinite conversation continuation', async ({
    page,
  }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send multiple messages to test conversation continuation
    const messages = [
      "Let's start a conversation about technology.",
      'What do you think about AI development?',
      'How does machine learning work?',
      'Can you explain neural networks?',
      'What about deep learning?',
    ];

    for (const message of messages) {
      await messageInput.fill(message);
      await sendButton.click();

      // Wait for response before sending next message
      await page.waitForTimeout(2000);
    }

    // All messages should be preserved in conversation
    for (const message of messages) {
      await expect(page.locator(`text=${message}`)).toBeVisible();
    }

    // Should have multiple assistant responses
    await expect(page.locator('[data-role="assistant"]')).toHaveCount(
      messages.length,
      { timeout: 30000 }
    );
  });

  test('should handle error scenarios gracefully', async ({ page }) => {
    // Mock API error
    await page.route('**/v1/chat/completions/agent', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Agent processing failed' }),
      });
    });

    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send message that would normally use agent processing
    await messageInput.fill(
      'Test error handling in natural language processing.'
    );
    await sendButton.click();

    // Should show fallback response or error handling
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });
  });
});

test.describe('Natural Language Processing - Performance', () => {
  test('should process natural language within reasonable time', async ({
    page,
  }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    const startTime = Date.now();

    // Send message
    await messageInput.fill(
      'Quick performance test for natural language processing.'
    );
    await sendButton.click();

    // Wait for NLP indicator
    await expect(page.locator('text=Natural Language Processing')).toBeVisible({
      timeout: 5000,
    });

    // Wait for response
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });

    const endTime = Date.now();
    const processingTime = endTime - startTime;

    // Should complete within reasonable time (15 seconds including network)
    expect(processingTime).toBeLessThan(15000);
  });

  test('should handle rapid successive inputs', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send multiple messages quickly
    const rapidMessages = [
      'First rapid message',
      'Second rapid message',
      'Third rapid message',
    ];

    for (const message of rapidMessages) {
      await messageInput.fill(message);
      await sendButton.click();
      await page.waitForTimeout(100); // Small delay between messages
    }

    // All messages should be processed
    for (const message of rapidMessages) {
      await expect(page.locator(`text=${message}`)).toBeVisible();
    }
  });
});
