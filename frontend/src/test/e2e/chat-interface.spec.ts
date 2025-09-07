/**
 * Chat Interface E2E Tests
 *
 * End-to-end tests for the main chat interface functionality
 */

import { test, expect } from '@playwright/test';

test.describe('Chat Interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');

    // Wait for the app to load
    await page.waitForLoadState('networkidle');

    // Wait for chat interface to be ready
    await page
      .waitForSelector('[data-testid="chat-interface"]', {
        timeout: 10000,
        state: 'visible',
      })
      .catch(() => {
        // Fallback: wait for input field
        return page.waitForSelector(
          'textarea[placeholder*="メッセージを入力"]'
        );
      });
  });

  test('should load chat interface successfully', async ({ page }) => {
    // Check that main elements are present
    await expect(
      page.locator('textarea[placeholder*="メッセージを入力"]')
    ).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
  });

  test('should send a message and receive response', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Type a test message
    await messageInput.fill('Hello, this is a test message');

    // Send the message
    await sendButton.click();

    // Verify input is cleared
    await expect(messageInput).toHaveValue('');

    // Wait for user message to appear
    await expect(
      page.locator('text=Hello, this is a test message')
    ).toBeVisible();

    // Wait for assistant response (with timeout for potential network delays)
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });
  });

  test('should send message with Enter key', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );

    // Type message and press Enter
    await messageInput.fill('Test message via Enter key');
    await messageInput.press('Enter');

    // Verify message was sent
    await expect(messageInput).toHaveValue('');
    await expect(page.locator('text=Test message via Enter key')).toBeVisible();
  });

  test('should add new line with Shift+Enter', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );

    // Type first line
    await messageInput.fill('First line');

    // Add new line with Shift+Enter
    await messageInput.press('Shift+Enter');

    // Type second line
    await messageInput.type('Second line');

    // Verify multiline content
    await expect(messageInput).toHaveValue('First line\nSecond line');
  });

  test('should not send empty messages', async ({ page }) => {
    const sendButton = page.locator('button[type="submit"]');

    // Try to send empty message
    await sendButton.click();

    // Should not create any message bubbles
    await expect(page.locator('[data-role="user"]')).toHaveCount(0);
  });

  test('should handle Japanese input correctly', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Type Japanese message
    const japaneseMessage = 'こんにちは、これはテストメッセージです。';
    await messageInput.fill(japaneseMessage);
    await sendButton.click();

    // Verify Japanese message appears
    await expect(page.locator(`text=${japaneseMessage}`)).toBeVisible();

    // Wait for response
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });
  });

  test('should show streaming indicator during response generation', async ({
    page,
  }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send a message
    await messageInput.fill('Generate a streaming response');
    await sendButton.click();

    // Should show streaming indicator
    await expect(page.locator('text=応答を生成中')).toBeVisible({
      timeout: 5000,
    });

    // Should show stop button during streaming
    await expect(
      page.locator('button[title*="ストリーミングを停止"]')
    ).toBeVisible();
  });

  test('should be able to stop streaming', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send a message
    await messageInput.fill('Start streaming response');
    await sendButton.click();

    // Wait for streaming to start
    await expect(
      page.locator('button[title*="ストリーミングを停止"]')
    ).toBeVisible();

    // Stop streaming
    await page.locator('button[title*="ストリーミングを停止"]').click();

    // Streaming indicator should disappear
    await expect(page.locator('text=応答を生成中')).not.toBeVisible();
  });

  test('should display message timestamps', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send a message
    await messageInput.fill('Test timestamp display');
    await sendButton.click();

    // Should show timestamp (format may vary)
    await expect(page.locator('text=/\\d{1,2}:\\d{2}/')).toBeVisible();
  });

  test('should handle long messages gracefully', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Create a long message
    const longMessage = 'This is a very long message. '.repeat(50);

    await messageInput.fill(longMessage);
    await sendButton.click();

    // Message should be sent and displayed
    await expect(
      page.locator('text=This is a very long message')
    ).toBeVisible();
  });

  test('should auto-resize textarea based on content', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );

    // Get initial height
    const initialHeight = await messageInput.evaluate(el => el.clientHeight);

    // Add multiple lines
    await messageInput.fill('Line 1\nLine 2\nLine 3\nLine 4\nLine 5');

    // Height should increase
    const newHeight = await messageInput.evaluate(el => el.clientHeight);
    expect(newHeight).toBeGreaterThan(initialHeight);
  });

  test('should show keyboard shortcuts hint when no messages', async ({
    page,
  }) => {
    // Should show shortcuts hint
    await expect(page.locator('text=でショートカット一覧を表示')).toBeVisible();
  });

  test('should handle network errors gracefully', async ({ page }) => {
    // Mock network failure
    await page.route('**/v1/chat/completions', route => {
      route.abort('failed');
    });

    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send a message
    await messageInput.fill('Test network error handling');
    await sendButton.click();

    // Should show error message
    await expect(page.locator('text=接続エラー')).toBeVisible({
      timeout: 10000,
    });
  });

  test('should maintain conversation history', async ({ page }) => {
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    const sendButton = page.locator('button[type="submit"]');

    // Send first message
    await messageInput.fill('First message');
    await sendButton.click();
    await expect(page.locator('text=First message')).toBeVisible();

    // Wait for response
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 15000,
    });

    // Send second message
    await messageInput.fill('Second message');
    await sendButton.click();
    await expect(page.locator('text=Second message')).toBeVisible();

    // Both messages should still be visible
    await expect(page.locator('text=First message')).toBeVisible();
    await expect(page.locator('text=Second message')).toBeVisible();
  });
});

test.describe('Chat Interface - Mobile', () => {
  test.use({ viewport: { width: 375, height: 667 } }); // iPhone SE size

  test('should work on mobile devices', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Chat interface should be responsive
    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );
    await expect(messageInput).toBeVisible();

    // Should be able to send messages on mobile
    await messageInput.fill('Mobile test message');
    await page.locator('button[type="submit"]').click();

    await expect(page.locator('text=Mobile test message')).toBeVisible();
  });

  test('should handle virtual keyboard on mobile', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const messageInput = page.locator(
      'textarea[placeholder*="メッセージを入力"]'
    );

    // Focus input (simulates virtual keyboard opening)
    await messageInput.focus();

    // Should still be able to type and send
    await messageInput.fill('Virtual keyboard test');
    await page.locator('button[type="submit"]').click();

    await expect(page.locator('text=Virtual keyboard test')).toBeVisible();
  });
});
