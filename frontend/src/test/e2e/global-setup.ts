/**
 * Playwright Global Setup
 *
 * Global setup for end-to-end tests
 */

import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(_config: FullConfig) {
  console.log('🚀 Starting global E2E test setup...');

  // Launch browser for setup
  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    // Wait for the application to be ready
    console.log('⏳ Waiting for application to be ready...');
    await page.goto('http://localhost:3000');

    // Wait for the main app to load
    await page
      .waitForSelector('[data-testid="app-container"]', {
        timeout: 30000,
        state: 'visible',
      })
      .catch(() => {
        // Fallback: wait for any content to load
        return page.waitForLoadState('networkidle');
      });

    // Check if backend is available
    try {
      const response = await page.request.get(
        'http://localhost:8000/v1/health'
      );
      if (response.ok()) {
        console.log('✅ Backend health check passed');
      } else {
        console.log(
          '⚠️ Backend health check failed, tests will use mocked data'
        );
      }
    } catch {
      console.log('⚠️ Backend not available, tests will use mocked data');
    }

    // Perform any global authentication or setup
    // For now, we'll just verify the app loads

    console.log('✅ Global E2E setup completed successfully');
  } catch (error) {
    console.error('❌ Global E2E setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;
