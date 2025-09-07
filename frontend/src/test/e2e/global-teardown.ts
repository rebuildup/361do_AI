/**
 * Playwright Global Teardown
 *
 * Global teardown for end-to-end tests
 */

import { FullConfig } from '@playwright/test';

async function globalTeardown(_config: FullConfig) {
  console.log('🧹 Starting global E2E test teardown...');

  try {
    // Perform any global cleanup
    // For example, clearing test data, stopping services, etc.

    // Clear any test sessions or data
    // This would depend on your backend implementation

    console.log('✅ Global E2E teardown completed successfully');
  } catch (error) {
    console.error('❌ Global E2E teardown failed:', error);
    // Don't throw here as it might mask test failures
  }
}

export default globalTeardown;
