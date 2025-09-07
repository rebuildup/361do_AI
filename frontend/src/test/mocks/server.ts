/**
 * MSW Server Setup
 *
 * Mock Service Worker server for testing
 */

import { setupServer } from 'msw/node';
import { handlers } from './handlers';

// Setup MSW server with default handlers
export const server = setupServer(...handlers);

// Start server before all tests
export const startMockServer = () => {
  server.listen({
    onUnhandledRequest: 'warn',
  });
};

// Reset handlers after each test
export const resetMockServer = () => {
  server.resetHandlers();
};

// Close server after all tests
export const closeMockServer = () => {
  server.close();
};

// Helper to use error handlers
export const useErrorHandlers = async () => {
  const { errorHandlers } = await import('./handlers');
  server.use(...errorHandlers);
};

// Helper to add custom handlers
export const addMockHandlers = (...customHandlers: any[]) => {
  server.use(...customHandlers);
};
