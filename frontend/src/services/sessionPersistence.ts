/**
 * Session Persistence Manager
 *
 * Handles indefinite conversation continuation without session clearing,
 * maintaining conversation state across browser sessions and page reloads.
 */

import { apiService } from './api';
import type { UIMessage } from '@/types';

export interface PersistentSession {
  sessionId: string;
  userId: string;
  sessionName: string;
  messages: UIMessage[];
  metadata: {
    createdAt: string;
    lastActivity: string;
    messageCount: number;
    language: 'ja' | 'en';
    agentState?: {
      learningEpoch: number;
      totalInteractions: number;
      rewardScore: number;
    };
  };
}

export interface SessionSummary {
  sessionId: string;
  sessionName: string;
  lastActivity: string;
  messageCount: number;
  preview: string;
}

/**
 * Session Persistence Manager class
 */
export class SessionPersistenceManager {
  private readonly STORAGE_KEY = 'kiro_persistent_sessions';
  private readonly MAX_SESSIONS = 50;
  private readonly MAX_MESSAGES_PER_SESSION = 1000;

  /**
   * Create or restore a persistent session
   */
  async createOrRestoreSession(
    userId: string = 'default_user'
  ): Promise<PersistentSession> {
    try {
      // Try to get existing session for user
      const existingSession = await this.getActiveSessionForUser(userId);

      if (existingSession) {
        // Restore existing session
        console.log('Restoring existing session:', existingSession.sessionId);
        return existingSession;
      }

      // Create new session
      const sessionResponse = await apiService.createSession({
        user_id: userId,
        session_name: this.generateSessionName(),
        metadata: {
          persistent: true,
          created_by: 'react_ui',
          language: 'ja',
        },
      });

      const newSession: PersistentSession = {
        sessionId: sessionResponse.session_id,
        userId,
        sessionName: sessionResponse.session_name || 'New Conversation',
        messages: [],
        metadata: {
          createdAt: new Date().toISOString(),
          lastActivity: new Date().toISOString(),
          messageCount: 0,
          language: 'ja',
        },
      };

      // Save to local storage
      await this.saveSession(newSession);

      console.log('Created new persistent session:', newSession.sessionId);
      return newSession;
    } catch (error) {
      console.error('Failed to create/restore session:', error);

      // Fallback: create local-only session
      const fallbackSession: PersistentSession = {
        sessionId: `local_${Date.now()}`,
        userId,
        sessionName: 'Local Session (Offline)',
        messages: [],
        metadata: {
          createdAt: new Date().toISOString(),
          lastActivity: new Date().toISOString(),
          messageCount: 0,
          language: 'ja',
        },
      };

      await this.saveSession(fallbackSession);
      return fallbackSession;
    }
  }

  /**
   * Get active session for a user (most recent)
   */
  async getActiveSessionForUser(
    userId: string
  ): Promise<PersistentSession | null> {
    const sessions = await this.getAllSessions();
    const userSessions = sessions.filter(s => s.userId === userId);

    if (userSessions.length === 0) {
      return null;
    }

    // Return most recently active session
    return userSessions.sort(
      (a, b) =>
        new Date(b.metadata.lastActivity).getTime() -
        new Date(a.metadata.lastActivity).getTime()
    )[0];
  }

  /**
   * Save session to persistent storage
   */
  async saveSession(session: PersistentSession): Promise<void> {
    try {
      const sessions = await this.getAllSessions();

      // Update existing session or add new one
      const existingIndex = sessions.findIndex(
        s => s.sessionId === session.sessionId
      );

      if (existingIndex >= 0) {
        sessions[existingIndex] = session;
      } else {
        sessions.push(session);
      }

      // Limit number of sessions
      if (sessions.length > this.MAX_SESSIONS) {
        // Remove oldest sessions
        sessions.sort(
          (a, b) =>
            new Date(b.metadata.lastActivity).getTime() -
            new Date(a.metadata.lastActivity).getTime()
        );
        sessions.splice(this.MAX_SESSIONS);
      }

      // Save to localStorage
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(sessions));

      // Also try to sync with backend if session exists there
      if (!session.sessionId.startsWith('local_')) {
        try {
          await this.syncWithBackend(session);
        } catch (error) {
          console.warn('Failed to sync session with backend:', error);
        }
      }
    } catch (error) {
      console.error('Failed to save session:', error);
    }
  }

  /**
   * Add message to session
   */
  async addMessageToSession(
    sessionId: string,
    message: UIMessage
  ): Promise<void> {
    const sessions = await this.getAllSessions();
    const sessionIndex = sessions.findIndex(s => s.sessionId === sessionId);

    if (sessionIndex >= 0) {
      const session = sessions[sessionIndex];

      // Add message
      session.messages.push(message);

      // Limit messages per session
      if (session.messages.length > this.MAX_MESSAGES_PER_SESSION) {
        session.messages = session.messages.slice(
          -this.MAX_MESSAGES_PER_SESSION
        );
      }

      // Update metadata
      session.metadata.lastActivity = new Date().toISOString();
      session.metadata.messageCount = session.messages.length;

      // Save updated session
      await this.saveSession(session);
    }
  }

  /**
   * Update session messages (bulk update)
   */
  async updateSessionMessages(
    sessionId: string,
    messages: UIMessage[]
  ): Promise<void> {
    const sessions = await this.getAllSessions();
    const sessionIndex = sessions.findIndex(s => s.sessionId === sessionId);

    if (sessionIndex >= 0) {
      const session = sessions[sessionIndex];

      // Update messages
      session.messages = messages.slice(-this.MAX_MESSAGES_PER_SESSION);

      // Update metadata
      session.metadata.lastActivity = new Date().toISOString();
      session.metadata.messageCount = session.messages.length;

      // Save updated session
      await this.saveSession(session);
    }
  }

  /**
   * Get session by ID
   */
  async getSession(sessionId: string): Promise<PersistentSession | null> {
    const sessions = await this.getAllSessions();
    return sessions.find(s => s.sessionId === sessionId) || null;
  }

  /**
   * Get all sessions
   */
  async getAllSessions(): Promise<PersistentSession[]> {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Failed to load sessions from storage:', error);
      return [];
    }
  }

  /**
   * Get session summaries for UI display
   */
  async getSessionSummaries(userId?: string): Promise<SessionSummary[]> {
    const sessions = await this.getAllSessions();
    const filteredSessions = userId
      ? sessions.filter(s => s.userId === userId)
      : sessions;

    return filteredSessions
      .sort(
        (a, b) =>
          new Date(b.metadata.lastActivity).getTime() -
          new Date(a.metadata.lastActivity).getTime()
      )
      .map(session => ({
        sessionId: session.sessionId,
        sessionName: session.sessionName,
        lastActivity: session.metadata.lastActivity,
        messageCount: session.metadata.messageCount,
        preview: this.generateSessionPreview(session),
      }));
  }

  /**
   * Delete session
   */
  async deleteSession(sessionId: string): Promise<void> {
    try {
      const sessions = await this.getAllSessions();
      const filteredSessions = sessions.filter(s => s.sessionId !== sessionId);

      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(filteredSessions));

      // Also try to delete from backend
      if (!sessionId.startsWith('local_')) {
        try {
          await apiService.clearConversationHistory(sessionId);
        } catch (error) {
          console.warn('Failed to delete session from backend:', error);
        }
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  }

  /**
   * Clear all sessions (with confirmation)
   */
  async clearAllSessions(userId?: string): Promise<void> {
    try {
      if (userId) {
        // Clear sessions for specific user
        const sessions = await this.getAllSessions();
        const filteredSessions = sessions.filter(s => s.userId !== userId);
        localStorage.setItem(
          this.STORAGE_KEY,
          JSON.stringify(filteredSessions)
        );
      } else {
        // Clear all sessions
        localStorage.removeItem(this.STORAGE_KEY);
      }
    } catch (error) {
      console.error('Failed to clear sessions:', error);
    }
  }

  /**
   * Export session data
   */
  async exportSession(
    sessionId: string,
    format: 'json' | 'markdown' | 'txt' = 'json'
  ): Promise<string> {
    const session = await this.getSession(sessionId);

    if (!session) {
      throw new Error('Session not found');
    }

    switch (format) {
      case 'json':
        return JSON.stringify(session, null, 2);

      case 'markdown':
        return this.formatSessionAsMarkdown(session);

      case 'txt':
        return this.formatSessionAsText(session);

      default:
        throw new Error('Unsupported export format');
    }
  }

  /**
   * Import session data
   */
  async importSession(
    data: string,
    format: 'json' = 'json'
  ): Promise<PersistentSession> {
    try {
      let session: PersistentSession;

      switch (format) {
        case 'json':
          session = JSON.parse(data);
          break;

        default:
          throw new Error('Unsupported import format');
      }

      // Validate session structure
      if (
        !session.sessionId ||
        !session.userId ||
        !Array.isArray(session.messages)
      ) {
        throw new Error('Invalid session data structure');
      }

      // Generate new session ID to avoid conflicts
      session.sessionId = `imported_${Date.now()}`;
      session.metadata.lastActivity = new Date().toISOString();

      await this.saveSession(session);
      return session;
    } catch (error) {
      console.error('Failed to import session:', error);
      throw error;
    }
  }

  /**
   * Sync session with backend
   */
  private async syncWithBackend(session: PersistentSession): Promise<void> {
    try {
      // Update session metadata on backend
      const backendSession = await apiService.getSession(session.sessionId);

      if (backendSession) {
        // Session exists, update if needed
        // Note: Backend session management would need to be enhanced
        // to support the full metadata we're tracking
      }
    } catch {
      // Session might not exist on backend, which is okay for local sessions
      console.debug('Session not found on backend:', session.sessionId);
    }
  }

  /**
   * Generate session name based on first message or timestamp
   */
  private generateSessionName(): string {
    const now = new Date();
    const timeStr = now.toLocaleString('ja-JP', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });

    return `会話 ${timeStr}`;
  }

  /**
   * Generate session preview from messages
   */
  private generateSessionPreview(session: PersistentSession): string {
    if (session.messages.length === 0) {
      return '新しい会話';
    }

    const firstUserMessage = session.messages.find(m => m.role === 'user');
    if (firstUserMessage) {
      return (
        firstUserMessage.content.slice(0, 50) +
        (firstUserMessage.content.length > 50 ? '...' : '')
      );
    }

    return `${session.messages.length}件のメッセージ`;
  }

  /**
   * Format session as markdown
   */
  private formatSessionAsMarkdown(session: PersistentSession): string {
    let markdown = `# ${session.sessionName}\n\n`;
    markdown += `**Session ID:** ${session.sessionId}\n`;
    markdown += `**Created:** ${new Date(session.metadata.createdAt).toLocaleString()}\n`;
    markdown += `**Last Activity:** ${new Date(session.metadata.lastActivity).toLocaleString()}\n`;
    markdown += `**Messages:** ${session.metadata.messageCount}\n\n`;
    markdown += `---\n\n`;

    for (const message of session.messages) {
      const timestamp = new Date(message.timestamp).toLocaleString();
      const role = message.role === 'user' ? 'ユーザー' : 'エージェント';

      markdown += `## ${role} (${timestamp})\n\n`;
      markdown += `${message.content}\n\n`;

      if (message.reasoning) {
        markdown += `**推論過程:**\n${message.reasoning}\n\n`;
      }

      markdown += `---\n\n`;
    }

    return markdown;
  }

  /**
   * Format session as plain text
   */
  private formatSessionAsText(session: PersistentSession): string {
    let text = `${session.sessionName}\n`;
    text += `${'='.repeat(session.sessionName.length)}\n\n`;
    text += `Session ID: ${session.sessionId}\n`;
    text += `Created: ${new Date(session.metadata.createdAt).toLocaleString()}\n`;
    text += `Last Activity: ${new Date(session.metadata.lastActivity).toLocaleString()}\n`;
    text += `Messages: ${session.metadata.messageCount}\n\n`;

    for (const message of session.messages) {
      const timestamp = new Date(message.timestamp).toLocaleString();
      const role = message.role === 'user' ? 'ユーザー' : 'エージェント';

      text += `[${timestamp}] ${role}:\n`;
      text += `${message.content}\n`;

      if (message.reasoning) {
        text += `推論: ${message.reasoning}\n`;
      }

      text += `\n${'-'.repeat(50)}\n\n`;
    }

    return text;
  }
}

// Export singleton instance
export const sessionPersistence = new SessionPersistenceManager();

// Export for testing and custom instances
export default SessionPersistenceManager;
