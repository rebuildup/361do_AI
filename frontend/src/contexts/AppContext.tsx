import React, { createContext, useContext, useReducer } from 'react';
import type { ReactNode } from 'react';
import type {
  AppState,
  User,
  SessionResponse,
  ChatMessage,
  ModelInfo,
} from '@/types';

// Initial state
const initialState: AppState = {
  // Authentication
  user: null,
  isAuthenticated: false,

  // Chat
  currentSession: null,
  messages: [],
  isStreaming: false,

  // UI
  sidebarCollapsed: false,
  theme: 'dark',

  // Agent
  availableModels: [],
  activeModel: 'deepseek-r1:7b',
  agentStatus: 'idle',

  // Settings
  streamingEnabled: true,
  temperature: 0.7,
  maxTokens: 2048,
};

// Action types
type AppAction =
  | { type: 'SET_USER'; payload: User | null }
  | { type: 'SET_AUTHENTICATED'; payload: boolean }
  | { type: 'SET_CURRENT_SESSION'; payload: SessionResponse | null }
  | { type: 'SET_MESSAGES'; payload: ChatMessage[] }
  | { type: 'ADD_MESSAGE'; payload: ChatMessage }
  | { type: 'UPDATE_MESSAGE'; payload: { index: number; message: ChatMessage } }
  | { type: 'SET_STREAMING'; payload: boolean }
  | { type: 'SET_SIDEBAR_COLLAPSED'; payload: boolean }
  | { type: 'SET_THEME'; payload: 'dark' | 'light' }
  | { type: 'SET_AVAILABLE_MODELS'; payload: ModelInfo[] }
  | { type: 'SET_ACTIVE_MODEL'; payload: string }
  | { type: 'SET_AGENT_STATUS'; payload: 'idle' | 'processing' | 'error' }
  | { type: 'SET_STREAMING_ENABLED'; payload: boolean }
  | { type: 'SET_TEMPERATURE'; payload: number }
  | { type: 'SET_MAX_TOKENS'; payload: number }
  | { type: 'RESET_STATE' };

// Reducer
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_USER':
      return { ...state, user: action.payload };
    case 'SET_AUTHENTICATED':
      return { ...state, isAuthenticated: action.payload };
    case 'SET_CURRENT_SESSION':
      return { ...state, currentSession: action.payload };
    case 'SET_MESSAGES':
      return { ...state, messages: action.payload };
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload] };
    case 'UPDATE_MESSAGE':
      const updatedMessages = [...state.messages];
      updatedMessages[action.payload.index] = action.payload.message;
      return { ...state, messages: updatedMessages };
    case 'SET_STREAMING':
      return { ...state, isStreaming: action.payload };
    case 'SET_SIDEBAR_COLLAPSED':
      return { ...state, sidebarCollapsed: action.payload };
    case 'SET_THEME':
      return { ...state, theme: action.payload };
    case 'SET_AVAILABLE_MODELS':
      return { ...state, availableModels: action.payload };
    case 'SET_ACTIVE_MODEL':
      return { ...state, activeModel: action.payload };
    case 'SET_AGENT_STATUS':
      return { ...state, agentStatus: action.payload };
    case 'SET_STREAMING_ENABLED':
      return { ...state, streamingEnabled: action.payload };
    case 'SET_TEMPERATURE':
      return { ...state, temperature: action.payload };
    case 'SET_MAX_TOKENS':
      return { ...state, maxTokens: action.payload };
    case 'RESET_STATE':
      return initialState;
    default:
      return state;
  }
}

// Context
interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider component
interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

// Custom hook to use the context
export const useApp = (): AppContextType => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

// Action creators for common operations
export const useAppActions = () => {
  const { dispatch } = useApp();

  return {
    setUser: (user: User | null) =>
      dispatch({ type: 'SET_USER', payload: user }),
    setAuthenticated: (authenticated: boolean) =>
      dispatch({ type: 'SET_AUTHENTICATED', payload: authenticated }),
    setCurrentSession: (session: SessionResponse | null) =>
      dispatch({ type: 'SET_CURRENT_SESSION', payload: session }),
    setMessages: (messages: ChatMessage[]) =>
      dispatch({ type: 'SET_MESSAGES', payload: messages }),
    addMessage: (message: ChatMessage) =>
      dispatch({ type: 'ADD_MESSAGE', payload: message }),
    updateMessage: (index: number, message: ChatMessage) =>
      dispatch({ type: 'UPDATE_MESSAGE', payload: { index, message } }),
    setStreaming: (streaming: boolean) =>
      dispatch({ type: 'SET_STREAMING', payload: streaming }),
    setSidebarCollapsed: (collapsed: boolean) =>
      dispatch({ type: 'SET_SIDEBAR_COLLAPSED', payload: collapsed }),
    setTheme: (theme: 'dark' | 'light') =>
      dispatch({ type: 'SET_THEME', payload: theme }),
    setAvailableModels: (models: ModelInfo[]) =>
      dispatch({ type: 'SET_AVAILABLE_MODELS', payload: models }),
    setActiveModel: (model: string) =>
      dispatch({ type: 'SET_ACTIVE_MODEL', payload: model }),
    setAgentStatus: (status: 'idle' | 'processing' | 'error') =>
      dispatch({ type: 'SET_AGENT_STATUS', payload: status }),
    setStreamingEnabled: (enabled: boolean) =>
      dispatch({ type: 'SET_STREAMING_ENABLED', payload: enabled }),
    setTemperature: (temperature: number) =>
      dispatch({ type: 'SET_TEMPERATURE', payload: temperature }),
    setMaxTokens: (maxTokens: number) =>
      dispatch({ type: 'SET_MAX_TOKENS', payload: maxTokens }),
    resetState: () => dispatch({ type: 'RESET_STATE' }),
  };
};
