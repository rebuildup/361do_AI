// Message role enum matching backend
export type MessageRole = 'system' | 'user' | 'assistant' | 'function';

// Chat related types (matching FastAPI backend)
export interface ChatMessage {
  role: MessageRole;
  content: string;
  name?: string;
  function_call?: Record<string, unknown>;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop?: string | string[];
  stream?: boolean;
  user?: string;
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatMessage;
  finish_reason?: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: Usage;
}

export interface StreamChoice {
  index: number;
  delta: Record<string, unknown>;
  finish_reason?: string;
}

export interface ChatCompletionStreamResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: StreamChoice[];
}

// Model related types
export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  permission?: Record<string, unknown>[];
}

export interface ModelsResponse {
  object: string;
  data: ModelInfo[];
}

// Session related types
export interface SessionRequest {
  user_id?: string;
  session_name?: string;
  metadata?: Record<string, unknown>;
}

export interface SessionResponse {
  session_id: string;
  created_at: string;
  user_id?: string;
  session_name?: string;
  metadata: Record<string, unknown>;
  status?: string;
}

// Inference types
export interface InferenceRequest {
  prompt: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  use_cot?: boolean;
  session_id?: string;
}

export interface InferenceResponse {
  id: string;
  response: string;
  reasoning_steps?: Record<string, unknown>[];
  confidence_score?: number;
  processing_time: number;
  memory_usage: Record<string, unknown>;
  model_info: Record<string, unknown>;
}

// Memory search types
export interface MemorySearchRequest {
  query: string;
  session_id?: string;
  max_results?: number;
  similarity_threshold?: number;
}

export interface MemorySearchResponse {
  results: Record<string, unknown>[];
  total_found: number;
  search_time: number;
  query: string;
}

// System related types
export interface SystemStatsRequest {
  include_gpu?: boolean;
  include_memory?: boolean;
  include_processes?: boolean;
}

export interface SystemStatsResponse {
  timestamp: string;
  cpu: Record<string, unknown>;
  memory: Record<string, unknown>;
  gpu?: Record<string, unknown>;
  processes?: Record<string, unknown>[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  system_info: Record<string, unknown>;
}

// Error types
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  retryable: boolean;
}

export interface UIError {
  component: string;
  error: Error;
  errorInfo: React.ErrorInfo;
}

// App state types
export interface User {
  id: string;
  name: string;
  email?: string;
}

export interface AppState {
  // Authentication
  user: User | null;
  isAuthenticated: boolean;

  // Chat
  currentSession: SessionResponse | null;
  messages: ChatMessage[];
  isStreaming: boolean;

  // UI
  sidebarCollapsed: boolean;
  theme: 'dark' | 'light';

  // Agent
  availableModels: ModelInfo[];
  activeModel: string;
  agentStatus: 'idle' | 'processing' | 'error';

  // Settings
  streamingEnabled: boolean;
  temperature: number;
  maxTokens: number;
}

// Extended UI types for better UX
export interface UIMessage extends ChatMessage {
  id: string;
  timestamp: Date;
  reasoning?: string;
  isStreaming?: boolean;
  error?: string;
}

// Component prop types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
}

export interface InputProps extends BaseComponentProps {
  type?: 'text' | 'email' | 'password' | 'number';
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  error?: string;
}
