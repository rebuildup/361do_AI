import React, { useState, useEffect } from 'react';
import { buildApiUrl } from '@/services/api';
import { ChevronDown, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { cn } from '@/utils';
import { useApp, useAppActions } from '@/contexts/AppContext';
import type { ModelInfo } from '@/types';

interface ModelSelectorProps {
  className?: string;
  onModelChange?: (modelId: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  className,
  onModelChange,
}) => {
  const { state } = useApp();
  const { setActiveModel, setAvailableModels, setAgentStatus } =
    useAppActions();

  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelStatuses, setModelStatuses] = useState<
    Record<string, 'available' | 'unavailable' | 'loading'>
  >({});

  // Sample models for development
  const sampleModels: ModelInfo[] = [
    {
      id: 'deepseek-r1:7b',
      object: 'model',
      created: Date.now(),
      owned_by: 'deepseek',
    },
    {
      id: 'qwen2.5:7b-instruct-q4_k_m',
      object: 'model',
      created: Date.now(),
      owned_by: 'qwen',
    },
    {
      id: 'llama3.2:3b',
      object: 'model',
      created: Date.now(),
      owned_by: 'meta',
    },
    {
      id: 'gemma2:2b',
      object: 'model',
      created: Date.now(),
      owned_by: 'google',
    },
  ];

  // Load available models on component mount
  useEffect(() => {
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Fetch from API; no demo fallback when a real backend is expected
      const response = await fetch(buildApiUrl('/models'));
      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.data || data);
      } else {
        throw new Error(`HTTP ${response.status}`);
      }

      await checkModelAvailability();
    } catch (e) {
      const errorMessage =
        e instanceof Error ? e.message : 'Failed to load models';
      setError(errorMessage);
      console.error('Error loading models:', e);
      // In static demo, avoid spamming logs and don't auto-poll rapidly
    } finally {
      setIsLoading(false);
    }
  };

  const checkModelAvailability = async () => {
    const models =
      state.availableModels.length > 0 ? state.availableModels : sampleModels;
    const statuses: Record<string, 'available' | 'unavailable' | 'loading'> =
      {};

    // Set all models to loading initially
    models.forEach(model => {
      statuses[model.id] = 'loading';
    });
    setModelStatuses(statuses);

    // Check each model's availability
    for (const model of models) {
      try {
        const response = await fetch(buildApiUrl(`/models/${model.id}/status`));
        if (response.ok) {
          const data = await response.json();
          statuses[model.id] = data.available ? 'available' : 'unavailable';
        } else {
          // Simulate availability for development
          statuses[model.id] =
            Math.random() > 0.2 ? 'available' : 'unavailable';
        }
      } catch {
        // Simulate availability for development
        statuses[model.id] = Math.random() > 0.2 ? 'available' : 'unavailable';
      }
    }

    setModelStatuses(statuses);
  };

  const handleModelSelect = async (modelId: string) => {
    if (modelStatuses[modelId] !== 'available') {
      return;
    }

    setIsLoading(true);
    setAgentStatus('processing');

    try {
      // Try to set active model via API
      try {
        const response = await fetch(buildApiUrl('/models/active'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ model_id: modelId }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (e) {
        console.warn('Failed to set active model via API:', e);
      }

      setActiveModel(modelId);
      onModelChange?.(modelId);
      setIsOpen(false);
      setError(null);

      // Simulate model loading time
      setTimeout(() => {
        setAgentStatus('idle');
      }, 2000);
    } catch (e) {
      const errorMessage =
        e instanceof Error ? e.message : 'Failed to select model';
      setError(errorMessage);
      setAgentStatus('error');
    } finally {
      setIsLoading(false);
    }
  };

  const getModelDisplayName = (model: ModelInfo) => {
    const parts = model.id.split(':');
    const baseName = parts[0];
    const version = parts[1] || '';

    // Format display name
    const displayName = baseName
      .split(/[-_]/)
      .map(part => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ');

    return version ? `${displayName} (${version})` : displayName;
  };

  const getStatusIcon = (modelId: string) => {
    const status = modelStatuses[modelId];
    switch (status) {
      case 'available':
        return <CheckCircle size={16} className="text-green-400" />;
      case 'unavailable':
        return <AlertCircle size={16} className="text-red-400" />;
      case 'loading':
        return <Loader2 size={16} className="text-yellow-400 animate-spin" />;
      default:
        return null;
    }
  };

  const models =
    state.availableModels.length > 0 ? state.availableModels : sampleModels;
  const activeModel = models.find(m => m.id === state.activeModel);

  return (
    <div className={cn('relative', className)}>
      <label className="block text-sm font-medium text-gray-400 mb-2">
        モデル選択
      </label>

      {/* Dropdown Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isLoading}
        className={cn(
          'w-full bg-gray-900 border border-gray-700 text-white rounded-lg px-3 py-2 text-left flex items-center justify-between transition-colors',
          'hover:border-gray-600 focus:outline-none focus:border-gray-600',
          isLoading && 'opacity-50 cursor-not-allowed'
        )}
      >
        <div className="flex items-center gap-2">
          {isLoading ? (
            <Loader2 size={16} className="animate-spin text-gray-400" />
          ) : (
            getStatusIcon(state.activeModel)
          )}
          <span className="truncate">
            {activeModel ? getModelDisplayName(activeModel) : 'モデルを選択'}
          </span>
        </div>
        <ChevronDown
          size={16}
          className={cn(
            'text-gray-400 transition-transform',
            isOpen && 'transform rotate-180'
          )}
        />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-gray-900 border border-gray-700 rounded-lg z-50 max-h-60 overflow-y-auto animate-scale-in backdrop-blur-sm">
          {models.map(model => {
            const isSelected = model.id === state.activeModel;
            const status = modelStatuses[model.id];
            const isAvailable = status === 'available';

            return (
              <button
                key={model.id}
                onClick={() => handleModelSelect(model.id)}
                disabled={!isAvailable || isLoading}
                className={cn(
                  'w-full px-3 py-2 text-left flex items-center gap-2 transition-all duration-200',
                  'hover:bg-gray-800 hover:scale-[1.02] focus:outline-none focus:bg-gray-800',
                  'active:scale-[0.98]',
                  isSelected && 'bg-gray-800',
                  !isAvailable && 'opacity-50 cursor-not-allowed'
                )}
              >
                {getStatusIcon(model.id)}
                <div className="flex-1 min-w-0">
                  <div className="text-white truncate">
                    {getModelDisplayName(model)}
                  </div>
                  <div className="text-xs text-gray-400 truncate">
                    {model.id}
                  </div>
                </div>
                {isSelected && (
                  <CheckCircle size={16} className="text-blue-400" />
                )}
              </button>
            );
          })}

          {/* Refresh Button */}
          <div className="border-t border-gray-700 p-2">
            <button
              onClick={loadAvailableModels}
              disabled={isLoading}
              className="w-full px-2 py-1 text-sm text-gray-400 hover:text-white transition-colors flex items-center gap-2"
            >
              <Loader2 size={14} className={cn(isLoading && 'animate-spin')} />
              モデルリストを更新
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-2 p-2 bg-red-900 border border-red-700 rounded text-red-100 text-sm flex items-center gap-2">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Click outside to close */}
      {isOpen && (
        <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
      )}
    </div>
  );
};

export default ModelSelector;
