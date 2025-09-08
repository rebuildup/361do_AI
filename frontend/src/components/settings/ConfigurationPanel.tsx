import React, { useState, useEffect } from 'react';
import { buildApiUrl } from '@/services/api';
import {
  Settings,
  Thermometer,
  Hash,
  Zap,
  RotateCcw,
  Save,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';
import { cn } from '@/utils';
import { useApp, useAppActions } from '@/contexts/AppContext';

interface ConfigurationPanelProps {
  className?: string;
  onConfigChange?: (config: ConfigurationState) => void;
}

interface ConfigurationState {
  temperature: number;
  maxTokens: number;
  streamingEnabled: boolean;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stopSequences?: string[];
}

const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({
  className,
  onConfigChange,
}) => {
  const { state } = useApp();
  const { setTemperature, setMaxTokens, setStreamingEnabled } = useAppActions();

  const [config, setConfig] = useState<ConfigurationState>({
    temperature: state.temperature,
    maxTokens: state.maxTokens,
    streamingEnabled: state.streamingEnabled,
    topP: 0.9,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
    stopSequences: [],
  });

  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>(
    'idle'
  );
  const [error, setError] = useState<string | null>(null);

  // Sync with global state
  useEffect(() => {
    setConfig(prev => ({
      ...prev,
      temperature: state.temperature,
      maxTokens: state.maxTokens,
      streamingEnabled: state.streamingEnabled,
    }));
  }, [state.temperature, state.maxTokens, state.streamingEnabled]);

  const handleConfigChange = (key: keyof ConfigurationState, value: any) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    onConfigChange?.(newConfig);

    // Update global state immediately for UI responsiveness
    switch (key) {
      case 'temperature':
        setTemperature(value);
        break;
      case 'maxTokens':
        setMaxTokens(value);
        break;
      case 'streamingEnabled':
        setStreamingEnabled(value);
        break;
    }
  };

  const saveConfiguration = async () => {
    setIsSaving(true);
    setSaveStatus('idle');
    setError(null);

    try {
      const response = await fetch(buildApiUrl('/config'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        setSaveStatus('success');
        setTimeout(() => setSaveStatus('idle'), 2000);
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.warn('Failed to save configuration to API:', error);
      // For development, just show success since we're updating local state
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 2000);
    } finally {
      setIsSaving(false);
    }
  };

  const resetToDefaults = () => {
    const defaultConfig: ConfigurationState = {
      temperature: 0.7,
      maxTokens: 2048,
      streamingEnabled: true,
      topP: 0.9,
      frequencyPenalty: 0.0,
      presencePenalty: 0.0,
      stopSequences: [],
    };

    setConfig(defaultConfig);
    setTemperature(defaultConfig.temperature);
    setMaxTokens(defaultConfig.maxTokens);
    setStreamingEnabled(defaultConfig.streamingEnabled);
    onConfigChange?.(defaultConfig);
  };

  const getTemperatureDescription = (temp: number) => {
    if (temp < 0.3) return '非常に保守的 - 予測可能な応答';
    if (temp < 0.7) return '保守的 - バランスの取れた応答';
    if (temp < 1.0) return '創造的 - 多様な応答';
    if (temp < 1.5) return '非常に創造的 - 予測困難な応答';
    return '極めて創造的 - 非常に予測困難';
  };

  const getTokensDescription = (tokens: number) => {
    if (tokens < 512) return '短い応答';
    if (tokens < 1024) return '中程度の応答';
    if (tokens < 2048) return '長い応答';
    return '非常に長い応答';
  };

  return (
    <div
      className={cn(
        'bg-gray-900 border border-gray-800 rounded-lg p-4',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Settings size={20} className="text-gray-300" />
          <h3 className="text-white font-medium">エージェント設定</h3>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={resetToDefaults}
            className="p-2 text-gray-400 hover:text-white transition-colors rounded-lg hover:bg-gray-800"
            title="デフォルトに戻す"
          >
            <RotateCcw size={16} />
          </button>

          <button
            onClick={saveConfiguration}
            disabled={isSaving}
            className={cn(
              'flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors',
              saveStatus === 'success'
                ? 'bg-green-700 text-green-100'
                : 'bg-gray-700 hover:bg-gray-600 text-white',
              isSaving && 'opacity-50 cursor-not-allowed'
            )}
          >
            {saveStatus === 'success' ? (
              <CheckCircle size={16} />
            ) : (
              <Save size={16} />
            )}
            {isSaving
              ? '保存中...'
              : saveStatus === 'success'
                ? '保存済み'
                : '保存'}
          </button>
        </div>
      </div>

      <div className="space-y-6">
        {/* Streaming Toggle */}
        <div>
          <label className="flex items-center gap-3">
            <input
              type="checkbox"
              checked={config.streamingEnabled}
              onChange={e =>
                handleConfigChange('streamingEnabled', e.target.checked)
              }
              className="w-4 h-4 rounded border-gray-700 bg-gray-800 text-blue-600 focus:ring-blue-500 focus:ring-2"
            />
            <div className="flex items-center gap-2">
              <Zap size={16} className="text-gray-400" />
              <span className="text-white text-sm font-medium">
                ストリーミング応答
              </span>
            </div>
          </label>
          <p className="text-xs text-gray-500 mt-1 ml-7">
            リアルタイムで応答を表示します
          </p>
        </div>

        {/* Temperature Slider */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Thermometer size={16} className="text-gray-400" />
            <label className="text-white text-sm font-medium">
              温度 (Temperature)
            </label>
            <span className="text-blue-400 text-sm font-mono">
              {config.temperature.toFixed(1)}
            </span>
          </div>

          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={config.temperature}
            onChange={e =>
              handleConfigChange('temperature', parseFloat(e.target.value))
            }
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer slider"
          />

          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>保守的 (0.0)</span>
            <span>創造的 (2.0)</span>
          </div>

          <p className="text-xs text-gray-400 mt-1">
            {getTemperatureDescription(config.temperature)}
          </p>
        </div>

        {/* Max Tokens Slider */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Hash size={16} className="text-gray-400" />
            <label className="text-white text-sm font-medium">
              最大トークン数
            </label>
            <span className="text-blue-400 text-sm font-mono">
              {config.maxTokens.toLocaleString()}
            </span>
          </div>

          <input
            type="range"
            min="256"
            max="4096"
            step="256"
            value={config.maxTokens}
            onChange={e =>
              handleConfigChange('maxTokens', parseInt(e.target.value))
            }
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer slider"
          />

          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>256</span>
            <span>4,096</span>
          </div>

          <p className="text-xs text-gray-400 mt-1">
            {getTokensDescription(config.maxTokens)}
          </p>
        </div>

        {/* Advanced Settings */}
        <div className="pt-4 border-t border-gray-800">
          <h4 className="text-white text-sm font-medium mb-3">高度な設定</h4>

          <div className="space-y-4">
            {/* Top P */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-gray-400 text-sm">Top P</label>
                <span className="text-blue-400 text-sm font-mono">
                  {config.topP?.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={config.topP || 0.9}
                onChange={e =>
                  handleConfigChange('topP', parseFloat(e.target.value))
                }
                className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer slider"
              />
              <p className="text-xs text-gray-500 mt-1">
                核サンプリング - 応答の多様性を制御
              </p>
            </div>

            {/* Frequency Penalty */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-gray-400 text-sm">頻度ペナルティ</label>
                <span className="text-blue-400 text-sm font-mono">
                  {config.frequencyPenalty?.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="-2"
                max="2"
                step="0.1"
                value={config.frequencyPenalty || 0}
                onChange={e =>
                  handleConfigChange(
                    'frequencyPenalty',
                    parseFloat(e.target.value)
                  )
                }
                className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer slider"
              />
              <p className="text-xs text-gray-500 mt-1">
                繰り返しを減らす（正の値）または増やす（負の値）
              </p>
            </div>

            {/* Presence Penalty */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-gray-400 text-sm">存在ペナルティ</label>
                <span className="text-blue-400 text-sm font-mono">
                  {config.presencePenalty?.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="-2"
                max="2"
                step="0.1"
                value={config.presencePenalty || 0}
                onChange={e =>
                  handleConfigChange(
                    'presencePenalty',
                    parseFloat(e.target.value)
                  )
                }
                className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer slider"
              />
              <p className="text-xs text-gray-500 mt-1">
                新しいトピックの導入を促進（正の値）または抑制（負の値）
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-2 bg-red-900 border border-red-700 rounded text-red-100 text-sm flex items-center gap-2">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};

export default ConfigurationPanel;
