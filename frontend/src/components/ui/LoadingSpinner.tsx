import React from 'react';
import { Loader2, Circle, RotateCw } from 'lucide-react';
import { cn } from '@/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'spinner' | 'dots' | 'pulse' | 'rotate';
  className?: string;
  text?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'spinner',
  className,
  text,
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
  };

  const renderSpinner = () => {
    switch (variant) {
      case 'spinner':
        return (
          <Loader2
            className={cn(
              'animate-spin text-gray-400',
              sizeClasses[size],
              className
            )}
          />
        );

      case 'dots':
        return (
          <div className={cn('flex space-x-1', className)}>
            {[0, 1, 2].map(i => (
              <Circle
                key={i}
                className={cn(
                  'animate-pulse text-gray-400',
                  sizeClasses[size],
                  `animation-delay-${i * 200}ms`
                )}
                style={{
                  animationDelay: `${i * 0.2}s`,
                }}
              />
            ))}
          </div>
        );

      case 'pulse':
        return (
          <div
            className={cn(
              'animate-pulse-subtle bg-gray-600 rounded-full',
              sizeClasses[size],
              className
            )}
          />
        );

      case 'rotate':
        return (
          <RotateCw
            className={cn(
              'animate-spin text-gray-400',
              sizeClasses[size],
              className
            )}
          />
        );

      default:
        return null;
    }
  };

  if (text) {
    return (
      <div className="flex items-center space-x-2">
        {renderSpinner()}
        <span className={cn('text-gray-400', textSizeClasses[size])}>
          {text}
        </span>
      </div>
    );
  }

  return renderSpinner();
};

export default LoadingSpinner;
