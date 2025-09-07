import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/utils';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  children: React.ReactNode;
  className?: string;
}

const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  children,
  className,
  disabled,
  ...props
}) => {
  const baseClasses = [
    'inline-flex items-center justify-center font-medium rounded-lg',
    'transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2',
    'disabled:opacity-50 disabled:cursor-not-allowed',
    'hover:scale-105 active:scale-95',
  ];

  const variantClasses = {
    primary: [
      'bg-gray-700 text-white hover:bg-gray-600',
      'focus:ring-gray-600 focus:ring-offset-gray-900',
    ],
    secondary: [
      'bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white',
      'focus:ring-gray-700 focus:ring-offset-gray-900',
    ],
    outline: [
      'border border-gray-600 text-gray-300 hover:bg-gray-800 hover:text-white',
      'focus:ring-gray-600 focus:ring-offset-gray-900',
    ],
    ghost: [
      'text-gray-400 hover:bg-gray-800 hover:text-white',
      'focus:ring-gray-700 focus:ring-offset-gray-900',
    ],
    danger: [
      'bg-red-700 text-white hover:bg-red-600',
      'focus:ring-red-600 focus:ring-offset-gray-900',
    ],
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
  };

  return (
    <button
      className={cn(
        ...baseClasses,
        ...variantClasses[variant],
        sizeClasses[size],
        loading && 'cursor-wait',
        className
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 size={16} className="animate-spin mr-2" />}
      {children}
    </button>
  );
};

export default Button;
