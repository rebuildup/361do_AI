import React from "react";
import { cn } from "@/utils";
import type { InputProps } from "@/types";

const Input: React.FC<InputProps> = ({
  className,
  type = "text",
  placeholder,
  value,
  onChange,
  disabled = false,
  error,
  ...props
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (onChange) {
      onChange(e.target.value);
    }
  };

  return (
    <div className="w-full">
      <input
        type={type}
        className={cn(
          "input-primary w-full",
          error && "border-red-500 focus:border-red-500",
          disabled && "opacity-50 cursor-not-allowed",
          className,
        )}
        placeholder={placeholder}
        value={value}
        onChange={handleChange}
        disabled={disabled}
        {...props}
      />
      {error && <p className="mt-1 text-sm text-red-500">{error}</p>}
    </div>
  );
};

export default Input;
