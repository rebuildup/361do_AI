import React, { useState, useEffect } from 'react';
import { cn } from '@/utils';
import Header from './Header';
import SimpleSidebar from './SimpleSidebar';

interface SimpleLayoutProps {
  children: React.ReactNode;
  className?: string;
}

const SimpleLayout: React.FC<SimpleLayoutProps> = ({ children, className }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    // Auto-collapse based on screen size
    return window.innerWidth < 1056; // 768px + 288px
  });

  const [agentStatus] = useState<'idle' | 'processing' | 'error'>('idle');

  // Handle responsive behavior
  useEffect(() => {
    const handleResize = () => {
      const shouldCollapse = window.innerWidth < 1056;
      if (shouldCollapse && !sidebarCollapsed) {
        setSidebarCollapsed(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sidebarCollapsed]);

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <div className={cn('min-h-screen bg-black text-white', className)}>
      {/* Header */}
      <Header
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={toggleSidebar}
        agentStatus={agentStatus}
      />

      <div className="flex h-[calc(100vh-64px)]">
        {/* Sidebar */}
        <SimpleSidebar collapsed={sidebarCollapsed} onToggle={toggleSidebar} />

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">{children}</main>
      </div>
    </div>
  );
};

export default SimpleLayout;
