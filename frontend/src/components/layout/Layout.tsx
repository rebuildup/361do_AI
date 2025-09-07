import React, { useState, useEffect } from 'react';
import { cn, shouldAutoCollapseSidebar, storage } from '@/utils';
import Header from './Header';
import Sidebar from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
  className?: string;
}

const Layout: React.FC<LayoutProps> = ({ children, className }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    // Initialize from localStorage or auto-collapse based on screen size
    const saved = storage.get('sidebarCollapsed', false);
    return saved || shouldAutoCollapseSidebar();
  });

  const [agentStatus, setAgentStatus] = useState<
    'idle' | 'processing' | 'error'
  >('idle');

  // Handle responsive behavior
  useEffect(() => {
    const handleResize = () => {
      const shouldCollapse = shouldAutoCollapseSidebar();
      if (shouldCollapse && !sidebarCollapsed) {
        setSidebarCollapsed(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sidebarCollapsed]);

  // Save sidebar state to localStorage
  useEffect(() => {
    storage.set('sidebarCollapsed', sidebarCollapsed);
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
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={toggleSidebar}
          agentStatus={agentStatus}
          onStatusChange={setAgentStatus}
        />

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">{children}</main>
      </div>
    </div>
  );
};

export default Layout;
