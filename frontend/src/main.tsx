import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.tsx';
import {
  registerServiceWorker,
  setupServiceWorkerPerformanceMonitoring,
} from './utils/serviceWorker';
import { performanceMonitor } from './utils/performance';

// Register service worker for production caching
registerServiceWorker({
  onSuccess: () => {
    console.log('✅ App is ready for offline use');
  },
  onUpdate: () => {
    console.log('🔄 New content available, please refresh');
    // You could show a toast notification here
  },
  onOfflineReady: () => {
    console.log('📱 App is ready to work offline');
  },
});

// Setup performance monitoring
setupServiceWorkerPerformanceMonitoring();

// Initialize performance monitoring in production
if (import.meta.env.VITE_ENABLE_PERFORMANCE_MONITORING === 'true') {
  // Monitor Core Web Vitals
  performanceMonitor.getCoreWebVitals().then(vitals => {
    console.log('Core Web Vitals:', vitals);
  });
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
