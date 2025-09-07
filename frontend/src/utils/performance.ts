/**
 * Performance Monitoring Utilities
 *
 * Tools for monitoring and optimizing application performance
 */

// Performance metrics interface
export interface PerformanceMetrics {
  renderTime: number;
  bundleSize: number;
  memoryUsage: number;
  networkRequests: number;
  cacheHitRate: number;
}

// Performance observer for monitoring
class PerformanceMonitor {
  private metrics: Map<string, number> = new Map();
  private observers: PerformanceObserver[] = [];

  constructor() {
    this.initializeObservers();
  }

  private initializeObservers() {
    if (typeof window === 'undefined') return;

    // Monitor navigation timing
    if ('PerformanceObserver' in window) {
      try {
        const navigationObserver = new PerformanceObserver(list => {
          const entries = list.getEntries();
          entries.forEach(entry => {
            if (entry.entryType === 'navigation') {
              const navEntry = entry as PerformanceNavigationTiming;
              this.metrics.set(
                'domContentLoaded',
                navEntry.domContentLoadedEventEnd -
                  navEntry.domContentLoadedEventStart
              );
              this.metrics.set(
                'loadComplete',
                navEntry.loadEventEnd - navEntry.loadEventStart
              );
              this.metrics.set(
                'firstPaint',
                navEntry.responseEnd - navEntry.requestStart
              );
            }
          });
        });

        navigationObserver.observe({ entryTypes: ['navigation'] });
        this.observers.push(navigationObserver);
      } catch (error) {
        console.warn('Navigation observer not supported:', error);
      }

      // Monitor resource loading
      try {
        const resourceObserver = new PerformanceObserver(list => {
          const entries = list.getEntries();
          entries.forEach(entry => {
            if (entry.entryType === 'resource') {
              const resourceEntry = entry as PerformanceResourceTiming;
              this.metrics.set(
                `resource_${entry.name}`,
                resourceEntry.duration
              );
            }
          });
        });

        resourceObserver.observe({ entryTypes: ['resource'] });
        this.observers.push(resourceObserver);
      } catch (error) {
        console.warn('Resource observer not supported:', error);
      }

      // Monitor long tasks
      try {
        const longTaskObserver = new PerformanceObserver(list => {
          const entries = list.getEntries();
          entries.forEach(entry => {
            if (entry.duration > 50) {
              // Tasks longer than 50ms
              console.warn(`Long task detected: ${entry.duration}ms`);
              this.metrics.set(
                'longTasks',
                (this.metrics.get('longTasks') || 0) + 1
              );
            }
          });
        });

        longTaskObserver.observe({ entryTypes: ['longtask'] });
        this.observers.push(longTaskObserver);
      } catch (error) {
        console.warn('Long task observer not supported:', error);
      }
    }
  }

  // Measure component render time
  measureRender<T>(componentName: string, renderFn: () => T): T {
    const startTime = performance.now();
    const result = renderFn();
    const endTime = performance.now();

    this.metrics.set(`render_${componentName}`, endTime - startTime);

    if (endTime - startTime > 16) {
      // Longer than one frame (60fps)
      console.warn(
        `Slow render detected for ${componentName}: ${(endTime - startTime).toFixed(2)}ms`
      );
    }

    return result;
  }

  // Measure async operations
  async measureAsync<T>(
    operationName: string,
    asyncFn: () => Promise<T>
  ): Promise<T> {
    const startTime = performance.now();
    try {
      const result = await asyncFn();
      const endTime = performance.now();
      this.metrics.set(`async_${operationName}`, endTime - startTime);
      return result;
    } catch (error) {
      const endTime = performance.now();
      this.metrics.set(`async_${operationName}_error`, endTime - startTime);
      throw error;
    }
  }

  // Get memory usage
  getMemoryUsage(): MemoryInfo | null {
    if ('memory' in performance) {
      return (performance as any).memory;
    }
    return null;
  }

  // Get all metrics
  getMetrics(): Record<string, number> {
    const result: Record<string, number> = {};
    this.metrics.forEach((value, key) => {
      result[key] = value;
    });
    return result;
  }

  // Get Core Web Vitals
  getCoreWebVitals(): Promise<Record<string, number>> {
    return new Promise(resolve => {
      const vitals: Record<string, number> = {};

      // Largest Contentful Paint (LCP)
      if ('PerformanceObserver' in window) {
        try {
          const lcpObserver = new PerformanceObserver(list => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            vitals.lcp = lastEntry.startTime;
          });
          lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

          // First Input Delay (FID)
          const fidObserver = new PerformanceObserver(list => {
            const entries = list.getEntries();
            entries.forEach(entry => {
              vitals.fid = (entry as any).processingStart - entry.startTime;
            });
          });
          fidObserver.observe({ entryTypes: ['first-input'] });

          // Cumulative Layout Shift (CLS)
          let clsValue = 0;
          const clsObserver = new PerformanceObserver(list => {
            const entries = list.getEntries();
            entries.forEach(entry => {
              if (!(entry as any).hadRecentInput) {
                clsValue += (entry as any).value;
              }
            });
            vitals.cls = clsValue;
          });
          clsObserver.observe({ entryTypes: ['layout-shift'] });

          // Resolve after a short delay to collect initial metrics
          setTimeout(() => {
            resolve(vitals);
          }, 2000);
        } catch (error) {
          console.warn('Core Web Vitals measurement failed:', error);
          resolve(vitals);
        }
      } else {
        resolve(vitals);
      }
    });
  }

  // Clean up observers
  disconnect() {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
  }
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor();

// React hook for performance monitoring
export function usePerformanceMonitor() {
  return {
    measureRender: performanceMonitor.measureRender.bind(performanceMonitor),
    measureAsync: performanceMonitor.measureAsync.bind(performanceMonitor),
    getMetrics: performanceMonitor.getMetrics.bind(performanceMonitor),
    getMemoryUsage: performanceMonitor.getMemoryUsage.bind(performanceMonitor),
    getCoreWebVitals:
      performanceMonitor.getCoreWebVitals.bind(performanceMonitor),
  };
}

// Bundle size analyzer
export function analyzeBundleSize(): Promise<{
  totalSize: number;
  chunks: Array<{ name: string; size: number }>;
}> {
  return new Promise(resolve => {
    if (typeof window === 'undefined') {
      resolve({ totalSize: 0, chunks: [] });
      return;
    }

    // Analyze loaded resources
    const resources = performance.getEntriesByType(
      'resource'
    ) as PerformanceResourceTiming[];
    const jsResources = resources.filter(
      resource =>
        resource.name.includes('.js') || resource.name.includes('.mjs')
    );

    const chunks = jsResources.map(resource => ({
      name: resource.name.split('/').pop() || 'unknown',
      size: resource.transferSize || 0,
    }));

    const totalSize = chunks.reduce((sum, chunk) => sum + chunk.size, 0);

    resolve({ totalSize, chunks });
  });
}

// Performance optimization helpers
export const performanceHelpers = {
  // Debounce function for performance
  debounce<T extends (...args: any[]) => any>(
    func: T,
    wait: number
  ): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout;
    return (...args: Parameters<T>) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    };
  },

  // Throttle function for performance
  throttle<T extends (...args: any[]) => any>(
    func: T,
    limit: number
  ): (...args: Parameters<T>) => void {
    let inThrottle: boolean;
    return (...args: Parameters<T>) => {
      if (!inThrottle) {
        func(...args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    };
  },

  // Lazy load images
  lazyLoadImage(src: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  },

  // Preload critical resources
  preloadResource(href: string, as: string): void {
    if (typeof document === 'undefined') return;

    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    document.head.appendChild(link);
  },

  // Check if user prefers reduced motion
  prefersReducedMotion(): boolean {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  },

  // Get connection information
  getConnectionInfo(): {
    effectiveType?: string;
    downlink?: number;
    rtt?: number;
  } {
    if (typeof navigator === 'undefined') return {};

    const connection =
      (navigator as any).connection ||
      (navigator as any).mozConnection ||
      (navigator as any).webkitConnection;

    if (connection) {
      return {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
      };
    }

    return {};
  },
};

// Performance report generator
export function generatePerformanceReport(): {
  metrics: Record<string, number>;
  bundleSize: Promise<{
    totalSize: number;
    chunks: Array<{ name: string; size: number }>;
  }>;
  coreWebVitals: Promise<Record<string, number>>;
  memoryUsage: MemoryInfo | null;
  connectionInfo: { effectiveType?: string; downlink?: number; rtt?: number };
} {
  return {
    metrics: performanceMonitor.getMetrics(),
    bundleSize: analyzeBundleSize(),
    coreWebVitals: performanceMonitor.getCoreWebVitals(),
    memoryUsage: performanceMonitor.getMemoryUsage(),
    connectionInfo: performanceHelpers.getConnectionInfo(),
  };
}
