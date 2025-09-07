/**
 * Service Worker Registration and Management
 *
 * Handles service worker registration for production caching
 */

const isLocalhost = Boolean(
  window.location.hostname === 'localhost' ||
    window.location.hostname === '[::1]' ||
    window.location.hostname.match(
      /^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/
    )
);

interface ServiceWorkerConfig {
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  onUpdate?: (registration: ServiceWorkerRegistration) => void;
  onOfflineReady?: () => void;
}

export function registerServiceWorker(config?: ServiceWorkerConfig) {
  if ('serviceWorker' in navigator) {
    // Only register in production or when explicitly enabled
    const shouldRegister =
      import.meta.env.PROD || import.meta.env.VITE_ENABLE_SW === 'true';

    if (shouldRegister) {
      const publicUrl = new URL(import.meta.env.BASE_URL, window.location.href);
      if (publicUrl.origin !== window.location.origin) {
        return;
      }

      window.addEventListener('load', () => {
        const swUrl = `${import.meta.env.BASE_URL}sw.js`;

        if (isLocalhost) {
          checkValidServiceWorker(swUrl, config);
          navigator.serviceWorker.ready.then(() => {
            console.log(
              'This web app is being served cache-first by a service worker with offline support.'
            );
            config?.onOfflineReady?.();
          });
        } else {
          registerValidServiceWorker(swUrl, config);
        }
      });
    }
  }
}

function registerValidServiceWorker(
  swUrl: string,
  config?: ServiceWorkerConfig
) {
  navigator.serviceWorker
    .register(swUrl)
    .then(registration => {
      console.log('Service Worker registered successfully:', registration);

      registration.onupdatefound = () => {
        const installingWorker = registration.installing;
        if (installingWorker == null) {
          return;
        }

        installingWorker.onstatechange = () => {
          if (installingWorker.state === 'installed') {
            if (navigator.serviceWorker.controller) {
              console.log(
                'New content is available and will be used when all tabs for this page are closed.'
              );
              config?.onUpdate?.(registration);
            } else {
              console.log('Content is cached for offline use.');
              config?.onSuccess?.(registration);
              config?.onOfflineReady?.();
            }
          }
        };
      };
    })
    .catch(error => {
      console.error('Error during service worker registration:', error);

      // Report service worker registration errors
      const errorReport = {
        type: 'service_worker_registration_error',
        message: error.message,
        stack: error.stack,
        timestamp: new Date().toISOString(),
        url: swUrl,
      };

      const existingErrors = JSON.parse(
        localStorage.getItem('errorReports') || '[]'
      );
      existingErrors.push(errorReport);
      localStorage.setItem(
        'errorReports',
        JSON.stringify(existingErrors.slice(-10))
      );
    });
}

function checkValidServiceWorker(swUrl: string, config?: ServiceWorkerConfig) {
  fetch(swUrl, {
    headers: { 'Service-Worker': 'script' },
  })
    .then(response => {
      const contentType = response.headers.get('content-type');
      if (
        response.status === 404 ||
        (contentType != null && contentType.indexOf('javascript') === -1)
      ) {
        navigator.serviceWorker.ready.then(registration => {
          registration.unregister().then(() => {
            window.location.reload();
          });
        });
      } else {
        registerValidServiceWorker(swUrl, config);
      }
    })
    .catch(() => {
      console.log(
        'No internet connection found. App is running in offline mode.'
      );
      config?.onOfflineReady?.();
    });
}

export function unregisterServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.ready
      .then(registration => {
        registration.unregister();
        console.log('Service Worker unregistered');
      })
      .catch(error => {
        console.error('Error unregistering service worker:', error);
      });
  }
}

// Service Worker update management
export class ServiceWorkerUpdateManager {
  private registration: ServiceWorkerRegistration | null = null;
  private updateAvailable = false;

  constructor() {
    this.init();
  }

  private async init() {
    if ('serviceWorker' in navigator) {
      try {
        this.registration = await navigator.serviceWorker.ready;
        this.setupUpdateListener();
      } catch (error) {
        console.error('Service Worker not ready:', error);
      }
    }
  }

  private setupUpdateListener() {
    if (!this.registration) return;

    this.registration.addEventListener('updatefound', () => {
      const newWorker = this.registration!.installing;
      if (!newWorker) return;

      newWorker.addEventListener('statechange', () => {
        if (
          newWorker.state === 'installed' &&
          navigator.serviceWorker.controller
        ) {
          this.updateAvailable = true;
          this.notifyUpdateAvailable();
        }
      });
    });
  }

  private notifyUpdateAvailable() {
    // Dispatch custom event for update notification
    window.dispatchEvent(new CustomEvent('sw-update-available'));
  }

  public async applyUpdate(): Promise<void> {
    if (!this.registration || !this.updateAvailable) return;

    const newWorker = this.registration.waiting;
    if (!newWorker) return;

    // Send message to service worker to skip waiting
    newWorker.postMessage({ type: 'SKIP_WAITING' });

    // Listen for controlling change
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      window.location.reload();
    });
  }

  public isUpdateAvailable(): boolean {
    return this.updateAvailable;
  }
}

// Performance monitoring integration
export function setupServiceWorkerPerformanceMonitoring() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('message', event => {
      if (event.data && event.data.type === 'CACHE_PERFORMANCE') {
        // Log cache performance metrics
        console.log('Cache Performance:', event.data.metrics);

        // Send to performance monitoring service if available
        if (window.gtag) {
          window.gtag('event', 'cache_hit', {
            event_category: 'performance',
            event_label: event.data.metrics.cacheHitRate,
          });
        }
      }
    });
  }
}

// Offline detection and handling
export class OfflineManager {
  private isOnline = navigator.onLine;
  private callbacks: Array<(online: boolean) => void> = [];

  constructor() {
    this.setupEventListeners();
  }

  private setupEventListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.notifyCallbacks();
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      this.notifyCallbacks();
    });
  }

  private notifyCallbacks() {
    this.callbacks.forEach(callback => callback(this.isOnline));
  }

  public onStatusChange(callback: (online: boolean) => void) {
    this.callbacks.push(callback);

    // Return unsubscribe function
    return () => {
      const index = this.callbacks.indexOf(callback);
      if (index > -1) {
        this.callbacks.splice(index, 1);
      }
    };
  }

  public getStatus(): boolean {
    return this.isOnline;
  }
}

// Export singleton instances
export const swUpdateManager = new ServiceWorkerUpdateManager();
export const offlineManager = new OfflineManager();
