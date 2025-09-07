/**
 * Service Worker for Production Caching
 *
 * Implements caching strategies for optimal performance
 */

const CACHE_NAME = 'react-custom-webui-v1';
const STATIC_CACHE = 'static-v1';
const DYNAMIC_CACHE = 'dynamic-v1';

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  // Add other critical assets here
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
  /\/v1\/health/,
  /\/v1\/models/,
  /\/v1\/agent\/status/,
];

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('Service Worker: Installing...');

  event.waitUntil(
    caches
      .open(STATIC_CACHE)
      .then(cache => {
        console.log('Service Worker: Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('Service Worker: Static assets cached');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('Service Worker: Failed to cache static assets', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('Service Worker: Activating...');

  event.waitUntil(
    caches
      .keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
              console.log('Service Worker: Deleting old cache', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('Service Worker: Activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Handle different types of requests
  if (url.origin === location.origin) {
    // Same origin requests
    if (request.url.includes('/api/') || request.url.includes('/v1/')) {
      // API requests - network first with cache fallback
      event.respondWith(networkFirstStrategy(request));
    } else if (request.destination === 'document') {
      // HTML documents - network first
      event.respondWith(networkFirstStrategy(request));
    } else {
      // Static assets - cache first
      event.respondWith(cacheFirstStrategy(request));
    }
  } else {
    // External requests - network only
    event.respondWith(fetch(request));
  }
});

// Cache first strategy - for static assets
async function cacheFirstStrategy(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    const networkResponse = await fetch(request);

    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.error('Cache first strategy failed:', error);

    // Return cached version if available
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    // Return offline page or error response
    return new Response('Offline', { status: 503 });
  }
}

// Network first strategy - for dynamic content
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);

    // Cache successful API responses (with expiration)
    if (networkResponse.ok && shouldCacheApiResponse(request)) {
      const cache = await caches.open(DYNAMIC_CACHE);
      const responseToCache = networkResponse.clone();

      // Add timestamp for cache expiration
      const headers = new Headers(responseToCache.headers);
      headers.set('sw-cached-at', Date.now().toString());

      const cachedResponse = new Response(responseToCache.body, {
        status: responseToCache.status,
        statusText: responseToCache.statusText,
        headers: headers,
      });

      cache.put(request, cachedResponse);
    }

    return networkResponse;
  } catch (error) {
    console.error('Network first strategy failed:', error);

    // Try to return cached version
    const cachedResponse = await caches.match(request);
    if (cachedResponse && !isCacheExpired(cachedResponse)) {
      console.log('Serving cached response for:', request.url);
      return cachedResponse;
    }

    // Return appropriate offline response based on request type
    if (request.url.includes('/v1/health')) {
      return new Response(
        JSON.stringify({
          status: 'offline',
          timestamp: new Date().toISOString(),
          version: 'unknown',
          system_info: {
            cpu_percent: 0,
            memory_percent: 0,
            note: 'Service Worker offline mode',
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    if (request.url.includes('/v1/models')) {
      return new Response(
        JSON.stringify({
          object: 'list',
          data: [
            {
              id: 'offline',
              object: 'model',
              created: Date.now(),
              owned_by: 'system',
            },
          ],
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    if (request.url.includes('/v1/chat/completions')) {
      return new Response(
        JSON.stringify({
          id: `offline_${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model: 'offline',
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content:
                  '申し訳ございませんが、現在オフラインです。ネットワーク接続を確認してから再度お試しください。',
              },
              finish_reason: 'stop',
            },
          ],
          usage: {
            prompt_tokens: 0,
            completion_tokens: 20,
            total_tokens: 20,
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Generic error response
    return new Response(
      JSON.stringify({
        error: 'ネットワークに接続できません',
        code: 'NETWORK_UNAVAILABLE',
        retryable: true,
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

// Check if API response should be cached
function shouldCacheApiResponse(request) {
  return API_CACHE_PATTERNS.some(pattern => pattern.test(request.url));
}

// Check if cached response is expired (1 hour for API responses)
function isCacheExpired(response) {
  const cachedAt = response.headers.get('sw-cached-at');
  if (!cachedAt) return false;

  const cacheAge = Date.now() - parseInt(cachedAt);
  const maxAge = 60 * 60 * 1000; // 1 hour

  return cacheAge > maxAge;
}

// Handle background sync for offline actions
self.addEventListener('sync', event => {
  if (event.tag === 'background-sync') {
    event.waitUntil(handleBackgroundSync());
  }
});

async function handleBackgroundSync() {
  console.log('Service Worker: Handling background sync');

  // Handle any queued offline actions
  // This could include sending messages, updating preferences, etc.

  try {
    // Example: sync offline messages
    const offlineActions = await getOfflineActions();

    for (const action of offlineActions) {
      try {
        await fetch(action.url, action.options);
        await removeOfflineAction(action.id);
      } catch (error) {
        console.error('Failed to sync offline action:', error);
      }
    }
  } catch (error) {
    console.error('Background sync failed:', error);
  }
}

// Helper functions for offline action queue
async function getOfflineActions() {
  // In a real implementation, this would read from IndexedDB
  return [];
}

async function removeOfflineAction(actionId) {
  // In a real implementation, this would remove from IndexedDB
  console.log('Removing offline action:', actionId);
}

// Handle push notifications (if needed in the future)
self.addEventListener('push', event => {
  if (event.data) {
    const data = event.data.json();

    const options = {
      body: data.body,
      icon: '/icon-192x192.png',
      badge: '/badge-72x72.png',
      tag: data.tag || 'default',
      requireInteraction: data.requireInteraction || false,
    };

    event.waitUntil(self.registration.showNotification(data.title, options));
  }
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  event.notification.close();

  event.waitUntil(clients.openWindow(event.notification.data?.url || '/'));
});

console.log('Service Worker: Loaded');
