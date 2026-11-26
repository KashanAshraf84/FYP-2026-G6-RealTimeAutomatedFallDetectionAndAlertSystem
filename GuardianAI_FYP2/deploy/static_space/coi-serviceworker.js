/*
 * Cross-origin isolation on a static host.
 *
 * ONNX Runtime Web can only use several CPU threads when the page is
 * cross-origin isolated (SharedArrayBuffer), which needs two HTTP headers the
 * static Space cannot set. This service worker adds them to every response
 * of this origin; the page reloads once so it starts isolated.
 * Browsers without service workers or COEP "credentialless" simply run
 * single-threaded. Everything the page loads is same-origin, so nothing is
 * blocked by the policy.
 */
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
  self.addEventListener('fetch', e => {
    const req = e.request;
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
    e.respondWith(fetch(req).then(res => {
      if (res.status === 0 || res.type === 'opaque') return res;
      const headers = new Headers(res.headers);
      headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    }));
  });
} else if ('serviceWorker' in navigator && !window.crossOriginIsolated && location.protocol === 'https:') {
  const src = document.currentScript && document.currentScript.src;
  navigator.serviceWorker.register(src).then(reg => {
    const reloadOnce = () => {
      if (sessionStorage.getItem('coi-reloaded')) return;
      sessionStorage.setItem('coi-reloaded', '1');
      location.reload();
    };
    if (reg.active && !navigator.serviceWorker.controller) reloadOnce();
    else navigator.serviceWorker.addEventListener('controllerchange', reloadOnce, { once: true });
  }).catch(() => { /* run without isolation */ });
}
