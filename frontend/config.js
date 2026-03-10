// Backend configuration for static frontend deployment.
// RunPod backend endpoint - updated March 2026
window.BACKEND_URL = 'https://ge96nc5o7lkdtu-5000.proxy.runpod.net';

// Local development fallback.
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
  window.BACKEND_URL = 'http://localhost:5000';
}
