// Backend configuration for static frontend deployment.
// Update this URL to your RunPod backend endpoint.
window.BACKEND_URL = 'https://your-runpod-url.pods.runpod.io';

// Local development fallback.
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
  window.BACKEND_URL = 'http://localhost:5000';
}
