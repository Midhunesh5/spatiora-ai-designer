#!/usr/bin/env node

/**
 * inject-env.js - Vercel Build Script
 * 
 * This script injects the REACT_APP_BACKEND_URL environment variable
 * into the frontend config.js file at build time.
 * 
 * This allows Vercel to properly pass the RunPod backend URL to the frontend
 * without requiring a Node.js build step (since we're using plain HTML/JS).
 */

const fs = require('fs');
const path = require('path');

const backendUrl = process.env.REACT_APP_BACKEND_URL || 'https://your-runpod-url.pods.runpod.io';

console.log('🔧 [Vercel Build] Injecting backend URL into config.js');
console.log(`   Backend URL: ${backendUrl}`);

const configPath = path.join(__dirname, 'frontend', 'config.js');
const configContent = `// Backend configuration for Vercel + RunPod deployment
// Generated at build time by inject-env.js

window.BACKEND_URL = '${backendUrl}';

// Fallback for local development
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
  window.BACKEND_URL = 'http://localhost:5000';
}

// Log for debugging (remove in production)
if (window.location.hostname === 'localhost') {
  console.log('[Config] Backend URL:', window.BACKEND_URL);
}
`;

try {
  fs.writeFileSync(configPath, configContent, 'utf-8');
  console.log(`✅ [Vercel Build] Successfully injected backend URL`);
} catch (error) {
  console.error(`❌ [Vercel Build] Error writing config.js:`, error);
  process.exit(1);
}
