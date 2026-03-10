// ===== CONSTANTS =====
// Auto-detect backend URL: use environment variable or default to production RunPod URL
const BACKEND_BASE = window.BACKEND_URL || 'https://ge96nc5o7lkdtu-5000.proxy.runpod.net';
const API_URL = `${BACKEND_BASE}/generate-stream`;

// ===== STATE =====
let currentPage = 'home';
let currentTool = null;
let selectedImageUrl = null;

// ===== DOM REFERENCES =====
const $ = (id) => document.getElementById(id);
const header = $('header');
const hamburger = $('hamburger');
const nav = $('nav');

// ===== HEADER SCROLL EFFECT =====
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

// ===== HAMBURGER MENU =====
hamburger.addEventListener('click', () => {
    nav.classList.toggle('mobile-open');
    hamburger.classList.toggle('active');
});

document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', () => {
        nav.classList.remove('mobile-open');
        hamburger.classList.remove('active');
    });
});

// ===== PAGE NAVIGATION =====
function hideAllPages() {
    const pagesToHide = [
        'home-page', 'tools-page', 'floorplan-page', 'interior-page',
        'gallery-page', 'learn-page', 'billing-page', 'login-page',
        'how-it-works-section', 'results-section', 'final-cta', 'settings-page',
        'about-page', 'contact-page'
    ];
    pagesToHide.forEach(id => {
        const el = $(id);
        if (el) el.style.display = 'none';
    });
    // Remove active nav link
    document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active-link'));

    // Show generating banner if generation is in progress and we're navigating away
    const banner = $('generating-banner');
    if (typeof isGenerating !== 'undefined' && isGenerating && banner) {
        banner.style.display = 'flex';
    }
}

function navigateAbout() {
    hideAllPages();
    showPage(['about-page']);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function navigateContact() {
    hideAllPages();
    showPage(['contact-page']);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function submitContact() {
    const name = $('contact-name')?.value?.trim();
    const email = $('contact-email')?.value?.trim();
    const subject = $('contact-subject')?.value?.trim();
    const message = $('contact-message')?.value?.trim();
    if (!name || !email || !message) {
        showToast('Please fill in all required fields', 'error');
        return;
    }
    showToast('Message sent! We\'ll get back to you soon.', 'success');
    if ($('contact-name')) $('contact-name').value = '';
    if ($('contact-email')) $('contact-email').value = '';
    if ($('contact-subject')) $('contact-subject').value = '';
    if ($('contact-message')) $('contact-message').value = '';
}

function showPage(ids, extras = {}) {
    hideAllPages();
    ids.forEach(id => {
        const el = $(id);
        if (el) {
            el.style.display = extras.displayType || 'block';
            el.style.visibility = 'visible';
            el.style.opacity = '1';
        }
    });
    // Always show header and footer unless overridden
    if (extras.hideHeader) {
        if (header) header.style.display = 'none';
    } else {
        if (header) { header.style.display = 'block'; header.style.visibility = 'visible'; }
    }
    const footer = document.querySelector('footer');
    if (extras.hideFooter) {
        if (footer) footer.style.display = 'none';
    } else {
        if (footer) { footer.style.display = 'block'; footer.style.visibility = 'visible'; }
    }
    window.scrollTo(0, 0);
    if (hamburger) hamburger.classList.remove('active');
    if (nav) nav.classList.remove('mobile-open');
}

function showHome() {
    showPage(['home-page', 'how-it-works-section', 'final-cta']);
    currentPage = 'home';
    setActiveNavLink('Home');
}

function navigateToTools() {
    showPage(['tools-page', 'final-cta']);
    currentPage = 'tools';
    setActiveNavLink('Tools');
}

function navigateToTool(toolType) {
    showPage([toolType === 'floorplan' ? 'floorplan-page' : 'interior-page']);
    currentTool = toolType;
}

function navigateBack() { navigateToTools(); }

async function navigateGallery() {
    showPage(['gallery-page']);
    currentPage = 'gallery';
    setActiveNavLink('Gallery');
    await initializeGallery();
}

function navigateLearnMore() {
    showPage(['learn-page']);
    currentPage = 'learn';
}

function navigateBilling() {
    showPage(['billing-page']);
    currentPage = 'billing';
    setActiveNavLink('Pricing');
}

function navigateLogin() {
    showPage(['login-page'], { displayType: 'flex', hideHeader: true, hideFooter: true });
    currentPage = 'login';
}

function navigateSettings() {
    showPage(['settings-page']);
    currentPage = 'settings';
    setActiveNavLink('Settings');
    loadSettingsData();
}

function setActiveNavLink(linkText) {
    document.querySelectorAll('.nav-links a').forEach(a => {
        if (a.textContent.trim() === linkText) {
            a.classList.add('active-link');
        }
    });
}

// ===== TOAST NOTIFICATIONS =====
function showToast(message, type = 'success', duration = 4000) {
    const container = $('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    let icon = '✓';
    if (type === 'error') icon = '✕';
    else if (type === 'info') icon = 'ℹ';
    toast.innerHTML = `<span class="toast-icon">${icon}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ===== LOGIN / SIGNUP TAB SWITCH =====
function switchTab(tab, element) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    element.classList.add('active');
    if (tab === 'login') {
        $('login-tab').style.display = 'block';
        $('signup-tab').style.display = 'none';
    } else {
        $('login-tab').style.display = 'none';
        $('signup-tab').style.display = 'block';
    }
}

// ===== PROMPT BOT =====
function togglePromptBot(forceOpen) {
    const panel = $('prompt-bot-panel');
    const input = $('prompt-bot-input');
    if (!panel) return;
    const shouldOpen = typeof forceOpen === 'boolean' ? forceOpen : !panel.classList.contains('open');
    panel.classList.toggle('open', shouldOpen);
    if (shouldOpen && input) input.focus();
}

function appendPromptBotMessage(text, role = 'bot', structuredPrompt = null) {
    const container = $('prompt-bot-messages');
    if (!container) return;
    const msg = document.createElement('div');
    msg.className = `prompt-bot-msg ${role}`;
    msg.textContent = text;
    container.appendChild(msg);
    if (structuredPrompt) {
        const actions = document.createElement('div');
        actions.className = 'prompt-bot-actions';
        const useBtn = document.createElement('button');
        useBtn.textContent = 'Use This Prompt';
        useBtn.className = 'settings-btn primary';
        useBtn.style.fontSize = '0.8rem';
        useBtn.style.padding = '0.5rem 1rem';
        useBtn.onclick = () => applyStructuredPrompt(structuredPrompt);
        actions.appendChild(useBtn);
        msg.appendChild(actions);
    }
    container.scrollTop = container.scrollHeight;
}

function getActivePromptInput() {
    const floorInput = $('promptInput-floorplan');
    const interiorInput = $('promptInput-interior');
    if (currentTool === 'interior' && interiorInput) return interiorInput;
    if (currentTool === 'floorplan' && floorInput) return floorInput;
    if (interiorInput && interiorInput.offsetParent !== null) return interiorInput;
    if (floorInput && floorInput.offsetParent !== null) return floorInput;
    return floorInput || interiorInput;
}

function applyStructuredPrompt(promptText) {
    const input = getActivePromptInput();
    if (!input) { showToast('Open a tool and try again', 'error'); return; }
    input.value = promptText;
    input.focus();
    showToast('Prompt applied', 'success');
    togglePromptBot(false);
}

function buildStructuredPrompt(raw) {
    const style = document.querySelector('.style-card.active')?.textContent?.trim() || 'Modern';
    const isInterior = currentTool === 'interior';
    if (isInterior) {
        return `Interior design render. Style: ${style}. Requirements: ${raw}. Lighting: natural daylight, balanced exposure. Camera: wide angle, high detail.`;
    }
    return `Architectural floor plan. Requirements: ${raw}. Include room labels, circulation flow, doors/windows placement, and approximate dimensions.`;
}

function sendPromptBot() {
    const input = $('prompt-bot-input');
    const raw = input ? input.value.trim() : '';
    if (!raw) return;
    appendPromptBotMessage(raw, 'user');
    if (input) input.value = '';
    const structured = buildStructuredPrompt(raw);
    appendPromptBotMessage('Here is a structured prompt you can use:', 'bot', structured);
}

function selectStyle(style, element) {
    document.querySelectorAll('.style-card, .style-chip').forEach(card => card.classList.remove('active'));
    element.classList.add('active');
}

// ===== GALLERY =====
async function initializeGallery() {
    const gallery = $('galleryGrid');
    const sampleImages = [
        { title: 'Modern Apartment', img: 'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=350&fit=crop' },
        { title: 'Luxury Office Space', img: 'https://images.unsplash.com/photo-1590080876-a370a6b6d5d0?w=400&h=350&fit=crop' },
        { title: 'Minimalist Bedroom', img: 'https://images.unsplash.com/photo-1540932424986-7db079933576?w=400&h=350&fit=crop' }
    ];
    const savedImages = await getGalleryImages();
    const allImages = [...savedImages, ...sampleImages];
    gallery.innerHTML = allImages.map(item => `
    <div class="gallery-card">
      <img src="${item.img}" alt="${item.title}" class="gallery-image">
      <div class="gallery-overlay">
        <button class="overlay-btn" onclick="downloadImage('${item.img}')">Download</button>
      </div>
      <div style="padding: 0.75rem; font-size: 0.85rem; color: var(--text-light);">${item.title}</div>
    </div>
  `).join('');
}

async function getGalleryImages() {
    const userId = localStorage.getItem('user_id');
    const token = localStorage.getItem('access_token');
    if (userId && token) {
        try {
            const response = await fetch(`${BACKEND_BASE}/my_creations`, {
                method: 'GET',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }
            });
            if (response.ok) {
                const data = await response.json();
                return (data.creations || []).map(item => ({ title: item.title || 'Saved Design', img: item.image_data }));
            }
        } catch (error) { console.error('Error fetching gallery:', error); }
    }
    const saved = localStorage.getItem('spatiora-gallery');
    return saved ? JSON.parse(saved) : [];
}

async function addToGallery(imageUrl, title) {
    const userId = localStorage.getItem('user_id');
    const token = localStorage.getItem('access_token');
    if (userId && token) {
        try {
            const response = await fetch(`${BACKEND_BASE}/save_creation`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: imageUrl, title: title })
            });
            if (response.ok) { console.log('Image saved to database'); return; }
        } catch (error) { console.error('Error saving to database:', error); }
    }
    const gallery = localStorage.getItem('spatiora-gallery') ? JSON.parse(localStorage.getItem('spatiora-gallery')) : [];
    gallery.push({ id: Date.now(), title, img: imageUrl, createdAt: new Date().toISOString() });
    localStorage.setItem('spatiora-gallery', JSON.stringify(gallery));
}

// ===== AUTH STATE =====
function initializeAuthState() {
    const token = localStorage.getItem('access_token');
    const authButton = $('auth-button');
    const profileDropdown = $('profile-dropdown');
    if (token) {
        if (authButton) authButton.style.display = 'none';
        if (profileDropdown) profileDropdown.style.display = 'block';
        showHome();
    } else {
        if (authButton) {
            authButton.style.display = '';
            authButton.textContent = 'Login / Signup';
            authButton.onclick = () => navigateLogin();
        }
        if (profileDropdown) profileDropdown.style.display = 'none';
        navigateLogin();
    }
}

// ===== PROFILE DROPDOWN =====
function toggleProfileDropdown() {
    const menu = $('profile-menu');
    if (menu) menu.classList.toggle('open');
}

document.addEventListener('click', (e) => {
    const dropdown = $('profile-dropdown');
    const menu = $('profile-menu');
    if (dropdown && menu && !dropdown.contains(e.target)) {
        menu.classList.remove('open');
    }
});

// ===== AUTH FUNCTIONS =====
async function signup() {
    const username = $('signup-username').value;
    const email = $('signup-email').value;
    const password = $('signup-password').value;
    const privacyAgree = $('signup-privacy').checked;
    if (!username || !email || !password) { showToast('Please fill in all fields', 'error'); return; }
    if (!privacyAgree) { showToast('Please agree to Privacy Policy and Terms', 'error'); return; }
    try {
        const response = await fetch(`${BACKEND_BASE}/signup`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });
        const data = await response.json();
        if (data.success) {
            showToast('✓ Account created successfully! Welcome ' + username, 'success');
            localStorage.setItem('access_token', data.access_token);
            localStorage.setItem('user_id', data.user_id);
            localStorage.setItem('user', JSON.stringify(data.user));
            setTimeout(() => { initializeAuthState(); }, 1500);
        } else { showToast(data.error || 'Signup failed', 'error'); }
    } catch (error) { console.error('Error:', error); showToast('Signup failed. Please try again.', 'error'); }
}

async function login() {
    const email = $('login-email').value;
    const password = $('login-password').value;
    const privacyAgree = $('login-privacy').checked;
    if (!email || !password) { showToast('Please fill in all fields', 'error'); return; }
    if (!privacyAgree) { showToast('Please agree to Privacy Policy and Terms', 'error'); return; }
    try {
        const response = await fetch(`${BACKEND_BASE}/login`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        const data = await response.json();
        if (data.success) {
            showToast('✓ Welcome back! Logged in successfully', 'success');
            localStorage.setItem('access_token', data.access_token);
            localStorage.setItem('user', JSON.stringify(data.user));
            localStorage.setItem('user_id', data.user_id);
            setTimeout(() => { initializeAuthState(); }, 1500);
        } else { showToast(data.error || 'Login failed', 'error'); }
    } catch (error) { console.error('Error:', error); showToast('Login failed. Please try again.', 'error'); }
}

function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    localStorage.removeItem('user_id');
    showToast('Logged out successfully', 'success');
    setTimeout(() => { initializeAuthState(); }, 1000);
}

// ===== PRIVACY MODAL =====
function openPrivacyModal() { $('privacy-modal').style.display = 'flex'; }
function agreePrivacy() {
    $('privacy-modal').style.display = 'none';
    if ($('login-tab').style.display !== 'none') $('login-privacy').checked = true;
    if ($('signup-tab').style.display !== 'none') $('signup-privacy').checked = true;
}
function openTermsModal() { openPrivacyModal(); }

// ===== SETTINGS =====
function loadSettingsData() {
    const userRaw = localStorage.getItem('user');
    if (userRaw) {
        try {
            const user = JSON.parse(userRaw);
            const usernameInput = $('settings-username');
            const emailInput = $('settings-email');
            if (usernameInput && user.username) usernameInput.value = user.username;
            if (emailInput && user.email) emailInput.value = user.email;
        } catch (e) { console.error('Error loading user data:', e); }
    }
    // Load dark mode state
    const darkMode = localStorage.getItem('spatiora-dark-mode') === 'true';
    const toggle = $('dark-mode-toggle');
    if (toggle) toggle.classList.toggle('active', darkMode);
}

async function saveProfile() {
    const username = $('settings-username').value;
    const email = $('settings-email').value;
    const token = localStorage.getItem('access_token');
    if (!token) { showToast('Please log in first', 'error'); return; }
    if (!username && !email) { showToast('No changes to save', 'error'); return; }
    try {
        const response = await fetch(`${BACKEND_BASE}/update_profile`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email })
        });
        const data = await response.json();
        if (data.success) {
            showToast('✓ Profile updated successfully', 'success');
            localStorage.setItem('user', JSON.stringify(data.user));
        } else { showToast(data.error || 'Update failed', 'error'); }
    } catch (error) { console.error('Error:', error); showToast('Update failed. Please try again.', 'error'); }
}

async function changePassword() {
    const current = $('settings-current-password').value;
    const newPass = $('settings-new-password').value;
    const token = localStorage.getItem('access_token');
    if (!token) { showToast('Please log in first', 'error'); return; }
    if (!current || !newPass) { showToast('Please fill in both password fields', 'error'); return; }
    if (newPass.length < 6) { showToast('New password must be at least 6 characters', 'error'); return; }
    try {
        const response = await fetch(`${BACKEND_BASE}/update_profile`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: newPass })
        });
        const data = await response.json();
        if (data.success) {
            showToast('✓ Password changed successfully', 'success');
            $('settings-current-password').value = '';
            $('settings-new-password').value = '';
        } else { showToast(data.error || 'Password update failed', 'error'); }
    } catch (error) { console.error('Error:', error); showToast('Password update failed', 'error'); }
}

function toggleDarkMode() {
    const toggle = $('dark-mode-toggle');
    const isDark = !toggle.classList.contains('active');
    toggle.classList.toggle('active', isDark);
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem('spatiora-dark-mode', isDark.toString());
}

function switchSettingsTab(tabName, btn) {
    document.querySelectorAll('.settings-nav-item').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.settings-card').forEach(card => {
        card.style.display = card.dataset.section === tabName ? 'block' : 'none';
    });
}

// ===== DESIGN GENERATION =====
let isGenerating = false;

async function generateFloorPlan() {
    const prompt = $('promptInput-floorplan').value;
    if (!prompt.trim()) { alert('Please describe your space'); return; }
    await generateDesign('floorplan', prompt, '');
}

async function generateInteriorDesign() {
    const prompt = $('promptInput-interior').value;
    const style = document.querySelector('.style-chip.active, .style-card.active')?.textContent?.trim() || 'modern';
    if (!prompt.trim()) { alert('Please describe your design'); return; }
    await generateDesign('interior', prompt, style);
}

function backToEditor() { currentTool ? navigateToTool(currentTool) : showHome(); }
function generateNew() {
    if (currentTool === 'floorplan') generateFloorPlan();
    else if (currentTool === 'interior') generateInteriorDesign();
}

function returnToResults() {
    // Re-show the results section from the generating banner
    hideAllPages();
    $('results-section').style.display = 'block';
    $('generating-banner').style.display = 'none';
    window.scrollTo(0, 0);
}

async function generateDesign(toolType, prompt, style) {
    isGenerating = true;
    hideAllPages();
    $('results-section').style.display = 'block';
    $('generating-banner').style.display = 'none';

    // Show status as generating
    const statusEl = $('results-status');
    if (statusEl) {
        statusEl.innerHTML = '<i class="fas fa-spinner fa-spin" style="margin-right: 0.5rem;"></i>Generating...';
        statusEl.className = 'results-status generating';
    }

    const resultsContainer = $('resultsContainer');
    $('selection-buttons').innerHTML = '';

    // Create placeholder cards
    const grid = document.createElement('div');
    grid.className = 'result-grid';
    for (let i = 1; i <= 2; i++) {
        const card = document.createElement('div');
        card.className = 'result-card placeholder';
        card.innerHTML = `
      <div class="result-skeleton"></div>
      <div class="result-placeholder-body">
        <div class="result-spinner"></div>
        Generating option ${i}...
      </div>
    `;
        grid.appendChild(card);
    }
    resultsContainer.innerHTML = '';
    resultsContainer.appendChild(grid);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, tool_type: toolType, style, num_images: 2 })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let isFirst = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('data: ')) {
                    try {
                        const json = JSON.parse(trimmed.slice(6));
                        if (json.error) {
                            console.error("Generation error:", json.error);
                            if (isFirst) resultsContainer.innerHTML = `<p style="color: var(--error); text-align: center;">${json.error}</p>`;
                            continue;
                        }
                        if (json.url) {
                            if (isFirst) isFirst = false;
                            const gridEl = resultsContainer.querySelector('.result-grid');
                            const card = gridEl ? gridEl.querySelector('.result-card.placeholder') : null;
                            if (!card) continue;
                            card.classList.remove('placeholder');
                            card.style.cursor = 'pointer';
                            card.innerHTML = '';

                            const imgWrapper = document.createElement('div');
                            imgWrapper.className = 'result-img-wrapper';
                            const img = document.createElement('img');
                            img.src = json.url;
                            img.alt = json.title;
                            img.className = 'result-img';
                            imgWrapper.appendChild(img);

                            const content = document.createElement('div');
                            content.className = 'result-card-content';
                            const h3 = document.createElement('h3');
                            h3.className = 'result-card-title';
                            const existingCount = gridEl ? gridEl.querySelectorAll('.result-card:not(.placeholder)').length : 0;
                            h3.textContent = (json.title || 'Design Option') + ' ' + (existingCount + 1);

                            const btnGroup = document.createElement('div');
                            btnGroup.className = 'result-card-buttons';

                            const selectBtn = document.createElement('button');
                            selectBtn.className = 'result-btn result-btn-primary';
                            selectBtn.innerHTML = '<i class="fas fa-check" style="margin-right: 0.4rem;"></i>Choose';
                            selectBtn.onclick = () => { selectDesign(json.url, card); };

                            const dlBtn = document.createElement('button');
                            dlBtn.className = 'result-btn result-btn-secondary';
                            dlBtn.innerHTML = '<i class="fas fa-download" style="margin-right: 0.4rem;"></i>Download';
                            dlBtn.onclick = () => downloadImage(json.url);

                            btnGroup.appendChild(selectBtn);
                            btnGroup.appendChild(dlBtn);
                            content.appendChild(h3);
                            content.appendChild(btnGroup);
                            card.appendChild(imgWrapper);
                            card.appendChild(content);
                        }
                    } catch (e) { console.error('Error parsing JSON:', e); }
                }
            }
        }
    } catch (error) {
        console.error('Error:', error);
        $('resultsContainer').innerHTML = '<p style="color: var(--error); text-align: center;">Error generating design. Please try again.</p>';
    }

    // Generation complete
    isGenerating = false;
    $('generating-banner').style.display = 'none';
    if (statusEl) {
        statusEl.innerHTML = '<i class="fas fa-check-circle" style="margin-right: 0.5rem; color: var(--success);"></i>Complete';
        statusEl.className = 'results-status complete';
    }
}

function downloadImage(url) {
    if (!url) return;
    const link = document.createElement('a');
    link.href = url; link.download = 'design.png';
    document.body.appendChild(link); link.click(); document.body.removeChild(link);
}

function selectDesign(url, cardElement) {
    selectedImageUrl = url;
    document.querySelectorAll('.selected-badge').forEach(b => b.remove());
    const badge = document.createElement('div');
    badge.className = 'selected-badge';
    badge.innerHTML = '<i class="fas fa-check" style="margin-right: 0.3rem;"></i>Selected';
    cardElement.style.position = 'relative';
    cardElement.appendChild(badge);

    const selectionButtons = $('selection-buttons');
    selectionButtons.innerHTML = '';

    const editBtn = document.createElement('button');
    editBtn.className = 'result-btn result-btn-primary';
    editBtn.innerHTML = '<i class="fas fa-pen-fancy" style="margin-right: 0.5rem;"></i>Edit Selected';
    editBtn.onclick = () => { currentTool ? navigateToTool(currentTool) : showHome(); };

    const saveBtn = document.createElement('button');
    saveBtn.className = 'result-btn result-btn-save';
    saveBtn.innerHTML = '<i class="fas fa-bookmark" style="margin-right: 0.5rem;"></i>Save to Gallery';
    saveBtn.onclick = async () => {
        saveBtn.disabled = true;
        saveBtn.innerHTML = '<i class="fas fa-check" style="margin-right: 0.5rem;"></i>Saved!';
        await addToGallery(selectedImageUrl, 'Saved Design - ' + new Date().toLocaleDateString());
        showToast('✓ Design saved to gallery!', 'success');
    };

    const genNewBtn = document.createElement('button');
    genNewBtn.className = 'result-btn result-btn-secondary';
    genNewBtn.innerHTML = '<i class="fas fa-repeat" style="margin-right: 0.5rem;"></i>Generate Again';
    genNewBtn.onclick = () => generateNew();

    selectionButtons.appendChild(editBtn);
    selectionButtons.appendChild(saveBtn);
    selectionButtons.appendChild(genNewBtn);
}

function closeResults() { showHome(); }

// ===== INITIALIZATION =====
window.addEventListener('load', () => {
    // Apply saved dark mode
    const darkMode = localStorage.getItem('spatiora-dark-mode') === 'true';
    if (darkMode) document.documentElement.setAttribute('data-theme', 'dark');

    initializeAuthState();
    initializeGallery();

    const botInput = $('prompt-bot-input');
    if (botInput) {
        botInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') { event.preventDefault(); sendPromptBot(); }
        });
    }
});
