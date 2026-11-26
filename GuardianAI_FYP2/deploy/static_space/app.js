/*
 * GuardianAI browser workspace.
 *
 * The Django application's pages (templates/*.html) rebuilt as a single-page
 * app for the free static Space: sign-in, monitor wall, cameras, camera
 * monitoring, event history and review, notifications, reports and settings.
 * Data lives in this browser (store.js); detection runs in this tab (engine.js).
 * Markup and CSS classes follow the Django templates so both look the same.
 */

import * as store from './store.js';
import { engine, start as engineStart, stop as engineStop, stopAll as engineStopAll, subscribe, drawOverlay, labelText, isPartial } from './engine.js';

const root = document.getElementById('root');
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pad2 = n => String(n).padStart(2, '0');
const cam2 = id => 'CAM-' + pad2(id);
const fmt = {
  list: iso => { const d = new Date(iso); return `${pad2(d.getDate())} ${MONTHS[d.getMonth()]} ${d.getFullYear()} · ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`; },
  short: iso => { const d = new Date(iso); return `${pad2(d.getDate())} ${MONTHS[d.getMonth()]} · ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`; },
  time: iso => { const d = new Date(iso); return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`; },
  detail: iso => { const d = new Date(iso); return `${pad2(d.getDate())} ${MONTHS[d.getMonth()]} ${d.getFullYear()}, ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())} ${Intl.DateTimeFormat().resolvedOptions().timeZone}`; },
  any: iso => new Date(iso).toLocaleString(),
};
let flashes = [];
const flash = msg => flashes.push(msg);
function toast(message) { const el = document.getElementById('toast'); if (!el) return; el.textContent = message; el.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => { el.hidden = true; }, 5000); }
const go = hash => { location.hash = hash; };

// ------------------------------------------------------------------ routing
function route() {
  const h = location.hash.replace(/^#/, '') || '/';
  const [path, query = ''] = h.split('?');
  return { path, parts: path.split('/').filter(Boolean), query: new URLSearchParams(query) };
}

let shellUser = null, unmount = null, theme = 'light';
try { theme = localStorage.getItem('guardian-theme') === 'dark' ? 'dark' : 'light'; } catch { /* ignore */ }

function render() {
  const user = store.currentUser();
  const r = route();
  if (unmount) { try { unmount(); } catch (e) { console.error(e); } unmount = null; }
  if (!user) {
    shellUser = null;
    document.body.className = 'login-page' + (theme === 'dark' ? ' dark' : '');
    root.innerHTML = loginPage(r).html;
    loginPage(r).mount();
    return;
  }
  if (r.path === '/login') { go('/'); return; }
  if (!shellUser || shellUser.username !== user.username) renderShell(user);
  const page = pageFor(r, user);
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('active', a.dataset.page.split(' ').includes(page.name)));
  const main = document.getElementById('main');
  main.innerHTML = flashes.map(m => `<div class="notice" role="status">${esc(m)}</div>`).join('') + page.html;
  flashes = [];
  window.scrollTo(0, 0);
  unmount = page.mount ? page.mount() : null;
}

function renderShell(user) {
  shellUser = user;
  const staff = store.isStaff(user);
  document.body.className = theme === 'dark' ? 'dark' : '';
  root.innerHTML = `<div class="app-shell"><aside class="sidebar"><a href="#/" class="brand"><span class="brand-mark">G<span>+</span></span><span>Guardian<span class="violet">AI</span><small>CARE, WITH A CLOSER EYE.</small></span></a><div class="nav-label">WORKSPACE</div><nav aria-label="Main navigation"><a href="#/" data-page="wall camera"><span>▦</span> Monitor wall</a><a href="#/cameras" data-page="cameras"><span>◉</span> Cameras</a><a href="#/events" data-page="events event"><span>◷</span> Event history</a><a href="#/notifications" data-page="notifications"><span>♧</span> Notifications</a><a href="#/reports" data-page="reports"><span>▥</span> Reports</a>${staff ? '<a href="#/settings" data-page="settings"><span>⚙</span> Settings</a>' : ''}</nav><div class="sidebar-bottom"><div class="project-note"><span class="dot violet-dot"></span> FYP-2 workspace<p>Real-time fall detection<br>& caregiver response</p></div><div class="profile"><span class="avatar">${esc(user.username[0].toUpperCase())}</span><div><strong>${esc(user.username)}</strong><small>${staff ? 'Administrator' : 'Caregiver'}</small></div><form class="signout"><button class="icon-button" title="Sign out" aria-label="Sign out">↗</button></form></div></div></aside>
<div class="main-shell"><header class="topbar"><span><span class="dot"></span> BROWSER DEMO WORKSPACE · DATA STAYS ON THIS DEVICE</span><div><a id="live-link" class="live-link" href="#/" hidden></a><time id="clock"></time><form class="mobile-signout signout"><button class="icon-button" aria-label="Sign out" title="Sign out">↗</button></form><button type="button" id="theme-toggle" class="icon-button" aria-label="Toggle dark theme">◐</button></div></header><main id="main"></main><footer>GuardianAI <span>FYP-2026-G6 · Research & development</span><span>Model scores are not measured accuracy.</span></footer></div></div><div id="toast" role="status" hidden></div>`;
  root.querySelectorAll('form.signout').forEach(f => f.addEventListener('submit', e => { e.preventDefault(); engineStopAll(); store.logout(); shellUser = null; go('/login'); render(); }));
  document.getElementById('theme-toggle').addEventListener('click', () => { theme = theme === 'dark' ? 'light' : 'dark'; document.body.classList.toggle('dark', theme === 'dark'); try { localStorage.setItem('guardian-theme', theme); } catch { /* ignore */ } });
  const clock = document.getElementById('clock'); const tick = () => { clock.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }; tick(); setInterval(tick, 1000);
  updateLiveLink();
}

function updateLiveLink() {
  const a = document.getElementById('live-link');
  if (!a) return;
  const ids = [...engine.sources.keys()];
  a.hidden = !ids.length;
  if (ids.length === 1) { const cam = store.camera(ids[0]); a.href = `#/cameras/${ids[0]}`; a.textContent = `● LIVE · ${(cam?.name || 'camera').toUpperCase()}`; }
  else if (ids.length) { a.href = '#/'; a.textContent = `● LIVE · ${ids.length} CAMERAS`; }
}

function pageFor(r, user) {
  const p = r.parts;
  if (p.length === 0) return wallPage();
  if (p[0] === 'cameras') {
    if (p.length === 1) return camerasPage(user);
    if (p[1] === 'new') return cameraFormPage(user, null);
    if (p[2] === 'edit') return cameraFormPage(user, p[1]);
    return cameraPage(user, p[1]);
  }
  if (p[0] === 'events') return p.length === 1 ? eventsPage(r.query) : eventPage(user, p[1]);
  if (p[0] === 'notifications') return notificationsPage();
  if (p[0] === 'reports') return reportsPage(r.query);
  if (p[0] === 'settings') return settingsPage(user);
  return { name: '', html: '<div class="panel empty"><h3>Page not found</h3><p><a href="#/">Back to the monitor wall</a></p></div>' };
}

const heading = (eyebrow, title, text, action = '') => `<div class="page-heading"><div>${eyebrow}<h1>${title}</h1><p>${text}</p></div>${action}</div>`;
const forbidden = () => ({ name: '', html: '<div class="panel forbidden"><h2>Administrator access required</h2><p>Sign in with an administrator account to open this page.</p><a class="button" href="#/">Back to the monitor wall</a></div>' });

// ------------------------------------------------------------------ login
function loginPage(r) {
  const html = `<main class="login-layout"><section class="login-story"><a class="brand" href="#/"><span class="brand-mark">G<span>+</span></span><span>GuardianAI</span></a><span class="eyebrow">A LITTLE MORE AWARE. A LITTLE MORE ASSURED.</span><h1>Every moment<br>of care matters<span class="violet">.</span></h1><p>A single workspace to monitor camera sources, review possible falls, and help caregivers respond.</p><div class="orb"><svg viewBox="0 0 200 200" fill="none" aria-hidden="true"><circle cx="100" cy="55" r="16"/><path d="M100 76v58m-40-35 40-15 40 15m-66 66 26-31 26 31"/><circle class="orb-ring" cx="100" cy="100" r="85"/></svg></div><div class="login-caption">IN-BROWSER PROCESSING <span>•</span> HUMAN REVIEW <span>•</span> MODEL READY</div></section><section class="login-form panel"><span class="eyebrow violet">WELCOME TO GUARDIANAI</span><h2>Your monitoring<br>workspace awaits.</h2><p class="muted">Sign in with your project account.</p><div id="login-error" class="notice danger" hidden>Sign-in failed. Check your username and password.</div><form id="login-form"><label for="id_username">Username</label><input id="id_username" name="username" autocomplete="username" required autofocus><label for="id_password">Password</label><input id="id_password" type="password" name="password" autocomplete="current-password" required><button class="button primary full">Sign in to workspace <span>→</span></button></form><div class="demo-accounts"><strong>Demo workspace accounts</strong><p>Administrator: <code>kashan</code> / <code>123</code><br>Caregiver: <code>caregiver</code> / <code>123</code></p></div><p class="fine">Accounts and incidents are stored only in this browser. Nothing is uploaded.</p></section></main>`;
  const mount = () => {
    const form = document.getElementById('login-form');
    form?.addEventListener('submit', async e => {
      e.preventDefault();
      const user = await store.login(form.username.value, form.password.value);
      if (!user) { document.getElementById('login-error').hidden = false; return; }
      const next = r.query.get('next') || '/';
      shellUser = null;
      if (location.hash === '#' + next) render(); else go(next);
    });
  };
  return { html, mount };
}

// ------------------------------------------------------------------ monitor wall
function cameraCard(c) {
  return `<a class="camera-card" href="#/cameras/${c.id}" data-camera="${c.id}"><div class="camera-preview"><div class="camera-top"><span>${cam2(c.id)}</span><span class="camera-status">OFFLINE</span></div><div class="preview-center"><span class="camera-glyph">⌾</span><strong>${esc(c.name)}</strong><small class="camera-message">Open source to begin monitoring</small></div><div class="camera-bottom"><span>${store.KIND_LABELS[c.kind]}</span><span class="camera-subjects">— subjects</span></div></div><div class="camera-caption"><div><h3>${esc(c.name)}</h3><p>${esc(c.location)}</p></div><span class="arrow-circle">↗</span></div></a>`;
}
function alertItems(alerts) {
  if (!alerts.length) return '<div class="empty"><span class="empty-icon">✓</span><h3>Nothing awaiting review</h3><p>New incidents will appear here when an alert is recorded.</p></div>';
  return alerts.map(e => `<a class="alert-item" href="#/events/${e.id}"><span class="pill danger">${e.is_demo ? 'DEMO · ' : ''}FALL</span><h3>${esc(store.camera(e.camera)?.name || '')}</h3><p>${fmt.time(e.created_at)} · score ${e.confidence.toFixed(2)}</p><span>Review incident →</span></a>`).join('');
}
function wallPage() {
  const st = store.status();
  const html = heading('<span class="eyebrow">OVERVIEW / LIVE OPERATIONS</span>', 'A closer eye on care<span class="violet">.</span>', 'Your camera sources and the moments that need attention.', '<a class="button primary" href="#/cameras">Manage cameras <span>↗</span></a>')
    + `<div class="mode-banner"><span class="mode-symbol">◎</span><div><strong>In-browser model ready · YOLOv8-Pose + FallNet v2</strong><p>Live metrics appear when a source sends frames. Demo incidents are always labelled. Everything stays on this device.</p></div><span class="pill">BROWSER WORKSPACE</span></div>`
    + `<div class="stats"><article><span class="stat-icon">◉</span><small>ACTIVE SOURCES</small><strong id="kpi-cameras">0<span> / ${st.cameras.length}</span></strong><p>Frames received in the last 10s</p></article><article><span class="stat-icon green">◍</span><small>TRACKED SUBJECTS</small><strong id="kpi-subjects">0</strong><p>Reported by the in-browser model</p></article><article><span class="stat-icon pink">△</span><small>OPEN INCIDENTS</small><strong id="kpi-events">0</strong><p>Real events awaiting review</p></article><article><span class="stat-icon amber">◷</span><small>FALL ALERTS TODAY</small><strong id="kpi-falls">0</strong><p>Demo incidents excluded</p></article></div>`
    + `<div class="wall-layout"><section><div class="section-heading"><h2>Monitor wall <span class="count">${st.cameras.length} sources</span></h2><span class="wall-tools"><button type="button" id="wall-live" class="button small danger" hidden></button><span id="sync-label" class="fine">Refreshing every 3 seconds</span></span></div><div class="camera-grid">${st.cameras.length ? st.cameras.map(cameraCard).join('') : '<div class="panel empty"><h3>Your wall starts here</h3><p>Add a camera to start the monitoring workflow.</p><a class="button" href="#/cameras/new">Add a camera</a></div>'}</div><div class="subtle-note">ⓘ Live sources show their video here, several at once. Start them on their camera pages; they keep running until you press Stop.</div></section><aside class="alert-column panel"><div class="section-heading"><h2>Needs attention</h2><span class="live-dot"></span></div><p class="muted fine">Open incidents, including labelled demos</p><div id="alert-list">${alertItems(st.alerts)}</div><a class="text-link" href="#/events">View all events →</a><div class="mini-info"><strong>A person stays in the loop.</strong><p>Review the evidence, acknowledge the incident, and record the outcome.</p></div></aside></div>`;
  const mount = () => {
    const refresh = () => {
      const s = store.status(), set = (id, v) => { const e = document.getElementById(id); if (e) e.innerHTML = v; };
      set('kpi-cameras', `${s.cameras.filter(c => c.status !== 'offline').length}<span> / ${s.cameras.length}</span>`);
      set('kpi-subjects', s.cameras.reduce((n, c) => n + (c.result.predictions?.length || 0), 0));
      set('kpi-events', s.open_events); set('kpi-falls', s.falls_today);
      for (const c of s.cameras) {
        const card = document.querySelector(`[data-camera="${c.id}"]`); if (!card) continue;
        card.querySelector('.camera-status').textContent = c.status === 'unknown' ? ((c.result.predictions || []).every(isPartial) ? 'PARTIAL VIEW' : 'WARMING UP') : c.status.replace('_', ' ').toUpperCase();
        card.querySelector('.camera-subjects').textContent = (c.result.predictions?.length || 0) + ' subjects';
        card.querySelector('.camera-message').textContent = c.status === 'offline' ? 'Open source to begin monitoring' : (c.result.message || 'Source telemetry connected');
      }
      const list = document.getElementById('alert-list'); if (list) list.innerHTML = alertItems(s.alerts);
      set('sync-label', 'Updated ' + new Date().toLocaleTimeString());
    };
    refresh();
    // Live sources are shown in their cards: the source's own <video> plus the box overlay.
    const mounted = new Map(); let alive = true;
    const syncLive = () => {
      for (const [id, src] of engine.sources) {
        const card = document.querySelector(`[data-camera="${id}"]`);
        if (!card || mounted.has(id)) continue;
        const preview = card.querySelector('.camera-preview'), layer = document.createElement('div'), canvas = document.createElement('canvas');
        layer.className = 'camera-live'; layer.append(src.video, canvas); preview.prepend(layer); preview.classList.add('is-live');
        mounted.set(id, { layer, canvas });
      }
      for (const [id, m] of mounted) if (!engine.sources.has(id)) { m.layer.remove(); document.querySelector(`[data-camera="${id}"] .camera-preview`)?.classList.remove('is-live'); mounted.delete(id); }
      const btn = document.getElementById('wall-live');
      if (btn) { btn.hidden = !engine.sources.size; btn.textContent = `● Live: ${engine.sources.size === 1 ? store.camera([...engine.sources.keys()][0])?.name + ' · Stop' : engine.sources.size + ' cameras · Stop all'}`; }
    };
    const draw = () => { if (!alive) return; for (const [id, m] of mounted) drawOverlay(m.canvas, id); requestAnimationFrame(draw); };
    syncLive(); draw();
    document.getElementById('wall-live')?.addEventListener('click', () => { engineStopAll(); syncLive(); refresh(); });
    const timer = setInterval(refresh, 3000);
    const off = subscribe(type => { if (type === 'state') syncLive(); if (type === 'update' || type === 'incident') refreshSoon(); });
    let pending = 0; const refreshSoon = () => { if (pending) return; pending = setTimeout(() => { pending = 0; refresh(); }, 500); };
    return () => { alive = false; clearInterval(timer); off(); clearTimeout(pending); };
  };
  return { name: 'wall', html, mount };
}

// ------------------------------------------------------------------ cameras
function camerasPage(user) {
  const staff = store.isStaff(user), cams = store.cameras();
  const rows = cams.length ? cams.map(c => `<tr><td><strong>${esc(c.name)}</strong><small>${cam2(c.id)}</small></td><td>${esc(c.location)}</td><td>${store.KIND_LABELS[c.kind]}</td><td><span class="pill">${c.enabled ? 'Enabled' : 'Disabled'}</span></td><td><a href="#/cameras/${c.id}">Open →</a>${staff ? ` · <a href="#/cameras/${c.id}/edit">Edit</a>` : ''}</td></tr>`).join('') : '<tr><td colspan="5">No cameras configured. Add your first source.</td></tr>';
  const html = heading('<span class="eyebrow">WORKSPACE / SOURCES</span>', 'Camera management', 'Connect the spaces you want to monitor.', staff ? '<a class="button primary" href="#/cameras/new">+ Add camera</a>' : '')
    + `<div class="panel table-wrap"><table><thead><tr><th>Camera</th><th>Location</th><th>Input source</th><th>Configuration</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table></div>`
    + `<div class="three-columns spaced"><article class="panel"><span class="step">01</span><h3>Browser webcam</h3><p>Allow camera access in your browser. Frames are analysed in this tab while monitoring is running; nothing is uploaded.</p></article><article class="panel"><span class="step">02</span><h3>Video playback</h3><p>Choose a video from your device and analyse it in the browser. The video file never leaves your device.</p></article><article class="panel"><span class="step">03</span><h3>Server camera worker</h3><p>USB cameras and RTSP/CCTV streams are read by the Python worker in the Django deployment. This browser workspace shows the concept only.</p></article></div>`;
  return { name: 'cameras', html };
}

function cameraFormPage(user, id) {
  if (!store.isStaff(user)) return forbidden();
  const cam = id ? store.camera(id) : null;
  if (id && !cam) return pageFor({ parts: ['missing'] }, user);
  const v = cam || { name: '', location: '', kind: 'browser', enabled: true };
  const html = heading('<span class="eyebrow">WORKSPACE / CONFIGURATION</span>', cam ? 'Edit camera' : 'Add camera', 'Name the source and choose how frames will reach GuardianAI.')
    + `<section class="panel form-panel"><form id="camera-form"><div class="field"><label for="id_name">Name</label><input id="id_name" name="name" maxlength="80" required value="${esc(v.name)}"></div><div class="field"><label for="id_location">Location</label><input id="id_location" name="location" maxlength="100" required value="${esc(v.location)}"></div><div class="field"><label for="id_kind">Kind</label><select id="id_kind" name="kind">${Object.entries(store.KIND_LABELS).map(([k, l]) => `<option value="${k}" ${k === v.kind ? 'selected' : ''}>${l}</option>`).join('')}</select></div><div class="field check"><label><input type="checkbox" id="id_enabled" name="enabled" ${v.enabled ? 'checked' : ''}> Enabled</label></div><div id="form-error" class="notice danger" hidden></div><div class="actions"><button class="button primary">Save camera</button><a class="button" href="#/cameras">Cancel</a></div></form></section>`;
  const mount = () => {
    const form = document.getElementById('camera-form');
    form.addEventListener('submit', e => {
      e.preventDefault();
      try { store.saveCamera({ name: form.name.value, location: form.location.value, kind: form.kind.value, enabled: form.enabled.checked }, user, cam?.id); flash('Camera saved.'); go('/cameras'); }
      catch (err) { const el = document.getElementById('form-error'); el.textContent = err.message; el.hidden = false; }
    });
  };
  return { name: 'cameras', html, mount };
}

// ------------------------------------------------------------------ camera monitoring
function subjectCards(container, preds, running) {
  if (!preds.length) { container.className = 'empty'; container.replaceChildren(); container.textContent = running ? 'No subjects returned by the model.' : 'No model predictions yet.'; return; }
  container.className = '';
  if (container.firstChild?.nodeType === 3) container.replaceChildren();
  const keep = new Set();
  for (const p of preds) {
    keep.add(p.track_id);
    let card = container.querySelector(`[data-track="${CSS.escape(p.track_id)}"]`);
    if (!card) { card = document.createElement('div'); card.className = 'subject-card'; card.dataset.track = p.track_id; card.append(document.createElement('h3'), document.createElement('p'), document.createElement('pre')); container.append(card); }
    card.querySelector('h3').textContent = `${p.track_id} · ${labelText(p)}`;
    card.querySelector('p').textContent = isPartial(p) ? 'Move back so hips and legs are in view. The model needs the whole body.' : p.label === 'unknown' ? 'Collecting 1.5 s of motion history' : `Model score: ${(p.confidence * 100).toFixed(1)}%`;
    const ev = JSON.stringify(p.evidence, null, 2); const pre = card.querySelector('pre'); if (pre.textContent !== ev) pre.textContent = ev;
  }
  for (const c of [...container.children]) if (!keep.has(c.dataset.track)) c.remove();
}

function cameraPage(user, id) {
  const cam = store.camera(id);
  if (!cam) return pageFor({ parts: ['missing'] }, user);
  const staff = store.isStaff(user), mine = engine.sources.has(cam.id);
  const recentHtml = () => { const recent = store.events({}).filter(e => e.camera === cam.id).slice(0, 5); return recent.length ? recent.map(e => `<a class="event-link" href="#/events/${e.id}">#${e.id} · ${store.STATE_LABELS[e.status]} ${e.is_demo ? '<span class="pill">DEMO</span>' : ''}</a>`).join('') : '<p>No incidents for this source.</p>'; };
  let controls = '';
  if (staff && cam.enabled && cam.kind !== 'worker') {
    controls = (cam.kind === 'browser' ? `<button class="button primary" id="start-webcam" ${mine ? 'disabled' : ''}>Start camera</button>` : `<label class="button" for="video-file">Choose video</label><input type="file" id="video-file" accept="video/*" hidden><button class="button primary" id="start-video" disabled>Start analysis</button>`) + `<button class="button" id="stop-source" ${mine ? '' : 'disabled'}>Stop</button>`;
  } else if (cam.kind !== 'worker') {
    controls = `<span class="fine">${!cam.enabled ? 'Source disabled. Enable it in camera settings.' : 'An administrator can start camera capture.'}</span>`;
  }
  const html = heading(`<a class="eyebrow" href="#/">← MONITOR WALL</a>`, esc(cam.name), `${esc(cam.location)} · ${store.KIND_LABELS[cam.kind]} · ${cam2(cam.id)}`, staff ? `<a class="button" href="#/cameras/${cam.id}/edit">Edit source</a>` : '')
    + `<div class="wall-layout"><section><div class="panel"><div id="monitor"><div class="video-stage"><canvas id="overlay"></canvas><div id="video-placeholder" ${mine ? 'hidden' : ''}><span class="camera-glyph">⌾</span><h2>Ready when you are</h2><p>${cam.kind === 'worker' ? 'Source preview runs on the server worker.' : 'Start a source to see the video here.'}</p></div><span class="video-tag" id="source-state">${mine ? 'ANALYSING' : 'STOPPED'}</span></div><div class="monitor-controls">${controls}<span id="frame-info" class="fine">No frames processed</span></div><div id="inference-message" class="notice">${cam.kind === 'worker' ? 'Server camera workers (USB / RTSP) run with the Django deployment. This browser workspace cannot read them; use a Browser webcam or Local video playback source.' : esc(engine.message)}</div></div></div><div class="panel spaced"><h2>Tracked subjects</h2><div id="subjects" class="empty">No model predictions yet.</div></div></section><aside><div class="panel"><span class="eyebrow">MODEL OUTPUT</span><h2>Evidence, not assumptions.</h2><p>Person IDs, model scores, and measured features appear here when the in-browser model returns them.</p><p class="fine">A score is confidence in one prediction. Accuracy requires a labelled, held-out evaluation set.</p><p class="fine" id="runtime-info"></p></div><div class="panel spaced"><h3>Verify the review workflow</h3><p>Create a clearly labelled sample fall incident without a camera or trained model.</p>${staff && cam.enabled ? '<button class="button full" id="create-demo">Create demo incident</button>' : ''}</div><div class="panel spaced"><h3>Recent incidents</h3><div id="recent-incidents">${recentHtml()}</div></div></aside></div>`;
  const mount = () => {
    const stage = document.querySelector('.video-stage'), overlay = document.getElementById('overlay');
    const message = document.getElementById('inference-message'), info = document.getElementById('frame-info'), state = document.getElementById('source-state'), subjects = document.getElementById('subjects'), placeholder = document.getElementById('video-placeholder'), runtime = document.getElementById('runtime-info');
    const startWebcam = document.getElementById('start-webcam'), startVideo = document.getElementById('start-video'), stopBtn = document.getElementById('stop-source'), fileInput = document.getElementById('video-file');
    let file = null, alive = true, starting = false, note = '';
    const sync = () => {
      const src = engine.sources.get(cam.id), running = !!src;
      if (running && src.video.parentElement !== stage) stage.prepend(src.video);
      placeholder.hidden = running; state.textContent = running ? 'ANALYSING' : starting ? 'CONNECTING' : 'STOPPED';
      if (cam.kind !== 'worker') message.textContent = running ? src.message : starting ? engine.message : note || engine.message;
      info.textContent = running ? `${src.latencyMs} ms processing · ${src.preds.length} subjects · ${src.fps.toFixed(1)} frames/s analysed` : 'No frames processed';
      runtime.textContent = engine.backend ? `Runtime: ${engine.backend} · ${engine.size} px · ${engine.modelVersion}${engine.sources.size > 1 ? ` · ${engine.sources.size} live sources share the model` : ''}` : '';
      subjectCards(subjects, running ? src.preds : [], running);
      if (startWebcam) startWebcam.disabled = running || starting;
      if (startVideo) startVideo.disabled = !file || starting || running;
      if (stopBtn) stopBtn.disabled = !running;
      updateLiveLink();
    };
    const off = subscribe((type, detail) => { sync(); if (type === 'incident') { toast('New incident recorded: #' + detail.map(e => e.id).join(', #')); document.getElementById('recent-incidents').innerHTML = recentHtml(); } });
    const draw = () => { if (!alive) return; drawOverlay(overlay, cam.id); requestAnimationFrame(draw); };
    draw(); sync();
    const begin = async opts => { starting = true; note = ''; sync(); try { await engineStart(cam.id, opts); } catch (e) { note = 'Source could not start: ' + e.message; } starting = false; sync(); };
    startWebcam?.addEventListener('click', () => begin({ kind: 'webcam' }));
    fileInput?.addEventListener('change', () => { file = fileInput.files[0] || null; if (file) note = `Selected ${file.name}. Start analysis to process it in this browser.`; sync(); });
    startVideo?.addEventListener('click', () => file && begin({ kind: 'video', file }));
    stopBtn?.addEventListener('click', () => engineStop(cam.id));
    document.getElementById('create-demo')?.addEventListener('click', () => {
      const { fresh } = store.createDemo(cam.id, user);
      flash(fresh ? 'Demo incident created. It is excluded from real reports and external email.' : 'A demo event was just created. Open it in Events.');
      go('/events');
    });
    return () => { alive = false; off(); };
  };
  return { name: 'camera', html, mount };
}

// ------------------------------------------------------------------ events
function eventsPage(query) {
  const filter = { q: query.get('q') || '', status: query.get('status') || '', mode: query.get('mode') || '' };
  const all = store.events(filter), perPage = 20, pages = Math.max(1, Math.ceil(all.length / perPage));
  const page = Math.min(pages, Math.max(1, Number(query.get('page') || 1))), slice = all.slice((page - 1) * perPage, page * perPage);
  const qs = p => `#/events?q=${encodeURIComponent(filter.q)}&status=${encodeURIComponent(filter.status)}&mode=${encodeURIComponent(filter.mode)}&page=${p}`;
  const rows = slice.length ? slice.map(e => `<tr><td><strong>#${e.id}</strong><small>${fmt.list(e.created_at)}</small></td><td>${esc(store.camera(e.camera)?.name || '')}<small>${esc(e.track_id)}</small></td><td><span class="pill danger">${e.label.toUpperCase()}</span>${e.is_demo ? ' <span class="pill">DEMO</span>' : ''}</td><td>${e.confidence.toFixed(2)}</td><td>${store.STATE_LABELS[e.status]}</td><td><a href="#/events/${e.id}">Review →</a></td></tr>`).join('')
    : '<tr><td colspan="6"><div class="empty"><h3>No events match this view</h3><p>Real detections and labelled demo incidents will appear here.</p></div></td></tr>';
  const html = heading('<span class="eyebrow">ACTIVITY / INCIDENT LOG</span>', 'Every event, in context.', 'Review alerts and record what actually happened.', '<button class="button" id="export-csv">↓ Export CSV</button>')
    + `<form class="filters panel" id="filters"><input name="q" value="${esc(filter.q)}" placeholder="Search camera, person or notes" aria-label="Search events"><select name="status" aria-label="Review status"><option value="">All review states</option>${Object.entries(store.STATE_LABELS).map(([v, l]) => `<option value="${v}" ${filter.status === v ? 'selected' : ''}>${l}</option>`).join('')}</select><select name="mode" aria-label="Event mode"><option value="">Real + demo</option><option value="real" ${filter.mode === 'real' ? 'selected' : ''}>Real only</option><option value="demo" ${filter.mode === 'demo' ? 'selected' : ''}>Demo only</option></select><button class="button primary">Apply filters</button><a href="#/events">Reset</a></form>`
    + `<div class="panel table-wrap"><table><thead><tr><th>Event / time</th><th>Camera / subject</th><th>Detection</th><th>Model score</th><th>Review state</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`
    + `<div class="pagination"><span>${all.length} events · Page ${page} of ${pages}</span>${page > 1 ? `<a href="${qs(page - 1)}">← Previous</a>` : ''}${page < pages ? `<a href="${qs(page + 1)}">Next →</a>` : ''}</div>`;
  const mount = () => {
    const form = document.getElementById('filters');
    form.addEventListener('submit', e => { e.preventDefault(); go(`/events?q=${encodeURIComponent(form.q.value)}&status=${encodeURIComponent(form.status.value)}&mode=${encodeURIComponent(form.mode.value)}`); });
    document.getElementById('export-csv').addEventListener('click', () => {
      const blob = new Blob([store.exportCsv(filter)], { type: 'text/csv' }), url = URL.createObjectURL(blob), a = document.createElement('a');
      a.href = url; a.download = 'guardian-events.csv'; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
  };
  return { name: 'events', html, mount };
}

function eventPage(user, id) {
  const e = store.event(id);
  if (!e) return pageFor({ parts: ['missing'] }, user);
  const cam = store.camera(e.camera), deliveries = store.deliveriesFor(e.id);
  const html = heading('<a class="eyebrow" href="#/events">← EVENT HISTORY</a>', `Incident #${e.id}`, `${esc(cam?.name || '')} · ${fmt.detail(e.created_at)}`, `<span class="pill danger">${e.is_demo ? 'DEMO · ' : ''}${e.label.toUpperCase()}</span>`)
    + (e.is_demo ? '<div class="notice">This incident was generated manually for testing. It is excluded from real reports and email delivery.</div>' : '')
    + `<div class="two-columns"><section><div class="panel"><h2>Detection record</h2><dl><dt>Tracked person</dt><dd>${esc(e.track_id)}</dd><dt>Model version</dt><dd>${esc(e.model_version)}</dd><dt>Model score</dt><dd>${e.confidence.toFixed(3)}</dd><dt>Review state</dt><dd>${store.STATE_LABELS[e.status]}</dd></dl><h3>Evidence returned by the model</h3><pre>${esc(JSON.stringify(e.evidence, null, 2))}</pre><p class="fine">Raw frames are not retained.</p></div><div class="panel spaced"><h3>Notification record</h3>${deliveries.length ? deliveries.map(d => `<div class="delivery-row"><strong>${esc(d.channel)}</strong><span class="pill">${esc(d.status)}</span><p>${esc(d.detail)}</p></div>`).join('') : '<p>No notification attempts recorded.</p>'}</div></section><section class="panel form-panel"><h2>Human review</h2><p>Acknowledge the incident, then record the outcome. Use false alarm for an incorrect detection.</p><form id="review-form"><div class="field"><label for="id_status">Status</label><select id="id_status" name="status">${Object.entries(store.STATE_LABELS).map(([v, l]) => `<option value="${v}" ${e.status === v ? 'selected' : ''}>${l}</option>`).join('')}</select></div><div class="field"><label for="id_notes">Notes</label><textarea id="id_notes" name="notes" rows="3" maxlength="4000">${esc(e.notes)}</textarea></div><button class="button primary">Save review</button></form>${e.updated_by ? `<p class="fine">Last reviewed by ${esc(e.updated_by)} · ${fmt.any(e.updated_at)}</p>` : ''}</section></div>`;
  const mount = () => {
    const form = document.getElementById('review-form');
    form.addEventListener('submit', ev => { ev.preventDefault(); try { store.reviewEvent(e.id, { status: form.status.value, notes: form.notes.value }, user); flash('Incident review saved.'); render(); } catch (err) { toast(err.message); } });
  };
  return { name: 'event', html, mount };
}

// ------------------------------------------------------------------ notifications, reports, settings
function notificationsPage() {
  const cfg = store.settings(), rows = store.deliveries(50);
  const html = heading('<span class="eyebrow">RESPONSE / CHANNELS</span>', 'Notification centre', 'See what was recorded, submitted, or failed.', '<button class="button" id="enable-notifications">Enable browser alerts</button>')
    + `<div class="three-columns"><article class="panel"><span class="pill success">AVAILABLE</span><h2>Dashboard</h2><p>Incidents appear in the monitor wall and event history for signed-in users.</p></article><article class="panel"><span class="pill">${cfg.email_enabled ? 'ENABLED' : 'DISABLED'}</span><h2>Email</h2><p>Browser workspace: no external email is sent. The Django deployment records SMTP submission outcomes here.</p></article><article class="panel"><span class="pill">WHILE PAGE IS OPEN</span><h2>Browser alerts</h2><p>Opt in on this device. New real incidents can trigger notifications while this workspace is open. This is not background push.</p></article></div>`
    + `<div class="section-heading spaced"><h2>Recent notification activity</h2><span class="fine">Latest 50 records</span></div><div class="panel table-wrap"><table><thead><tr><th>Time</th><th>Incident</th><th>Channel</th><th>State</th><th>Details</th></tr></thead><tbody>${rows.length ? rows.map(d => { const e = store.event(d.event); return `<tr><td>${fmt.short(d.created_at)}</td><td><a href="#/events/${d.event}">#${d.event}${e?.is_demo ? ' · DEMO' : ''}</a></td><td>${esc(d.channel)}</td><td><span class="pill">${esc(d.status)}</span></td><td>${esc(d.detail)}</td></tr>`; }).join('') : '<tr><td colspan="5">No notification attempts yet.</td></tr>'}</tbody></table></div>`;
  const mount = () => { document.getElementById('enable-notifications').addEventListener('click', enableNotifications); };
  return { name: 'notifications', html, mount };
}
async function enableNotifications() {
  if (!('Notification' in window)) { toast('Browser notifications are not supported.'); return; }
  const p = await Notification.requestPermission();
  try { localStorage.setItem('guardian-notifications', p === 'granted' ? 'yes' : 'no'); } catch { /* ignore */ }
  toast(p === 'granted' ? 'Alerts enabled while the workspace is open.' : 'Browser permission was not granted.');
}

function reportsPage(query) {
  const days = ['1', '7', '30'].includes(query.get('days')) ? Number(query.get('days')) : 7, r = store.report(days);
  const html = heading('<span class="eyebrow">INSIGHTS / RECORDED ACTIVITY</span>', 'Care, with perspective.', 'Real incident records by local calendar date. Demo events are excluded.', `<form id="range" class="actions"><select name="days" aria-label="Report range"><option value="1" ${days === 1 ? 'selected' : ''}>Today</option><option value="7" ${days === 7 ? 'selected' : ''}>7 days</option><option value="30" ${days === 30 ? 'selected' : ''}>30 days</option></select><button class="button">Update</button></form>`)
    + `<div class="stats"><article><small>RECORDED FALL ALERTS</small><strong>${r.total}</strong><p>In the selected period</p></article><article><small>MARKED FALSE ALARM</small><strong>${r.false_alarms}</strong><p>Based on human review</p></article><article><small>RESOLVED</small><strong>${r.resolved}</strong><p>Based on human review</p></article><article><small>MODEL ACCURACY</small><strong>—</strong><p>Requires labelled evaluation</p></article></div>`
    + `<div class="two-columns"><section class="panel"><h2>Daily alert volume</h2><p class="fine">Recorded real incidents by local date</p><div class="chart">${r.daily.map(d => `<div class="chart-column" title="${d.label}: ${d.count}"><span>${d.count}</span><i style="height:${d.height}%"></i><small>${d.label}</small></div>`).join('')}</div></section><section class="panel"><h2>By camera</h2>${r.by_camera.length ? r.by_camera.map(row => `<div class="rank-row"><span>${esc(row.name)}</span><strong>${row.count}</strong></div>`).join('') : '<div class="empty"><h3>No real incidents recorded</h3><p>Start a camera source to gather evidence.</p></div>'}</section></div>`
    + '<div class="notice spaced">Missed falls, recall, and accuracy cannot be inferred from alerts alone. The held-out evaluation (leave-one-subject-out on GMDCSA-24) is in the project model card.</div>';
  const mount = () => { const f = document.getElementById('range'); f.addEventListener('submit', e => { e.preventDefault(); go(`/reports?days=${f.days.value}`); }); };
  return { name: 'reports', html, mount };
}

function settingsPage(user) {
  if (!store.isStaff(user)) return forbidden();
  const s = store.settings(), accounts = store.users();
  const field = (name, label, input, help = '') => `<div class="field" data-field="${name}"><label for="id_${name}">${label}</label>${input}${help ? `<span class="helptext">${help}</span>` : ''}</div>`;
  const html = heading('<span class="eyebrow">WORKSPACE / CONFIGURATION</span>', 'Detection settings', 'Control when a model prediction becomes a recorded alert.')
    + `<div class="two-columns"><section class="panel form-panel"><form id="settings-form">${field('confidence', 'Confidence', `<input type="number" id="id_confidence" name="confidence" step="0.01" min="0.5" max="1" value="${s.confidence}">`, 'Minimum fall score, 0.5 to 1.')}${field('consecutive_frames', 'Consecutive frames', `<input type="number" id="id_consecutive_frames" name="consecutive_frames" min="1" max="15" value="${s.consecutive_frames}">`)}${field('cooldown_seconds', 'Cooldown seconds', `<input type="number" id="id_cooldown_seconds" name="cooldown_seconds" min="1" max="600" value="${s.cooldown_seconds}">`)}<div class="field check" data-field="email_enabled"><label><input type="checkbox" id="id_email_enabled" name="email_enabled" ${s.email_enabled ? 'checked' : ''}> Email enabled</label></div>${field('email_recipient', 'Email recipient', `<input type="email" id="id_email_recipient" name="email_recipient" value="${esc(s.email_recipient)}">`)}<button class="button primary">Save settings</button></form></section><aside><div class="panel"><h2>How an alert is confirmed</h2><ol class="explanation"><li>A person receives a fall prediction above the confidence threshold.</li><li>The same person must satisfy the consecutive-frame requirement.</li><li>The cooldown prevents repeated alerts for that camera and person.</li></ol><p>Missing tracks, low scores, and a gap longer than three seconds reset the streak.</p></div><div class="panel spaced"><h3>Notification transport</h3><p>In this browser workspace email is recorded but never sent. The Django deployment uses SMTP environment variables for real submissions.</p><p>SMS, mobile push, and physical buzzers are future integrations.</p></div></aside></div>`
    + `<div class="two-columns spaced"><section class="panel"><h2>Workspace accounts</h2><p class="fine">Accounts exist only in this browser. Caregivers can view and review incidents; administrators can also change cameras, settings and start sources.</p><div id="accounts">${accounts.map(a => `<div class="account-row"><span><strong>${esc(a.username)}</strong><span class="pill">${a.role === 'admin' ? 'Administrator' : 'Caregiver'}</span></span>${a.username !== user.username ? `<button class="text-button danger" data-remove="${esc(a.username)}">Remove</button>` : '<span class="fine">signed in</span>'}</div>`).join('')}</div><form id="account-form" class="inline-form"><div><label for="acc_user">Username</label><input id="acc_user" required maxlength="150"></div><div><label for="acc_pass">Password</label><input id="acc_pass" type="password" required minlength="3"></div><div><label for="acc_role">Role</label><select id="acc_role"><option value="caregiver">Caregiver</option><option value="admin">Administrator</option></select></div><button class="button">Add account</button></form><div id="account-error" class="notice danger spaced" hidden></div></section><section class="panel"><h2>Browser workspace data</h2><p>Incidents, notification records and telemetry are stored in this browser only. Clear them before a fresh demonstration.</p><button class="button" id="clear-incidents">Clear incidents and telemetry</button></section></div>`;
  const mount = () => {
    const form = document.getElementById('settings-form');
    form.addEventListener('submit', e => {
      e.preventDefault();
      form.querySelectorAll('.errorlist').forEach(el => el.remove());
      try { store.saveSettings({ confidence: form.confidence.value, consecutive_frames: form.consecutive_frames.value, cooldown_seconds: form.cooldown_seconds.value, email_enabled: form.email_enabled.checked, email_recipient: form.email_recipient.value }, user); flash('Settings saved. New predictions use these values.'); render(); }
      catch (err) { for (const [name, msg] of Object.entries(err.fields || {})) { const f = form.querySelector(`[data-field="${name}"]`); const ul = document.createElement('ul'); ul.className = 'errorlist'; ul.innerHTML = `<li>${esc(msg)}</li>`; f.append(ul); } toast(err.message); }
    });
    const accForm = document.getElementById('account-form'), accErr = document.getElementById('account-error');
    accForm.addEventListener('submit', async e => { e.preventDefault(); try { await store.addUser(accForm.querySelector('#acc_user').value, accForm.querySelector('#acc_pass').value, accForm.querySelector('#acc_role').value, user); flash('Account created.'); render(); } catch (err) { accErr.textContent = err.message; accErr.hidden = false; } });
    document.querySelectorAll('[data-remove]').forEach(b => b.addEventListener('click', () => { if (!confirm(`Remove account ${b.dataset.remove}?`)) return; try { store.removeUser(b.dataset.remove, user); flash('Account removed.'); render(); } catch (err) { toast(err.message); } }));
    document.getElementById('clear-incidents').addEventListener('click', () => { if (!confirm('Remove all incidents and notification records from this browser?')) return; engineStopAll(); store.clearIncidents(user); flash('Incidents and telemetry cleared.'); render(); });
  };
  return { name: 'settings', html, mount };
}

// ------------------------------------------------------------------ boot
subscribe((type, detail) => {
  if (type === 'state') updateLiveLink();
  if (type !== 'incident') return;
  if (!document.querySelector('#monitor')) toast('New incident recorded: #' + detail.map(e => e.id).join(', #'));
  let enabled = false; try { enabled = localStorage.getItem('guardian-notifications') === 'yes'; } catch { /* ignore */ }
  if (enabled && 'Notification' in window && Notification.permission === 'granted') {
    for (const e of detail) { const n = new Notification('GuardianAI: possible fall', { body: `${store.camera(e.camera)?.name || 'Camera'} · review incident #${e.id}`, tag: 'guardian-' + e.id, icon: './icon.png' }); n.onclick = () => { window.focus(); go(`/events/${e.id}`); }; }
  }
});
window.addEventListener('hashchange', render);
store.init().then(render);
