/*
 * Workspace data for the browser demo.
 *
 * A JavaScript counterpart of monitor/models.py + monitor/services.py backed
 * by localStorage: users, cameras, detection settings, incidents, notification
 * records, audit entries and the per-person confirmation / cooldown logic.
 * Everything stays in this browser; nothing is sent anywhere.
 */

const KEY = 'guardianai-fyp2-workspace-v1';
const TELEMETRY_KEY = 'guardianai-fyp2-telemetry-v1';
const SESSION_KEY = 'guardianai-fyp2-session';

export const KIND_LABELS = { browser: 'Browser webcam', video: 'Local video playback', worker: 'Server camera worker' };
export const STATE_LABELS = { open: 'Open', acknowledged: 'Acknowledged', resolved: 'Resolved', false_alarm: 'False alarm' };
export const DEMO_ACCOUNTS = [
  { username: 'kashan', password: '123', role: 'admin' },
  { username: 'caregiver', password: '123', role: 'caregiver' },
];
const PASSWORD_VERSION = 2;   // bump to reset the demo accounts' passwords in browsers that already have a workspace

function defaults() {
  return {
    users: [],
    cameras: [
      { id: 1, name: 'Bed bay 1–4', location: 'Ward A', kind: 'browser', enabled: true },
      { id: 2, name: 'Corridor east', location: 'Ground floor', kind: 'worker', enabled: true },
      { id: 3, name: 'Day room', location: 'Ward A', kind: 'video', enabled: true },
      { id: 4, name: 'Bed bay 7', location: 'Ward B', kind: 'browser', enabled: true },
    ],
    settings: { confidence: 0.8, consecutive_frames: 3, cooldown_seconds: 30, email_enabled: false, email_recipient: '' },
    events: [],
    deliveries: [],
    audit: [],
    next: { camera: 5, event: 1, delivery: 1 },
  };
}

let data = null;
let telemetry = {};          // camera id -> { last_seen (ms), last_status, last_result }
const tracks = new Map();    // `${camera}:${track_id}` -> { streak, last_seen (s), last_alert (s) }

function load() {
  if (data) return data;
  try { data = JSON.parse(localStorage.getItem(KEY)) || defaults(); } catch { data = defaults(); }
  for (const [k, v] of Object.entries(defaults())) if (data[k] === undefined) data[k] = v;
  try { telemetry = JSON.parse(localStorage.getItem(TELEMETRY_KEY)) || {}; } catch { telemetry = {}; }
  return data;
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(data)); } catch { /* storage full or blocked */ } }
function saveTelemetry() { try { localStorage.setItem(TELEMETRY_KEY, JSON.stringify(telemetry)); } catch { /* ignore */ } }

async function digest(username, password) {
  const bytes = new TextEncoder().encode(`guardianai:${username}:${password}`);
  if (!crypto.subtle) return 'plain:' + username + ':' + password;   // non-secure context fallback (http://)
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, '0')).join('');
}

// ------------------------------------------------------------------ accounts
export async function init() {
  load();
  if (!data.users.length || data.passwordVersion !== PASSWORD_VERSION) {
    for (const a of DEMO_ACCOUNTS) {
      const hash = await digest(a.username, a.password), u = data.users.find(x => x.username === a.username);
      if (u) u.hash = hash; else data.users.push({ username: a.username, role: a.role, hash });
    }
    data.passwordVersion = PASSWORD_VERSION;
    save();
  }
}
export function currentUser() {
  load();
  try {
    const s = JSON.parse(sessionStorage.getItem(SESSION_KEY));
    return s ? data.users.find(u => u.username === s.username) || null : null;
  } catch { return null; }
}
export const isStaff = user => !!user && user.role === 'admin';
export async function login(username, password) {
  load();
  const user = data.users.find(u => u.username === username.trim());
  if (!user || user.hash !== await digest(user.username, password)) return null;
  sessionStorage.setItem(SESSION_KEY, JSON.stringify({ username: user.username }));
  return user;
}
export function logout() { sessionStorage.removeItem(SESSION_KEY); }
export function users() { return load().users.map(u => ({ username: u.username, role: u.role })); }
export async function addUser(username, password, role, actor) {
  load();
  username = username.trim();
  if (!/^[\w.@+-]{1,150}$/.test(username)) throw new Error('Use letters, digits and @/./+/-/_ only.');
  if (password.length < 3) throw new Error('Password must be at least 3 characters.');
  if (data.users.some(u => u.username === username)) throw new Error('That username already exists.');
  data.users.push({ username, role: role === 'admin' ? 'admin' : 'caregiver', hash: await digest(username, password) });
  audit(actor, 'user_created', `User ${username} (${role})`);
  save();
}
export function removeUser(username, actor) {
  load();
  if (data.users.filter(u => u.role === 'admin').length <= 1 && data.users.find(u => u.username === username)?.role === 'admin') {
    throw new Error('Keep at least one administrator.');
  }
  data.users = data.users.filter(u => u.username !== username);
  audit(actor, 'user_removed', `User ${username}`);
  save();
}

// ------------------------------------------------------------------ cameras & settings
export function cameras() { return load().cameras; }
export function camera(id) { return load().cameras.find(c => c.id === Number(id)) || null; }
export function saveCamera(fields, actor, id = null) {
  load();
  const name = String(fields.name || '').trim(), location = String(fields.location || '').trim();
  if (!name || name.length > 80) throw new Error('Name is required (max 80 characters).');
  if (!location || location.length > 100) throw new Error('Location is required (max 100 characters).');
  if (!KIND_LABELS[fields.kind]) throw new Error('Choose an input source.');
  let cam = id ? camera(id) : null;
  if (id && !cam) throw new Error('Camera not found.');
  if (!cam) { cam = { id: data.next.camera++ }; data.cameras.push(cam); }
  Object.assign(cam, { name, location, kind: fields.kind, enabled: !!fields.enabled });
  audit(actor, 'camera_saved', `Camera #${cam.id}`);
  save();
  return cam;
}
export function settings() { return load().settings; }
export function saveSettings(fields, actor) {
  load();
  const s = {
    confidence: Number(fields.confidence), consecutive_frames: Number(fields.consecutive_frames),
    cooldown_seconds: Number(fields.cooldown_seconds), email_enabled: !!fields.email_enabled,
    email_recipient: String(fields.email_recipient || '').trim(),
  };
  const errors = {};
  if (!(s.confidence >= 0.5 && s.confidence <= 1)) errors.confidence = 'Enter a value between 0.5 and 1.';
  if (!(Number.isInteger(s.consecutive_frames) && s.consecutive_frames >= 1 && s.consecutive_frames <= 15)) errors.consecutive_frames = 'Enter a whole number from 1 to 15.';
  if (!(Number.isInteger(s.cooldown_seconds) && s.cooldown_seconds >= 1 && s.cooldown_seconds <= 600)) errors.cooldown_seconds = 'Enter a whole number from 1 to 600.';
  if (s.email_recipient && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(s.email_recipient)) errors.email_recipient = 'Enter a valid email address.';
  if (s.email_enabled && !s.email_recipient) errors.email_recipient = 'Enter a recipient to enable email.';
  if (Object.keys(errors).length) { const e = new Error('Please correct the highlighted fields.'); e.fields = errors; throw e; }
  data.settings = s;
  audit(actor, 'settings_saved', 'Alert thresholds and email configuration updated');
  save();
}

// ------------------------------------------------------------------ ingestion (services.ingest)
function statusFor(result) {
  if (!result.ready) return 'preview';
  const labels = new Set(result.predictions.map(p => p.label));
  return ['fall', 'warning', 'normal', 'unknown'].find(l => labels.has(l)) || 'no_subject';
}

/** Record one detector result for a camera. Returns the incidents it created. */
export function ingest(cameraId, result, nowMs = Date.now()) {
  load();
  const cam = camera(cameraId);
  if (!cam) throw new Error('Camera not found');
  if (!cam.enabled) throw new Error('Camera is disabled');
  const cfg = data.settings, now = nowMs / 1000, created = [];
  telemetry[cam.id] = { last_seen: nowMs, last_status: statusFor(result), last_result: result };
  saveTelemetry();
  const present = new Set(result.ready ? result.predictions.map(p => p.track_id) : []);
  for (const [key, t] of tracks) {
    if (key.startsWith(cam.id + ':') && !present.has(key.slice(String(cam.id).length + 1))) t.streak = 0;
    if (now - t.last_seen > 3600) tracks.delete(key);
  }
  if (!result.ready) return created;
  for (const p of result.predictions) {
    const key = `${cam.id}:${p.track_id}`;
    const t = tracks.get(key) || { streak: 0, last_seen: now, last_alert: null };
    if (now - t.last_seen > 3) t.streak = 0;
    t.streak = p.label === 'fall' && p.confidence >= cfg.confidence ? t.streak + 1 : 0;
    t.last_seen = now;
    if (t.streak >= cfg.consecutive_frames && (t.last_alert === null || now - t.last_alert >= cfg.cooldown_seconds)) {
      const event = {
        id: data.next.event++, camera: cam.id, created_at: new Date(nowMs).toISOString(), track_id: p.track_id,
        label: 'fall', confidence: p.confidence, model_version: result.model_version, evidence: p.evidence || {},
        is_demo: false, status: 'open', notes: '', updated_by: null, updated_at: null,
      };
      data.events.push(event);
      created.push(event);
      t.last_alert = now;
    }
    tracks.set(key, t);
  }
  for (const e of created) notify(e);
  if (created.length) save();
  return created;
}

function notify(event) {
  data.deliveries.push({ id: data.next.delivery++, event: event.id, created_at: new Date().toISOString(),
    channel: 'dashboard', status: 'recorded', detail: 'Available to signed-in users' });
  if (data.settings.email_enabled && !event.is_demo) {
    data.deliveries.push({ id: data.next.delivery++, event: event.id, created_at: new Date().toISOString(),
      channel: 'email', status: 'unavailable_in_browser', detail: 'Email needs the server deployment; no message was sent.' });
  }
}

export function createDemo(cameraId, actor) {
  load();
  const cam = camera(cameraId);
  if (!cam || !cam.enabled) throw new Error('Camera not found or disabled');
  const recent = data.events.find(e => e.camera === cam.id && e.is_demo && Date.now() - Date.parse(e.created_at) < 10000);
  if (recent) return { event: recent, fresh: false };
  const event = {
    id: data.next.event++, camera: cam.id, created_at: new Date().toISOString(), track_id: 'DEMO-P1', label: 'fall',
    confidence: 0.94, model_version: 'manual-demo-v1', is_demo: true, status: 'open', notes: '', updated_by: null, updated_at: null,
    evidence: { source: 'Manually generated fixture; not model inference', purpose: 'Verify review and dashboard workflows' },
  };
  data.events.push(event);
  notify(event);
  audit(actor, 'demo_event_created', `Event #${event.id}`);
  save();
  return { event, fresh: true };
}

// ------------------------------------------------------------------ queries
export function status() {
  load();
  const now = Date.now(), today = localDate(new Date());
  const cams = data.cameras.map(c => {
    const t = telemetry[c.id];
    const fresh = c.enabled && t && now - t.last_seen < 10000;
    return { ...c, status: fresh ? t.last_status : 'offline', result: fresh ? t.last_result : {}, last_seen: t ? t.last_seen : null };
  });
  const open = data.events.filter(e => e.status === 'open');
  return {
    cameras: cams,
    open_events: open.filter(e => !e.is_demo).length,
    demo_events: open.filter(e => e.is_demo).length,
    falls_today: data.events.filter(e => !e.is_demo && localDate(new Date(e.created_at)) === today).length,
    alerts: [...open].sort((a, b) => b.id - a.id).slice(0, 6),
  };
}
export function events(filter = {}) {
  load();
  let list = [...data.events].sort((a, b) => b.id - a.id);
  if (STATE_LABELS[filter.status]) list = list.filter(e => e.status === filter.status);
  if (filter.mode === 'real') list = list.filter(e => !e.is_demo);
  if (filter.mode === 'demo') list = list.filter(e => e.is_demo);
  const q = (filter.q || '').trim().slice(0, 100).toLowerCase();
  if (q) list = list.filter(e => camera(e.camera)?.name.toLowerCase().includes(q) || e.track_id.toLowerCase().includes(q) || e.notes.toLowerCase().includes(q));
  return list;
}
export function event(id) { return load().events.find(e => e.id === Number(id)) || null; }
export function deliveriesFor(eventId) { return load().deliveries.filter(d => d.event === Number(eventId)); }
export function deliveries(limit = 50) { return [...load().deliveries].sort((a, b) => b.id - a.id).slice(0, limit); }
export function reviewEvent(id, fields, actor) {
  load();
  const e = event(id);
  if (!e) throw new Error('Event not found');
  if (!STATE_LABELS[fields.status]) throw new Error('Choose a review state.');
  if ((fields.notes || '').length > 4000) throw new Error('Use at most 4000 characters.');
  e.status = fields.status; e.notes = fields.notes || ''; e.updated_by = actor.username; e.updated_at = new Date().toISOString();
  audit(actor, 'incident_reviewed', `Event #${e.id}: ${e.status}`);
  save();
  return e;
}
export function report(days) {
  load();
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const start = new Date(today); start.setDate(start.getDate() - (days - 1));
  const qs = data.events.filter(e => !e.is_demo && new Date(e.created_at) >= start);
  const daily = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(today); d.setDate(d.getDate() - i);
    const key = localDate(d);
    daily.push({ label: d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }), count: qs.filter(e => localDate(new Date(e.created_at)) === key).length });
  }
  const max = Math.max(1, ...daily.map(d => d.count));
  for (const d of daily) d.height = Math.max(2, d.count / max * 100);
  const byCamera = {};
  for (const e of qs) byCamera[e.camera] = (byCamera[e.camera] || 0) + 1;
  return {
    total: qs.length, false_alarms: qs.filter(e => e.status === 'false_alarm').length, resolved: qs.filter(e => e.status === 'resolved').length, daily,
    by_camera: Object.entries(byCamera).map(([id, count]) => ({ name: camera(id)?.name || `Camera #${id}`, count })).sort((a, b) => b.count - a.count),
  };
}
export function exportCsv(filter) {
  const safe = v => { const s = String(v); return /^\s*[=+\-@]/.test(s) || /^[\t\r\n]/.test(s) ? "'" + s : s; };
  const cell = v => { const s = safe(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  const rows = [['id', 'created_at', 'camera', 'track_id', 'label', 'confidence', 'model_version', 'is_demo', 'review_status', 'notes']];
  for (const e of events(filter)) rows.push([e.id, e.created_at, camera(e.camera)?.name || '', e.track_id, e.label, e.confidence, e.model_version, e.is_demo ? 'True' : 'False', e.status, e.notes]);
  return rows.map(r => r.map(cell).join(',')).join('\r\n') + '\r\n';
}
export function audit(actor, action, detail) {
  load();
  data.audit.push({ created_at: new Date().toISOString(), actor: actor?.username || null, action, detail: String(detail).slice(0, 250) });
  if (data.audit.length > 500) data.audit.splice(0, data.audit.length - 500);
}
export function clearIncidents(actor) {
  load();
  data.events = []; data.deliveries = []; data.next.event = 1; data.next.delivery = 1;
  telemetry = {}; tracks.clear(); saveTelemetry();
  audit(actor, 'incidents_cleared', 'All incidents and notification records removed');
  save();
}
export function resetWorkspace() { localStorage.removeItem(KEY); localStorage.removeItem(TELEMETRY_KEY); data = null; telemetry = {}; tracks.clear(); }
export function localDate(d) { return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; }
