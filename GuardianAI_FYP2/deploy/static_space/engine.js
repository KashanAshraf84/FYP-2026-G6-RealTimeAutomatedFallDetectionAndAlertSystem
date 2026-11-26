/*
 * Monitor engine for the browser workspace.
 *
 * Any number of webcam / video sources, analysed entirely in this tab:
 *   frame -> YOLOv8n-Pose (ONNX Runtime Web: WebGPU when available, else
 *   WebAssembly in a worker) -> pipeline.js tracker + features -> FallNet v2
 *   -> store.ingest() (the same confirmation / cooldown rule as the server).
 *
 * Smoothness: one frame is analysed at a time and the sources take turns, each
 * waiting long enough that the model is busy at most DUTY of the time, so the
 * page stays responsive however many cameras are live. Boxes are animated
 * between analyses. The GPU path runs 640 px frames when a frame takes under
 * SLOW_MS, otherwise 480 px. Each source owns its <video> element, so monitoring
 * continues while you move between pages; the wall and the camera page show it.
 */

import { P, PersonTracker, detectorStep, decodePoses, letterbox } from './pipeline.js';
import * as store from './store.js';

const TARGET_MS = 160;              // one source: analyse ~6 times per second
const MAX_MS = 500;                 // never slower than 2 analyses per second per source
const DUTY = 0.7;                   // share of time the model may be busy
const SLOW_MS = 130;                // above this per frame on 640 px, use 480 px (Intel UHD WebGPU: 640 ≈ 170 ms, 480 ≈ 90 ms)
const MAX_W = 640;                  // frames are analysed at <= 640 px wide, like the Django page
const COLORS = { fall: '#ff6b91', warning: '#f9c65a', normal: '#43ddb0', unknown: '#a0a8b8' };

const listeners = new Set();
const idCounters = new Map();       // camera id -> next person number (never reused while the tab lives)

export const engine = {
  loading: false,
  sources: new Map(),               // camera id -> source
  message: 'In-browser model ready. Start a source to connect.',
  backend: '', size: 0, modelVersion: '', threads: 1,
};

let pose = null, fallnet = null, busy = false, rafId = 0, latencies = [];
const work = document.createElement('canvas'), lb = document.createElement('canvas');

export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
const emit = (type, detail) => { for (const fn of listeners) { try { fn(type, detail); } catch (e) { console.error(e); } } };
export const isPartial = p => p.label === 'unknown' && p.evidence?.status === 'partial_view';
export const labelText = p => (p.label !== 'unknown' ? p.label.toUpperCase() : isPartial(p) ? 'PARTIAL VIEW' : 'WARMING UP');

// ------------------------------------------------------------------ models
async function pickBackend() {
  ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;
  engine.threads = window.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.numThreads = engine.threads;
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return { ep: 'webgpu', size: 640, label: 'WebGPU' };
    } catch { /* fall through */ }
  }
  ort.env.wasm.proxy = true;        // run WebAssembly inference in a worker so the page never freezes
  return { ep: 'wasm', size: 480, label: `WebAssembly · ${engine.threads} thread${engine.threads > 1 ? 's' : ''}` };
}

async function loadPose(ep, size) {
  const session = await ort.InferenceSession.create(`./models/yolov8n-pose-${size}.onnx`, { executionProviders: [ep] });
  engine.size = size;
  return session;
}

/** Median time of a few blank inferences, so the input size can be chosen before the first real frame. */
async function timeInference(session, size) {
  const x = new ort.Tensor('float32', new Float32Array(3 * size * size).fill(0.45), [1, 3, size, size]);
  const feeds = { [session.inputNames[0]]: x }, times = [];
  await session.run(feeds);
  for (let i = 0; i < 3; i++) { const t = performance.now(); await session.run(feeds); times.push(performance.now() - t); }
  return times.sort((a, b) => a - b)[1];
}

let loadingPromise = null;
function loadModels() {
  if (pose && fallnet) return Promise.resolve();
  if (!loadingPromise) loadingPromise = (async () => {
    engine.loading = true; engine.message = 'Loading the models (about 14 MB, cached after the first visit)…'; emit('update');
    try {
      const choice = await pickBackend();
      engine.backend = choice.label; engine.ep = choice.ep;
      try {
        pose = await loadPose(choice.ep, choice.size);
        if (choice.size === 640 && await timeInference(pose, 640) > SLOW_MS) { const old = pose; pose = await loadPose(choice.ep, 480); old.release?.(); }
      } catch (e) {
        if (choice.ep === 'wasm') throw e;
        ort.env.wasm.proxy = true; engine.ep = 'wasm'; engine.backend = `WebAssembly · ${engine.threads} thread${engine.threads > 1 ? 's' : ''}`;
        pose = await loadPose('wasm', 480);
      }
      fallnet = await ort.InferenceSession.create('./models/fallnet_v2.onnx', { executionProviders: ['wasm'] });
      engine.modelVersion = (await fetch('./models/model_version.txt').then(r => (r.ok ? r.text() : '')).catch(() => '')).trim() || 'fallnet-v2';
    } finally { engine.loading = false; loadingPromise = null; }
  })();
  return loadingPromise;
}

const runFallnet = async (batch, n) =>
  (await fallnet.run({ features: new ort.Tensor('float32', batch, [n, P.seqLen, 62]) })).probs.data;

// ------------------------------------------------------------------ analysis
function grab(v) {
  const s = Math.min(MAX_W / v.videoWidth, 1);
  const w = Math.round(v.videoWidth * s), h = Math.round(v.videoHeight * s);
  if (work.width !== w || work.height !== h) { work.width = w; work.height = h; }
  work.getContext('2d', { willReadFrequently: true }).drawImage(v, 0, 0, w, h);
  const geo = letterbox(w, h, engine.size), S = geo.size;
  if (lb.width !== S) { lb.width = S; lb.height = S; }
  const ctx = lb.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(114,114,114)'; ctx.fillRect(0, 0, S, S);
  ctx.drawImage(work, geo.padX, geo.padY, geo.newW, geo.newH);
  const { data } = ctx.getImageData(0, 0, S, S), A = S * S, t = new Float32Array(3 * A);
  for (let i = 0, j = 0; i < A; i++, j += 4) { t[i] = data[j] / 255; t[A + i] = data[j + 1] / 255; t[2 * A + i] = data[j + 2] / 255; }
  return { tensor: t, geo, w, h };
}

const round = (v, d) => Math.round(v * 10 ** d) / 10 ** d;
const boxAt = (tw, now) => { const k = Math.min(1, Math.max(0, (now - tw.t0) / (tw.t1 - tw.t0 || 1))); return tw.from.map((v, i) => v + (tw.to[i] - v) * k); };
const mediaTime = src => (src.kind === 'video' ? src.video.currentTime : performance.now() / 1000);

/** Wait between two analyses of the same source so that all sources together respect DUTY. */
function interval() {
  const med = latencies.length ? [...latencies].sort((a, b) => a - b)[latencies.length >> 1] : 60;
  return Math.min(MAX_MS, Math.max(TARGET_MS, engine.sources.size * med / DUTY));
}

async function analyse(src, t) {
  busy = true;
  const started = performance.now();
  try {
    const { tensor, geo, w, h } = grab(src.video);
    const out = await pose.run({ [pose.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, geo.size, geo.size]) });
    const raw = out[pose.outputNames[0]];
    const dets = decodePoses(raw.data, raw.dims[2], geo);
    const preds = await detectorStep(src.tracker, t, dets, h, runFallnet);
    if (!engine.sources.has(src.cameraId)) { busy = false; return; }   // stopped while analysing
    const people = src.tracker.active(t);
    const predictions = preds.map((p, i) => ({
      track_id: p.trackId, label: p.label, confidence: p.confidence,
      bbox: [p.box[0] / w, p.box[1] / h, p.box[2] / w, p.box[3] / h].map(v => Math.min(1, Math.max(0, v))),
      evidence: p.partial ? { status: 'partial_view', note: 'Only head and shoulders are visible. Move back so hips and legs are in view.' }
        : p.probs ? {
          torso_tilt_from_vertical_deg: round(p.evidence.tiltDeg, 1),
          hip_vertical_velocity_frame_heights_per_s: round(p.evidence.hipVy, 3),
          class_probabilities: { normal: round(p.probs[0], 4), warning: round(p.probs[1], 4), fall: round(p.probs[2], 4) },
          window_s: P.windowS, tracked_for_s: round(t - (people[i]?.firstT ?? t), 1), classifier: engine.modelVersion, runtime: `browser · ${engine.backend} · ${engine.size}px`,
        } : { status: 'warming_up', history_needed_s: P.minHistoryS },
    }));
    const message = !predictions.length ? 'No person detected.'
      : predictions.every(isPartial) ? 'Body not fully visible: move back so hips and legs are in view.'
        : predictions.every(p => p.label === 'unknown') ? 'Collecting motion history.'
          : `${predictions.length} person(s) tracked · ${engine.backend}`;
    const result = { predictions, model_version: engine.modelVersion, ready: true, message };
    const created = store.ingest(src.cameraId, result, Date.now());
    const latency = performance.now() - started;
    latencies.push(latency); if (latencies.length > 12) latencies.shift();
    const wall = performance.now(), gap = interval();
    src.arrivals.push(wall); src.arrivals = src.arrivals.filter(x => wall - x < 3000);
    src.fps = src.arrivals.length > 1 ? (src.arrivals.length - 1) / ((wall - src.arrivals[0]) / 1000) : 0;
    src.latencyMs = Math.round(latency); src.preds = predictions; src.message = `Model connected · ${engine.modelVersion} · ${message}`;
    const live = new Set();
    for (const p of predictions) {
      live.add(p.track_id);
      const prev = src.tweens.get(p.track_id);
      src.tweens.set(p.track_id, { from: prev ? boxAt(prev, wall) : p.bbox, to: p.bbox, t0: wall, t1: wall + gap, label: p.label,
        tag: `${p.track_id} · ${labelText(p)}${p.label === 'unknown' ? '' : ' ' + Math.round(p.confidence * 100) + '%'}` });
    }
    for (const id of [...src.tweens.keys()]) if (!live.has(id)) src.tweens.delete(id);
    // Video files: slow playback so every 0.2 s of video is still analysed on a slow device.
    if (src.kind === 'video') src.video.playbackRate = Math.min(1, Math.max(0.2, 200 / gap));
    if (created.length) emit('incident', created);
    emit('update', src.cameraId);
  } catch (e) {
    console.error(e);
    src.message = 'Inference error: ' + e.message; emit('update', src.cameraId);
  }
  busy = false;
}

function loop() {
  if (!engine.sources.size) { rafId = 0; return; }
  rafId = requestAnimationFrame(loop);
  const wall = performance.now(), gap = interval();
  let next = null;
  for (const src of engine.sources.values()) {
    const v = src.video;
    if (src.kind === 'video' && v.ended) { stop(src.cameraId, 'Video finished. Recorded incidents are available in Event history.'); continue; }
    if (v.paused && !v.ended) v.play().catch(() => {});       // moving the element between pages pauses it
    if (!v.videoWidth) continue;
    if (src.lastWall && wall - src.lastWall > 3000) newTracker(src);   // ai.runtime: >3 s gap resets the tracker
    if (src.lastWall && wall - src.lastWall < gap) continue;
    if (!next || src.lastWall < next.lastWall) next = src;    // the source that has waited longest goes first
  }
  if (next && !busy) { next.lastWall = wall; analyse(next, mediaTime(next)); }
}

function newTracker(src) {
  const counter = idCounters.get(src.cameraId) || { n: 0 };
  idCounters.set(src.cameraId, counter);
  src.tracker = new PersonTracker(() => `P${++counter.n}`);
  src.tweens.clear();
}

// ------------------------------------------------------------------ overlay
/** Draw a source's animated boxes onto a canvas laid over its (object-fit: contain) video. */
export function drawOverlay(canvas, cameraId) {
  const src = engine.sources.get(Number(cameraId)), box = canvas.getBoundingClientRect();
  if (!box.width) return;
  if (canvas.width !== Math.round(box.width) || canvas.height !== Math.round(box.height)) { canvas.width = Math.round(box.width); canvas.height = Math.round(box.height); }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const v = src?.video;
  if (!v || !v.videoWidth || !src.tweens.size) return;
  const scale = Math.min(box.width / v.videoWidth, box.height / v.videoHeight), w = v.videoWidth * scale, h = v.videoHeight * scale;
  const x = (box.width - w) / 2, y = (box.height - h) / 2, now = performance.now(), small = box.width < 480, fs = small ? 11 : 13, th = small ? 17 : 20;
  for (const tw of src.tweens.values()) {
    const [x1, y1, x2, y2] = boxAt(tw, now), bx = x + x1 * w, by = y + y1 * h, color = COLORS[tw.label] || COLORS.unknown;
    ctx.strokeStyle = color; ctx.lineWidth = tw.label === 'fall' ? 3 : 2; ctx.strokeRect(bx, by, (x2 - x1) * w, (y2 - y1) * h);
    ctx.font = `bold ${fs}px "Segoe UI", Arial, sans-serif`;
    const tw2 = ctx.measureText(tw.tag).width + 10, ty = by >= th ? by - th : by;
    ctx.fillStyle = color; ctx.fillRect(bx, ty, tw2, th); ctx.fillStyle = '#111318'; ctx.fillText(tw.tag, bx + 5, ty + th - 6);
  }
}

// ------------------------------------------------------------------ control
/** Start a source for a camera: {kind: 'webcam'} or {kind: 'video', file}. Other sources keep running. */
export async function start(cameraId, { kind, file } = {}) {
  const cam = store.camera(cameraId);
  if (!cam) throw new Error('Camera not found');
  if (!cam.enabled) throw new Error('Source disabled. Enable it in camera settings.');
  if (engine.sources.has(cam.id)) stop(cam.id);
  await loadModels();
  const video = document.createElement('video');
  video.playsInline = true; video.muted = true; video.autoplay = true;
  const src = { cameraId: cam.id, kind, name: kind === 'video' ? (file?.name || 'video') : 'Webcam', video, stream: null, objectUrl: null,
    tracker: null, tweens: new Map(), preds: [], message: 'Connecting to the model…', latencyMs: 0, fps: 0, arrivals: [], lastWall: 0 };
  if (kind === 'webcam') {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Webcam access requires HTTPS or localhost.');
    src.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, audio: false });
    video.srcObject = src.stream;
  } else {
    src.objectUrl = URL.createObjectURL(file);
    video.src = src.objectUrl;
  }
  await video.play();
  newTracker(src);
  engine.sources.set(cam.id, src);
  emit('update', cam.id); emit('state');
  if (!rafId) rafId = requestAnimationFrame(loop);
}

export function stop(cameraId, message) {
  const src = engine.sources.get(Number(cameraId));
  if (!src) return;
  engine.sources.delete(src.cameraId);
  if (src.stream) src.stream.getTracks().forEach(t => t.stop());
  src.video.pause(); src.video.srcObject = null; src.video.removeAttribute('src'); src.video.load(); src.video.remove();
  if (src.objectUrl) URL.revokeObjectURL(src.objectUrl);
  engine.message = message || 'Source stopped.';
  emit('update', src.cameraId); emit('state');
}

export function stopAll() { for (const id of [...engine.sources.keys()]) stop(id); }

window.addEventListener('beforeunload', stopAll);
