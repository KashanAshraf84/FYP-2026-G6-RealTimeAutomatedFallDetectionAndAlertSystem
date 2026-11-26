/*
 * GuardianAI FYP-2 - detection pipeline, browser port.
 *
 * Line-by-line JavaScript port of the Python modules the Django app uses:
 *   ai/pose_features.py  -> decodePoses, resampleWindow, windowFeatures
 *   ai/tracking.py       -> PersonTracker (IDs P1, P2, ...)
 *   ai/fallnet.py        -> PoseHistory
 *   ai/adapters.py       -> Detector.step (one prediction per person)
 *   live_demo.py         -> AlertRule (fall >= 0.80 on 3 consecutive frames)
 * FallNet v2 runs as ONNX with normalisation + temperature baked in, so the
 * probabilities here are the calibrated ones the evaluation reported.
 * deploy/static_space/test_parity.mjs checks this file against Python.
 * No DOM access: runs in the browser and in Node.
 */

export const LABELS = ['normal', 'warning', 'fall'];
export const P = {
  windowS: 2.0, seqLen: 30, minHistoryS: 1.5, detConf: 0.25, imgsz: 640, nmsIou: 0.7,
  visibleConf: 0.3, velocityClip: 8.0,
  // tracker
  highConf: 0.5, newTrackConf: 0.5, newTrackKptConf: 0.5, confirmHits: 3, confirmMinS: 0.3,
  confirmWindowS: 1.5, tentativeMaxGapS: 0.5, matchMin: 0.25, maxAgeS: 2.0, duplicateOverlap: 0.6,
  centreGate: 0.75, recoverGate: 1.2, recoverMaxS: 1.5, recoverKptConf: 0.3,
  holdS: 1.0, keepS: 3.0, maxPeople: 30,
};
const HEAD = [0, 1, 2, 3, 4], SHOULDERS = [5, 6], HIPS = [11, 12];
export const SKELETON = [[0, 5], [0, 6], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];

// ------------------------------------------------------------ YOLO decoding
/** Letterbox geometry for a srcW x srcH frame into the square model input (Ultralytics LetterBox). */
export function letterbox(srcW, srcH, size = P.imgsz) {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale), newH = Math.round(srcH * scale);
  const padX = Math.round((size - newW) / 2 - 0.1), padY = Math.round((size - newH) / 2 - 0.1);
  return { scale, newW, newH, padX, padY, srcW, srcH, size };
}

/**
 * Decode the YOLOv8-pose output (1 x 56 x N, channel-major) into every person:
 * confidence >= 0.25, NMS IoU 0.7, coordinates mapped back to source pixels.
 * Returns { kps: [[x,y,c] x17][], boxes: [x1,y1,x2,y2,conf][] }.
 */
export function decodePoses(data, N, lb) {
  const cand = [];
  for (let i = 0; i < N; i++) {
    const c = data[4 * N + i];
    if (c > P.detConf) cand.push(i);
  }
  cand.sort((a, b) => data[4 * N + b] - data[4 * N + a]);
  const kept = [];
  const box = i => {
    const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  };
  for (const i of cand) {
    const b = box(i);
    if (kept.every(j => iou(b, box(j)) <= P.nmsIou)) kept.push(i);
    if (kept.length >= 300) break;
  }
  const clampX = v => Math.min(Math.max(v, 0), lb.srcW), clampY = v => Math.min(Math.max(v, 0), lb.srcH);
  const kps = [], boxes = [];
  for (const i of kept) {
    const [x1, y1, x2, y2] = box(i);
    boxes.push([clampX((x1 - lb.padX) / lb.scale), clampY((y1 - lb.padY) / lb.scale),
      clampX((x2 - lb.padX) / lb.scale), clampY((y2 - lb.padY) / lb.scale), data[4 * N + i]]);
    const k = [];
    for (let j = 0; j < 17; j++) {
      const base = (5 + j * 3) * N + i;
      k.push([clampX((data[base] - lb.padX) / lb.scale), clampY((data[base + N] - lb.padY) / lb.scale),
        data[base + 2 * N]]);
    }
    kps.push(k);
  }
  return { kps, boxes };
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
}

// ------------------------------------------------------- features (pose_features.py)
const toHeightUnits = (kp, H) => kp.map(([x, y, c]) => [x / H, y / H, c]);

/** resample_window: poses (n x 17 x 3) at `times` -> seqLen steps ending at tEnd. */
export function resampleWindow(times, poses, tEnd, windowS = P.windowS, steps = P.seqLen) {
  const n = times.length;
  if (n === 1) return Array.from({ length: steps }, () => poses[0]);
  const out = [];
  for (let k = 1; k <= steps; k++) {
    const g = tEnd - windowS + windowS * (k / steps);
    let i1 = 0;
    while (i1 < n && times[i1] <= g) i1++;           // searchsorted side="right"
    i1 = Math.min(Math.max(i1, 1), n - 1);
    const i0 = i1 - 1, t0 = times[i0], t1 = times[i1];
    const w = Math.min(Math.max(t1 > t0 ? (g - t0) / Math.max(t1 - t0, 1e-9) : 0, 0), 1);
    const p0 = poses[i0], p1 = poses[i1];
    out.push(p0.map((a, j) => [a[0] * (1 - w) + p1[j][0] * w, a[1] * (1 - w) + p1[j][1] * w,
      a[2] * (1 - w) + p1[j][2] * w]));
  }
  return out;
}

function weightedPoint(frame, idx) {
  let sx = 0, sy = 0, sw = 0;
  for (const j of idx) { const w = frame[j][2] + 1e-3; sx += frame[j][0] * w; sy += frame[j][1] * w; sw += w; }
  return [sx / sw, sy / sw];
}

function gradient(v) {
  const n = v.length, g = new Array(n);
  g[0] = v[1] - v[0]; g[n - 1] = v[n - 1] - v[n - 2];
  for (let i = 1; i < n - 1; i++) g[i] = (v[i + 1] - v[i - 1]) / 2;
  return g;
}

function median(a) {
  const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** window_features: seqLen x 17 x 3 window -> Float32Array(seqLen * 62). */
export function windowFeatures(win) {
  const T = win.length;
  const hip = win.map(f => weightedPoint(f, HIPS)), sho = win.map(f => weightedPoint(f, SHOULDERS));
  const head = win.map(f => weightedPoint(f, HEAD));
  const bw = [], bh = [];
  for (const f of win) {
    const vis = f.filter(p => p[2] >= P.visibleConf);
    const pts = vis.length >= 3 ? vis : f;
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    bw.push(Math.max(...xs) - Math.min(...xs)); bh.push(Math.max(...ys) - Math.min(...ys));
  }
  const scale = Math.max(median(bw.map((w, i) => Math.hypot(w, bh[i]))), 0.05);
  const dt = P.windowS / T, clip = v => Math.min(Math.max(v, -P.velocityClip), P.velocityClip);
  const hvx = gradient(hip.map(p => p[0])), hvy = gradient(hip.map(p => p[1])), dvy = gradient(head.map(p => p[1]));
  const out = new Float32Array(T * 62);
  for (let t = 0; t < T; t++) {
    const f = win[t], o = t * 62;
    for (let j = 0; j < 17; j++) {
      out[o + 2 * j] = (f[j][0] - hip[t][0]) / scale;
      out[o + 2 * j + 1] = (f[j][1] - hip[t][1]) / scale;
    }
    out[o + 34] = (hip[t][0] - hip[0][0]) / scale; out[o + 35] = (hip[t][1] - hip[0][1]) / scale;
    out[o + 36] = (head[t][0] - head[0][0]) / scale; out[o + 37] = (head[t][1] - head[0][1]) / scale;
    out[o + 38] = clip(hvx[t] / dt / scale); out[o + 39] = clip(hvy[t] / dt / scale);
    out[o + 40] = clip(dvy[t] / dt / scale);
    const angle = Math.atan2(sho[t][0] - hip[t][0], -(sho[t][1] - hip[t][1]));
    out[o + 41] = Math.sin(angle); out[o + 42] = Math.cos(angle);
    out[o + 43] = Math.log((bw[t] + 1e-3) / (bh[t] + 1e-3)); out[o + 44] = bh[t] / scale;
    for (let j = 0; j < 17; j++) out[o + 45 + j] = f[j][2];
  }
  return out;
}

/**
 * pose_features.partial_view: upper-body view (person cut off by the bottom of the frame, knees
 * and ankles not visible, head and shoulders visible). Reported as "partial view", never as a fall.
 */
export function partialView(win, boxBottom, H) {
  const med = idx => median(win.map(f => idx.reduce((s, j) => s + f[j][2], 0) / idx.length));
  return boxBottom >= 0.97 * H && med([13, 14, 15, 16]) < 0.2 && med([0, 1, 2, 3, 4, 5, 6]) >= 0.3;
}

/** window_evidence: torso tilt and hip motion of the newest frame. */
export function windowEvidence(win) {
  const hip = win.map(f => weightedPoint(f, HIPS)), last = win[win.length - 1];
  const sho = weightedPoint(last, SHOULDERS), h = hip[hip.length - 1];
  const tilt = Math.atan2(Math.abs(sho[0] - h[0]), -(sho[1] - h[1])) * 180 / Math.PI;
  const g = gradient(hip.map(p => p[1]));
  return { tiltDeg: tilt, hipVy: g[g.length - 1] / (P.windowS / win.length) };
}

// ------------------------------------------------------------ PoseHistory
export class PoseHistory {
  constructor() { this.times = []; this.poses = []; }
  add(t, pose) {
    this.times.push(t); this.poses.push(pose);
    while (this.times.length && this.times[0] < t - P.keepS) { this.times.shift(); this.poses.shift(); }
  }
  window(now) {
    const n = this.times.length;
    if (!n || now - this.times[n - 1] > P.holdS) return null;
    if (this.times[n - 1] - this.times[0] < P.minHistoryS) return null;
    return resampleWindow(this.times, this.poses, this.times[n - 1]);
  }
}

// --------------------------------------------------------- tracker (tracking.py)
const centre = b => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
const diag = b => Math.hypot(b[2] - b[0], b[3] - b[1]);
const area = b => (b[2] - b[0]) * (b[3] - b[1]);
const inter = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
  Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));

function similarity(tb, db) {
  return tb.map(a => db.map(b => {
    const i = inter(a, b), u = i / (area(a) + area(b) - i + 1e-9);
    const [ax, ay] = centre(a), [bx, by] = centre(b);
    const prox = Math.min(Math.max(1 - Math.hypot(ax - bx, ay - by) / (P.centreGate * (Math.max(diag(a), diag(b)) + 1e-9)), 0), 1);
    return 0.5 * u + 0.5 * prox;
  }));
}

function trackSimilarity(tracks, db, t) {
  const a = similarity(tracks.map(tr => tr.predictedBox(t)), db), b = similarity(tracks.map(tr => tr.box), db);
  return a.map((row, i) => row.map((v, j) => Math.max(v, b[i][j])));
}

function recoverySimilarity(tracks, db, t) {
  return tracks.map(tr => db.map(b => {
    if (t - tr.lastT > P.recoverMaxS) return 0;
    const [ax, ay] = centre(tr.box), [bx, by] = centre(b);
    return Math.min(Math.max(1 - Math.hypot(ax - bx, ay - by) / (P.recoverGate * (Math.max(diag(tr.box), diag(b)) + 1e-9)), 0), 1);
  }));
}

/** Maximum-weight assignment (scipy.optimize.linear_sum_assignment on -sim). Returns [[row, col], ...]. */
export function hungarian(sim) {
  const R = sim.length, C = R ? sim[0].length : 0;
  if (!R || !C) return [];
  const transpose = R > C, n = transpose ? C : R, m = transpose ? R : C;
  const cost = (i, j) => -(transpose ? sim[j - 1][i - 1] : sim[i - 1][j - 1]);
  const u = new Array(n + 1).fill(0), v = new Array(m + 1).fill(0), p = new Array(m + 1).fill(0), way = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0;
    const minv = new Array(m + 1).fill(Infinity), used = new Array(m + 1).fill(false);
    do {
      used[j0] = true; const i0 = p[j0]; let delta = Infinity, j1 = 0;
      for (let j = 1; j <= m; j++) if (!used[j]) {
        const cur = cost(i0, j) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const pairs = [];
  for (let j = 1; j <= m; j++) if (p[j]) pairs.push(transpose ? [j - 1, p[j] - 1] : [p[j] - 1, j - 1]);
  return pairs.sort((a, b) => a[0] - b[0]);
}

class Track {
  constructor(box, t, conf) {
    this.box = box.slice(0, 4); this.firstT = t; this.lastT = t; this.trackId = null;
    this.velocity = [0, 0]; this.hits = 1; this.history = new PoseHistory(); this.lastConf = conf;
  }
  get confirmed() { return this.trackId !== null; }
  predictedBox(t) {
    const dt = Math.min(Math.max(t - this.lastT, 0), 0.5), sx = this.velocity[0] * dt, sy = this.velocity[1] * dt;
    return [this.box[0] + sx, this.box[1] + sy, this.box[2] + sx, this.box[3] + sy];
  }
}

const kptMean = k => k.reduce((s, p) => s + p[2], 0) / k.length;
/** tracking.well_formed: 17-point mean, or head + shoulders for a person cut off by the bottom of the frame (webcam close-up). */
const wellFormed = (k, box, H) => kptMean(k) >= P.newTrackKptConf
  || (box[3] >= 0.97 * H && kptMean(k.slice(0, 7)) >= P.newTrackKptConf);

export class PersonTracker {
  constructor(nextId) {
    let n = 0;
    this.nextId = nextId || (() => `P${++n}`);
    this.tracks = [];
  }

  update(t, kps, boxes, H) {
    const n = boxes.length, assigned = new Map();
    const free = this.tracks.map((_, k) => k);
    const match = (trackIds, detIds, simFn, minSim, confidentStage) => {
      if (!trackIds.length || !detIds.length) return;
      const sim = simFn(trackIds.map(k => this.tracks[k]), detIds.map(d => boxes[d]), t);
      for (const [r, c] of hungarian(sim)) {
        if (sim[r][c] >= minSim) {
          const k = trackIds[r], d = detIds[c];
          const confident = confidentStage && wellFormed(kps[d], boxes[d], H);
          this.apply(this.tracks[k], t, kps[d], boxes[d], H, confident);
          assigned.set(d, this.tracks[k]);
          free.splice(free.indexOf(k), 1);
        }
      }
    };
    const confirmed = () => free.filter(k => this.tracks[k].confirmed);
    const candidates = () => free.filter(k => !this.tracks[k].confirmed);
    const left = dets => dets.filter(d => !assigned.has(d));
    const all = [...Array(n).keys()];
    const high = all.filter(d => boxes[d][4] >= P.highConf), low = all.filter(d => boxes[d][4] < P.highConf);

    match(confirmed(), left(high), trackSimilarity, P.matchMin, true);
    match(candidates(), left(high), trackSimilarity, P.matchMin, true);
    match(confirmed(), left(low), trackSimilarity, P.matchMin, false);
    match(confirmed(), left(all).filter(d => kptMean(kps[d]) >= P.recoverKptConf), recoverySimilarity, 1e-6, false);

    for (const d of left(high)) {
      if (boxes[d][4] < P.newTrackConf || !wellFormed(kps[d], boxes[d], H)) continue;
      const b = boxes[d];
      const occupied = this.tracks.filter(tr => t - tr.lastT <= P.tentativeMaxGapS);
      if (occupied.some(tr => inter(b, tr.box) / (area(b) + 1e-9) > P.duplicateOverlap)) continue;
      const track = new Track(b, t, b[4]);
      track.history.add(t, toHeightUnits(kps[d], H));
      this.tracks.push(track);
      assigned.set(d, track);
    }
    this.tracks = this.tracks.filter(tr => tr.confirmed
      ? t - tr.lastT <= P.maxAgeS
      : t - tr.lastT <= P.tentativeMaxGapS && t - tr.firstT <= P.confirmWindowS);
    return assigned;
  }

  apply(track, t, kp, box, H, confident) {
    const dt = t - track.lastT;
    if (dt > 1e-3) {
      const [ox, oy] = centre(track.box), [nx, ny] = centre(box), d = Math.max(dt, 0.05);
      track.velocity = [0.6 * track.velocity[0] + 0.4 * (nx - ox) / d, 0.6 * track.velocity[1] + 0.4 * (ny - oy) / d];
    }
    track.box = box.slice(0, 4); track.lastT = t; track.lastConf = box[4];
    track.history.add(t, toHeightUnits(kp, H));
    if (!track.confirmed && confident) {
      track.hits += 1;
      if (track.hits >= P.confirmHits && t - track.firstT >= P.confirmMinS) track.trackId = this.nextId();
    }
  }

  active(t, holdS = P.holdS) {
    return this.tracks.filter(tr => tr.confirmed && t - tr.lastT <= holdS).sort((a, b) => a.firstT - b.firstT);
  }
}

// ------------------------------------------------------------ detector step
/**
 * One frame: detections -> per-person predictions (ai.adapters.TeamDetector.predict).
 * `runModel(Float32Array, batch)` must return probabilities (batch x 3).
 */
export async function detectorStep(tracker, t, dets, H, runModel) {
  tracker.update(t, dets.kps, dets.boxes, H);
  const people = tracker.active(t).slice(0, P.maxPeople);
  const windows = people.map(p => p.history.window(t));
  const partial = windows.map((w, i) => !!w && partialView(w, people[i].box[3], H));
  const ready = windows.map((w, i) => (w && !partial[i] ? i : -1)).filter(i => i >= 0);
  const probs = {};
  if (ready.length) {
    const batch = new Float32Array(ready.length * P.seqLen * 62);
    ready.forEach((i, b) => batch.set(windowFeatures(windows[i]), b * P.seqLen * 62));
    const out = await runModel(batch, ready.length);
    ready.forEach((i, b) => { probs[i] = Array.from(out.slice(b * 3, b * 3 + 3)); });
  }
  return people.map((p, i) => {
    if (!probs[i]) return { trackId: p.trackId, label: 'unknown', confidence: 0, box: p.box, probs: null, partial: partial[i] };
    const pr = probs[i];
    const best = pr.indexOf(Math.max(...pr));
    return { trackId: p.trackId, label: LABELS[best], confidence: pr[best], box: p.box, probs: pr,
      evidence: windowEvidence(windows[i]) };
  });
}

// ------------------------------------------------------------ alert rule
export class AlertRule {
  constructor(confidence = 0.8, frames = 3, cooldownS = 30) {
    Object.assign(this, { confidence, frames, cooldownS }); this.streak = {}; this.lastAlert = {};
  }
  update(preds, t) {
    const fired = [], present = new Set(preds.map(p => p.trackId));
    for (const id of Object.keys(this.streak)) if (!present.has(id)) this.streak[id] = 0;
    for (const p of preds) {
      const hit = p.label === 'fall' && p.confidence >= this.confidence;
      this.streak[p.trackId] = hit ? (this.streak[p.trackId] || 0) + 1 : 0;
      const last = this.lastAlert[p.trackId];
      if (this.streak[p.trackId] >= this.frames && (last === undefined || t - last >= this.cooldownS)) {
        this.lastAlert[p.trackId] = t; fired.push(p.trackId);
      }
    }
    return fired;
  }
}
