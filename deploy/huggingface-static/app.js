/*
 * GuardianAI - Browser demo controller.
 *
 * Owns camera capture, ONNX inference, rendering and alerting. All detection
 * logic lives in pipeline.js, which is a direct port of the Python modules.
 *
 * Nothing is uploaded: frames are captured, analysed and discarded entirely
 * within this page. There is no server-side component.
 */

import {
  CONFIG, SKELETON, FeatureExtractor,
  fusePredictions, smoothPredictions,
  decodePose, toCriticalNormalised, softmaxArgmax,
} from './pipeline.js';

const MODEL_SIZE = 480;   // YOLO ONNX was exported at 480x480
const CAP_W = 640, CAP_H = 480;

const el = id => document.getElementById(id);

const ui = {
  video: el('camera'), work: el('work'), view: el('view'),
  placeholder: el('placeholder'), loader: el('loader'), loaderText: el('loader-text'),
  liveBadge: el('live-badge'), startBtn: el('start-btn'), startLabel: el('start-label'),
  startIcon: el('start-icon'), muteBtn: el('mute-btn'), muteIcon: el('mute-icon'),
  muteLabel: el('mute-label'), errorBar: el('error-bar'),
  statusText: el('status-text'), statusCard: el('main-status-card'),
  confFill: el('confidence-fill'), confVal: el('confidence-val'),
  personCount: el('person-count'), bodyAngle: el('body-angle'),
  dropSpeed: el('drop-speed'), totalFalls: el('total-falls'),
  eventLog: el('event-log'), fallOverlay: el('fall-overlay'),
  engineStatus: el('engine-status'), cameraStatus: el('camera-status'),
  fpsStatus: el('fps-status'), backendStatus: el('backend-status'),
  notifyStatus: el('notify-status'),
};

let poseSession = null, lstmSession = null;
let extractor = new FeatureExtractor();
let history = [];
let running = false, muted = false, stream = null;
let lostFrames = 0, lastKeypoints = null;
let lastStatus = 'normal', fallCount = 0, lastAlertTime = 0;
let frameTimes = [];
let loopScheduled = false;

// ---------------------------------------------------------------- helpers

function setError(msg) {
  ui.errorBar.textContent = msg || '';
  ui.errorBar.style.display = msg ? 'block' : 'none';
}

function updateClock() {
  el('current-time').textContent =
    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addLogItem(status, confidence) {
  const ph = ui.eventLog.querySelector('.log-placeholder');
  if (ph) ph.remove();
  const item = document.createElement('div');
  item.className = `log-item ${status}`;
  const t = new Date().toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  item.innerHTML = `<span class="log-status">${status.toUpperCase()}</span>` +
                   `<span class="log-time">${t} · ${(confidence * 100).toFixed(1)}%</span>`;
  ui.eventLog.prepend(item);
  const items = ui.eventLog.querySelectorAll('.log-item');
  if (items.length > 20) items[items.length - 1].remove();
}

/*
 * Native OS notification, the browser equivalent of
 * alert_system._desktop_notification() (plyer) in the local application.
 *
 * This is what makes an alert visible when the operator has switched to
 * another application or the dashboard tab is not in front.
 */
async function requestNotificationPermission() {
  if (!('Notification' in window)) {
    ui.notifyStatus.textContent = 'Unsupported';
    ui.notifyStatus.className = 'value offline';
    return;
  }
  let perm = Notification.permission;
  if (perm === 'default') {
    // Must be triggered by a user gesture; the Start Camera click qualifies.
    try { perm = await Notification.requestPermission(); } catch { /* ignore */ }
  }
  const map = {
    granted: ['Enabled', 'value online'],
    denied: ['Blocked', 'value offline'],
    default: ['Not allowed', 'value'],
  };
  const [text, cls] = map[perm] || map.default;
  ui.notifyStatus.textContent = text;
  ui.notifyStatus.className = cls;
}

function showNotification(status, confidence) {
  if (muted) return;
  if (!('Notification' in window) || Notification.permission !== 'granted') return;
  try {
    const n = new Notification(
      status === 'fall' ? '🚨 Fall Detected' : '⚠️ Warning',
      {
        body: `Status: ${status.toUpperCase()} (${(confidence * 100).toFixed(1)}% confidence)\n`
            + `Please check on the individual immediately.`,
        icon: './icon.png',
        badge: './favicon.png',
        // `tag` collapses repeats into one entry rather than stacking them.
        tag: 'guardianai-alert',
        renotify: true,
      });
    n.onclick = () => { window.focus(); n.close(); };
    // Auto-dismiss, mirroring the 3-second popup in the local application.
    setTimeout(() => n.close(), 8000);
  } catch (e) {
    console.warn('Notification failed:', e);
  }
}

// Reproduces the 1000/1500/1000 Hz pattern of alert_system._sound_alert().
function playAlarm() {
  if (muted) return;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    [[1000, 0], [1500, 0.22], [1000, 0.44]].forEach(([f, off]) => {
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = 'square'; osc.frequency.value = f;
      gain.gain.setValueAtTime(0.18, ctx.currentTime + off);
      gain.gain.setValueAtTime(0, ctx.currentTime + off + 0.2);
      osc.start(ctx.currentTime + off);
      osc.stop(ctx.currentTime + off + 0.2);
    });
  } catch (e) { console.warn('Alarm tone unavailable', e); }
}

// ---------------------------------------------------------------- models

async function loadModels() {
  ui.loader.style.display = 'flex';
  ui.placeholder.style.display = 'none';

  ort.env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
  ort.env.wasm.simd = true;

  // WebGPU is dramatically faster where available; fall back to WASM.
  const providers = ('gpu' in navigator) ? ['webgpu', 'wasm'] : ['wasm'];

  ui.loaderText.textContent = 'Loading pose model (12.7 MB)…';
  poseSession = await ort.InferenceSession.create(
    './models/yolov8n-pose-480.onnx', { executionProviders: providers });

  ui.loaderText.textContent = 'Loading CNN-LSTM classifier (2.9 MB)…';
  try {
    lstmSession = await ort.InferenceSession.create(
      './models/fall_detector.onnx', { executionProviders: ['wasm'] });
  } catch (e) {
    // Mirrors the Python fallback: run on rules alone if the model is absent.
    console.warn('CNN-LSTM unavailable, continuing rule-only:', e);
    lstmSession = null;
  }

  const backend = poseSession.handler?._backendName ||
                  (providers[0] === 'webgpu' ? 'webgpu' : 'wasm');
  ui.backendStatus.textContent = backend.toUpperCase();
  ui.backendStatus.className = 'value active';
  ui.engineStatus.textContent = lstmSession ? 'Hybrid' : 'Rules only';
  ui.engineStatus.className = 'value online';
  ui.loader.style.display = 'none';
}

// ------------------------------------------------------------- inference

/** Letterbox the capture canvas into a square model input, preserving aspect. */
function preprocess(sourceCanvas) {
  const scale = Math.min(MODEL_SIZE / CAP_W, MODEL_SIZE / CAP_H);
  const newW = Math.round(CAP_W * scale), newH = Math.round(CAP_H * scale);
  const padX = (MODEL_SIZE - newW) / 2, padY = (MODEL_SIZE - newH) / 2;

  const c = document.createElement('canvas');
  c.width = MODEL_SIZE; c.height = MODEL_SIZE;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#727272';
  ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  ctx.drawImage(sourceCanvas, padX, padY, newW, newH);

  const { data } = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const area = MODEL_SIZE * MODEL_SIZE;
  const tensor = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {          // RGBA -> planar RGB, scaled to [0,1]
    tensor[i] = data[i * 4] / 255;
    tensor[area + i] = data[i * 4 + 1] / 255;
    tensor[2 * area + i] = data[i * 4 + 2] / 255;
  }
  return {
    tensor,
    letterbox: { scale, padX, padY, srcW: CAP_W, srcH: CAP_H },
  };
}

function drawFrame(sourceCanvas, keypoints, status, confidence) {
  const ctx = ui.view.getContext('2d');
  ctx.drawImage(sourceCanvas, 0, 0, CAP_W, CAP_H);

  if (keypoints) {
    // Skeleton, drawn as a dark outer stroke beneath a light inner line so it
    // stays legible against both bright and dark backgrounds.
    for (const [a, b] of SKELETON) {
      if (keypoints[a][2] > CONFIG.jointDrawConfidence &&
          keypoints[b][2] > CONFIG.jointDrawConfidence) {
        ctx.beginPath();
        ctx.moveTo(keypoints[a][0], keypoints[a][1]);
        ctx.lineTo(keypoints[b][0], keypoints[b][1]);
        ctx.strokeStyle = 'rgba(40,40,40,0.9)'; ctx.lineWidth = 4; ctx.stroke();
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1; ctx.stroke();
      }
    }
    for (const [x, y, c] of keypoints) {
      if (c > CONFIG.jointDrawConfidence) {
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#4e7cfe'; ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff'; ctx.fill();
      }
    }
  }

  // HUD status box
  const colours = { normal: '#00c800', warning: '#00c8ff', fall: '#ff2020' };
  ctx.fillStyle = 'rgba(0,0,0,0.85)';
  ctx.fillRect(10, 10, 290, 60);
  ctx.font = 'bold 20px Consolas, monospace';
  ctx.fillStyle = colours[status] || '#ffffff';
  ctx.fillText(`STATUS: ${status.toUpperCase()}`, 20, 38);
  ctx.font = '14px Consolas, monospace';
  ctx.fillStyle = '#ffffff';
  ctx.fillText(`CONF: ${(confidence * 100).toFixed(1)}%`, 20, 60);

  // Telemetry panel
  const px = CAP_W - 210;
  ctx.fillStyle = 'rgba(20,20,20,0.88)';
  ctx.fillRect(px, 80, 200, 160);
  ctx.strokeStyle = 'rgba(100,100,100,0.9)'; ctx.lineWidth = 1;
  ctx.strokeRect(px, 80, 200, 160);
  ctx.fillStyle = '#c8c8c8'; ctx.font = '13px Consolas, monospace';
  ctx.fillText('TELEMETRY', px + 55, 105);
  ctx.fillStyle = '#ffffff'; ctx.font = '13px Consolas, monospace';
  ctx.fillText(`ANGLE: ${extractor.lastAngle.toFixed(1)} deg`, px + 15, 145);
  ctx.fillText(`SPEED: ${extractor.lastSpeed.toFixed(1)} px/f`, px + 15, 175);

  const stability = Math.max(0, Math.min(1, 1 - Math.abs(extractor.lastSpeed) / 50));
  ctx.fillStyle = '#282828'; ctx.fillRect(px + 15, 205, 170, 10);
  ctx.fillStyle = '#ff9600'; ctx.fillRect(px + 15, 205, 170 * stability, 10);
  ctx.fillStyle = '#969696'; ctx.font = '10px Consolas, monospace';
  ctx.fillText('STABILITY', px + 15, 230);
}

function updatePanels(status, confidence, personCount) {
  ui.statusText.textContent = status.toUpperCase();
  ui.confVal.textContent = `${(confidence * 100).toFixed(1)}%`;
  ui.confFill.style.width = `${Math.min(confidence * 100, 100)}%`;
  ui.statusCard.classList.remove('warning', 'danger');
  if (status === 'warning') ui.statusCard.classList.add('warning');
  if (status === 'fall') ui.statusCard.classList.add('danger');
  ui.fallOverlay.style.display = status === 'fall' ? 'block' : 'none';

  ui.personCount.textContent = personCount;
  ui.bodyAngle.textContent = `${extractor.lastAngle.toFixed(1)}°`;
  ui.dropSpeed.textContent = extractor.lastSpeed.toFixed(1);
  ui.totalFalls.textContent = fallCount;
}

async function processFrame() {
  if (!running) return;

  if (!ui.video.videoWidth) { setTimeout(processFrame, 100); return; }

  const work = ui.work;
  work.width = CAP_W; work.height = CAP_H;
  work.getContext('2d').drawImage(ui.video, 0, 0, CAP_W, CAP_H);

  try {
    const { tensor, letterbox } = preprocess(work);
    const feeds = {};
    feeds[poseSession.inputNames[0]] =
      new ort.Tensor('float32', tensor, [1, 3, MODEL_SIZE, MODEL_SIZE]);
    const out = await poseSession.run(feeds);
    const raw = out[poseSession.outputNames[0]];
    const numAnchors = raw.dims[2];

    const detection = decodePose(raw.data, numAnchors, letterbox);

    let keypoints = null, critical = null, personCount = 0;
    if (detection) {
      keypoints = detection.keypoints;
      critical = toCriticalNormalised(keypoints, CAP_W, CAP_H);
      lastKeypoints = keypoints;
      lostFrames = 0;
      personCount = 1;
    } else if (lastKeypoints && lostFrames < CONFIG.maxLostFrames) {
      // Occlusion tolerance, matching PoseEstimator._handle_lost_tracking()
      lostFrames++;
      keypoints = lastKeypoints;
      critical = toCriticalNormalised(keypoints, CAP_W, CAP_H);
      personCount = 1;
    } else {
      lastKeypoints = null;
    }

    let status = 'normal', confidence = 0.0;

    if (critical) {
      const sequence = extractor.update(critical);
      const [ruleStatus, ruleConf] = extractor.getRuleBasedStatus(critical);

      let nnStatus = 'normal', nnConf = 0.0;
      if (sequence && lstmSession) {
        const lf = {};
        lf[lstmSession.inputNames[0]] = new ort.Tensor('float32', sequence, [1, 30, 51]);
        const lo = await lstmSession.run(lf);
        [nnStatus, nnConf] = softmaxArgmax(
          Array.from(lo[lstmSession.outputNames[0]].data));
      }

      [status, confidence] = fusePredictions(ruleStatus, ruleConf, nnStatus, nnConf);
      [status, confidence] = smoothPredictions(history, status, confidence);

      // Alert gate + cooldown, matching inference.py and AlertSystem
      const now = Date.now() / 1000;
      if (status === 'fall' && confidence > CONFIG.confidenceThreshold &&
          now - lastAlertTime >= CONFIG.alertCooldownSeconds) {
        lastAlertTime = now;
        fallCount++;
        playAlarm();
        showNotification(status, confidence);
      }
    } else {
      extractor.update(null);
    }

    drawFrame(work, keypoints, status, confidence);
    updatePanels(status, confidence, personCount);

    if (status !== 'normal' && lastStatus === 'normal') addLogItem(status, confidence);
    lastStatus = status;

    const now = performance.now();
    frameTimes.push(now);
    frameTimes = frameTimes.filter(t => now - t < 4000);
    if (frameTimes.length > 1) {
      const span = (frameTimes[frameTimes.length - 1] - frameTimes[0]) / 1000;
      ui.fpsStatus.textContent = `${(frameTimes.length / span).toFixed(1)} fps`;
      ui.fpsStatus.className = 'value active';
    }
    setError('');
  } catch (err) {
    console.error(err);
    setError('Inference error: ' + err.message);
  }

  scheduleNext();
}

/*
 * Keep the detection loop alive when the tab is not in front.
 *
 * requestAnimationFrame is suspended outright while a document is hidden, so
 * relying on it alone would freeze detection the moment the operator switches
 * to another application - precisely when the OS notification matters most.
 * When hidden we fall back to a timer instead. Browsers still throttle
 * background timers, so throughput drops, but detection continues.
 */
function scheduleNext() {
  if (!running || loopScheduled) return;
  loopScheduled = true;
  if (document.hidden) {
    setTimeout(runOnce, 120);
  } else {
    requestAnimationFrame(runOnce);
  }
}

function runOnce() {
  loopScheduled = false;
  processFrame();
}

// A pending rAF never fires once the tab is hidden, so restart the loop on
// every visibility transition; the loopScheduled guard prevents duplicates.
document.addEventListener('visibilitychange', () => {
  if (running) scheduleNext();
});

// ------------------------------------------------------------- controls

async function start() {
  setError('');
  if (!navigator.mediaDevices?.getUserMedia) {
    setError('This browser does not support camera access. Try Chrome, Edge or Firefox.');
    return;
  }

  ui.startBtn.disabled = true;
  await requestNotificationPermission();
  try {
    if (!poseSession) await loadModels();
  } catch (e) {
    setError('Could not load the models: ' + e.message);
    ui.loader.style.display = 'none';
    ui.placeholder.style.display = 'flex';
    ui.startBtn.disabled = false;
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: CAP_W }, height: { ideal: CAP_H } }, audio: false,
    });
  } catch (err) {
    const msgs = {
      NotAllowedError: 'Camera permission denied. Allow access and try again.',
      NotFoundError: 'No camera found on this device.',
      NotReadableError: 'The camera is in use by another application.',
    };
    setError(msgs[err.name] || ('Could not open the camera: ' + err.message));
    ui.placeholder.style.display = 'flex';
    ui.startBtn.disabled = false;
    return;
  }

  ui.video.srcObject = stream;
  ui.view.width = CAP_W; ui.view.height = CAP_H;
  running = true;

  ui.placeholder.style.display = 'none';
  ui.view.style.display = 'block';
  ui.liveBadge.style.display = 'inline-block';
  ui.startLabel.textContent = 'Stop';
  ui.startIcon.textContent = '⏹';
  ui.startBtn.classList.add('paused');
  ui.startBtn.disabled = false;
  ui.cameraStatus.textContent = 'Active';
  ui.cameraStatus.className = 'value active';

  processFrame();
}

function stop() {
  running = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  extractor.reset();
  history = [];
  lastKeypoints = null; lostFrames = 0; lastStatus = 'normal';

  ui.view.style.display = 'none';
  ui.placeholder.style.display = 'flex';
  ui.liveBadge.style.display = 'none';
  ui.startLabel.textContent = 'Start Camera';
  ui.startIcon.textContent = '▶';
  ui.startBtn.classList.remove('paused');
  ui.cameraStatus.textContent = 'Stopped';
  ui.cameraStatus.className = 'value';
  ui.fpsStatus.textContent = '—';
  ui.fpsStatus.className = 'value';
  ui.statusText.textContent = 'STANDBY';
  ui.statusCard.classList.remove('warning', 'danger');
  ui.fallOverlay.style.display = 'none';
}

ui.startBtn.addEventListener('click', () => (running ? stop() : start()));
ui.muteBtn.addEventListener('click', () => {
  muted = !muted;
  ui.muteBtn.classList.toggle('paused', muted);
  ui.muteIcon.textContent = muted ? '🔕' : '🔔';
  ui.muteLabel.textContent = muted ? 'Muted' : 'Mute';
});

updateClock();
setInterval(updateClock, 30000);
window.addEventListener('beforeunload', stop);
