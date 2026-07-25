/*
 * GuardianAI - Cloud demo client.
 *
 * Captures frames from the visitor's camera, sends them to the server for
 * detection, and renders the annotated result. The next frame is only sent
 * once the previous response arrives, which gives natural backpressure: the
 * client automatically matches whatever throughput the server can sustain
 * instead of flooding it.
 */

document.addEventListener('DOMContentLoaded', () => {

  const els = {
    video: document.getElementById('camera'),
    canvas: document.getElementById('capture'),
    output: document.getElementById('output'),
    placeholder: document.getElementById('placeholder'),
    liveBadge: document.getElementById('live-badge'),
    startBtn: document.getElementById('start-btn'),
    startLabel: document.getElementById('start-label'),
    muteBtn: document.getElementById('mute-btn'),
    muteIcon: document.getElementById('mute-icon'),
    muteLabel: document.getElementById('mute-label'),
    errorBar: document.getElementById('error-bar'),
    statusText: document.getElementById('status-text'),
    statusCard: document.getElementById('main-status-card'),
    confFill: document.getElementById('confidence-fill'),
    confVal: document.getElementById('confidence-val'),
    personCount: document.getElementById('person-count'),
    bodyAngle: document.getElementById('body-angle'),
    dropSpeed: document.getElementById('drop-speed'),
    totalFalls: document.getElementById('total-falls'),
    eventLog: document.getElementById('event-log'),
    fallOverlay: document.getElementById('fall-overlay'),
    serverStatus: document.getElementById('server-status'),
    cameraStatus: document.getElementById('camera-status'),
    fpsStatus: document.getElementById('fps-status'),
    clock: document.getElementById('current-time'),
  };

  // A per-visitor identifier keeps this browser's temporal buffer separate
  // from other people viewing the demo at the same time.
  let sessionId = sessionStorage.getItem('guardianai-session');
  if (!sessionId) {
    sessionId = (crypto.randomUUID && crypto.randomUUID()) ||
                (Date.now() + '-' + Math.random().toString(16).slice(2));
    sessionStorage.setItem('guardianai-session', sessionId);
  }

  let running = false;
  let muted = false;
  let stream = null;
  let lastStatus = 'normal';
  let frameTimes = [];

  // ---------- helpers ----------

  function setError(message) {
    els.errorBar.textContent = message || '';
    els.errorBar.style.display = message ? 'block' : 'none';
  }

  function updateClock() {
    els.clock.textContent = new Date()
      .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function addLogItem(status, confidence) {
    const placeholder = els.eventLog.querySelector('.log-placeholder');
    if (placeholder) placeholder.remove();

    const item = document.createElement('div');
    item.className = `log-item ${status}`;
    const time = new Date().toLocaleTimeString([], {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    item.innerHTML =
      `<span class="log-status">${status.toUpperCase()}</span>` +
      `<span class="log-time">${time} · ${confidence.toFixed(1)}%</span>`;
    els.eventLog.prepend(item);

    const items = els.eventLog.querySelectorAll('.log-item');
    if (items.length > 20) items[items.length - 1].remove();
  }

  // Short two-tone alarm generated in the browser. The server is headless and
  // cannot make a sound, so alerting happens client-side in this deployment.
  function playAlarm() {
    if (muted) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      [[1000, 0], [1500, 0.22], [1000, 0.44]].forEach(([freq, offset]) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = freq;
        osc.type = 'square';
        gain.gain.setValueAtTime(0.18, ctx.currentTime + offset);
        gain.gain.setValueAtTime(0, ctx.currentTime + offset + 0.2);
        osc.start(ctx.currentTime + offset);
        osc.stop(ctx.currentTime + offset + 0.2);
      });
    } catch (e) {
      console.warn('Alarm tone unavailable:', e);
    }
  }

  function updateUI(data) {
    const status = data.status || 'normal';
    const confidence = data.confidence || 0;

    els.statusText.textContent = status.toUpperCase();
    els.confVal.textContent = `${confidence.toFixed(1)}%`;
    els.confFill.style.width = `${Math.min(confidence, 100)}%`;

    els.statusCard.classList.remove('warning', 'danger');
    if (status === 'warning') els.statusCard.classList.add('warning');
    if (status === 'fall') els.statusCard.classList.add('danger');

    els.fallOverlay.style.display = (status === 'fall') ? 'block' : 'none';

    els.personCount.textContent = data.person_count ?? 0;
    els.bodyAngle.textContent = `${(data.angle ?? 0).toFixed(1)}°`;
    els.dropSpeed.textContent = (data.speed ?? 0).toFixed(1);
    els.totalFalls.textContent = data.falls ?? 0;

    if (status !== 'normal' && lastStatus === 'normal') {
      addLogItem(status, confidence);
      if (status === 'fall') playAlarm();
    }
    lastStatus = status;
  }

  function updateThroughput() {
    const now = performance.now();
    frameTimes.push(now);
    frameTimes = frameTimes.filter(t => now - t < 4000);
    if (frameTimes.length > 1) {
      const span = (frameTimes[frameTimes.length - 1] - frameTimes[0]) / 1000;
      els.fpsStatus.textContent = `${(frameTimes.length / span).toFixed(1)} fps`;
      els.fpsStatus.className = 'value active';
    }
  }

  // ---------- main loop ----------

  async function sendFrame() {
    if (!running) return;

    const video = els.video;
    if (!video.videoWidth) {           // camera still warming up
      setTimeout(sendFrame, 120);
      return;
    }

    // Downscale to 640x480 to match the pipeline's virtual reference frame
    // and to keep each upload small.
    const canvas = els.canvas;
    canvas.width = 640;
    canvas.height = 480;
    canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);

    try {
      const response = await fetch('/api/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: sessionId, image: dataUrl }),
      });

      if (response.status === 503) {
        const body = await response.json();
        setError(body.error || 'Demo is at capacity.');
        stopCamera();
        return;
      }
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || `Server returned ${response.status}`);
      }

      const data = await response.json();
      els.output.src = 'data:image/jpeg;base64,' + data.image;
      updateUI(data);
      updateThroughput();
      setError('');
      els.serverStatus.textContent = 'Online';
      els.serverStatus.className = 'value online';

    } catch (err) {
      console.error(err);
      setError('Connection problem: ' + err.message);
      els.serverStatus.textContent = 'Error';
      els.serverStatus.className = 'value offline';
    }

    // Only queue the next frame after this one completes.
    if (running) setTimeout(sendFrame, 40);
  }

  // ---------- camera control ----------

  async function startCamera() {
    setError('');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('This browser does not support camera access. Try Chrome, Edge or Firefox.');
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (err) {
      const messages = {
        NotAllowedError: 'Camera permission was denied. Allow access in your browser and try again.',
        NotFoundError: 'No camera was found on this device.',
        NotReadableError: 'The camera is already in use by another application.',
      };
      setError(messages[err.name] || ('Could not open the camera: ' + err.message));
      return;
    }

    els.video.srcObject = stream;
    running = true;

    els.placeholder.style.display = 'none';
    els.output.style.display = 'block';
    els.liveBadge.style.display = 'inline-block';
    els.startLabel.textContent = 'Stop Camera';
    els.startBtn.querySelector('span').textContent = '⏹';
    els.startBtn.classList.add('paused');
    els.cameraStatus.textContent = 'Active';
    els.cameraStatus.className = 'value active';

    sendFrame();
  }

  function stopCamera() {
    running = false;
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    els.placeholder.style.display = 'flex';
    els.output.style.display = 'none';
    els.liveBadge.style.display = 'none';
    els.startLabel.textContent = 'Start Camera';
    els.startBtn.querySelector('span').textContent = '▶';
    els.startBtn.classList.remove('paused');
    els.cameraStatus.textContent = 'Stopped';
    els.cameraStatus.className = 'value';
    els.fpsStatus.textContent = '—';
    els.fpsStatus.className = 'value';
    els.statusText.textContent = 'STANDBY';
    els.statusCard.classList.remove('warning', 'danger');
    els.fallOverlay.style.display = 'none';
  }

  // ---------- wiring ----------

  els.startBtn.addEventListener('click', () => {
    running ? stopCamera() : startCamera();
  });

  els.muteBtn.addEventListener('click', () => {
    muted = !muted;
    els.muteBtn.classList.toggle('paused', muted);
    els.muteIcon.textContent = muted ? '🔕' : '🔔';
    els.muteLabel.textContent = muted ? 'Muted' : 'Mute';
  });

  fetch('/api/health')
    .then(r => r.json())
    .then(d => {
      els.serverStatus.textContent = 'Online';
      els.serverStatus.className = 'value online';
      if (d.active_sessions >= d.max_sessions) {
        setError('The demo is currently at capacity. You may need to wait a moment.');
      }
    })
    .catch(() => {
      els.serverStatus.textContent = 'Offline';
      els.serverStatus.className = 'value offline';
    });

  updateClock();
  setInterval(updateClock, 30000);
  window.addEventListener('beforeunload', stopCamera);
});
