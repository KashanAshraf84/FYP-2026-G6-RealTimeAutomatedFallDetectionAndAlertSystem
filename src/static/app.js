// dashboard UI logic - Imran
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const statusText = document.getElementById('current-status-text');
    const statusCard = document.getElementById('main-status-card');
    const confidenceFill = document.getElementById('status-confidence-fill');
    const confidenceVal = document.getElementById('status-confidence-val');
    const personCountText = document.getElementById('person-count');
    const totalFallsText = document.getElementById('total-falls');
    const eventLog = document.getElementById('event-log');
    const alertHistory = document.getElementById('alert-history');
    const fallOverlay = document.getElementById('fall-overlay');
    const currentTime = document.getElementById('current-time');
    const alertSound = document.getElementById('alert-sound');

    const bodyAngleText = document.getElementById('body-angle');
    const dropSpeedText = document.getElementById('drop-speed');

    const feedToggleBtn = document.getElementById('feed-toggle-btn');
    const feedToggleIcon = document.getElementById('feed-toggle-icon');
    const feedToggleLabel = document.getElementById('feed-toggle-label');
    const videoFeed = document.getElementById('video-feed');
    const feedPausedOverlay = document.getElementById('feed-paused-overlay');

    const muteToggleBtn = document.getElementById('mute-toggle-btn');
    const muteToggleIcon = document.getElementById('mute-toggle-icon');
    const muteToggleLabel = document.getElementById('mute-toggle-label');

    const systemStatusValue = document.getElementById('system-status-value');
    const detectionStatusValue = document.getElementById('detection-status-value');

    document.querySelectorAll('.nav-item.disabled').forEach((item) => {
        item.addEventListener('click', (e) => e.preventDefault());
    });

    let lastStatus = 'normal';
    let fallCount = 0;
    let feedPaused = false;

    async function fetchStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            const current = data.current_status;
            const summary = data.event_summary;

            // 1. Update Current Status UI
            updateStatusUI(current);

            // 2. Update Stats
            personCountText.textContent = current.person_count || 0;
            totalFallsText.textContent = summary.falls;
            
            // 2.1 Update Telemetry
            if (bodyAngleText && current.angle !== undefined) {
                bodyAngleText.textContent = `${current.angle.toFixed(1)}°`;
            }
            if (dropSpeedText && current.speed !== undefined) {
                dropSpeedText.textContent = current.speed.toFixed(1);
            }

            // 3. Check for new fall events
            if (summary.falls > fallCount) {
                fallCount = summary.falls;
                triggerFallUI();
            }

            // 4. Update Event Log
            // Assuming current_status might have a timestamp or we use current time
            if (current.status !== 'normal' && lastStatus === 'normal') {
                addLogItem(current.status, new Date().toISOString(), current.confidence);
            }
            lastStatus = current.status;

            // 5. System / Detection indicators
            systemStatusValue.textContent = 'Online';
            systemStatusValue.className = 'value online';

            if (feedPaused) {
                detectionStatusValue.textContent = 'Paused';
                detectionStatusValue.className = 'value paused';
            } else if (data.camera_connected) {
                detectionStatusValue.textContent = 'Active';
                detectionStatusValue.className = 'value active';
            } else {
                detectionStatusValue.textContent = 'No Camera';
                detectionStatusValue.className = 'value offline';
            }

            applyMuteState(data.alerts_muted);

        } catch (error) {
            console.error('Error fetching system status:', error);
            systemStatusValue.textContent = 'Offline';
            systemStatusValue.className = 'value offline';
            detectionStatusValue.textContent = 'Unknown';
            detectionStatusValue.className = 'value offline';
        }
    }

    function updateTime() {
        const now = new Date();
        currentTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function addLogItem(status, timestamp, confidence) {
        const placeholder = eventLog.querySelector('.log-placeholder');
        if (placeholder) placeholder.remove();

        const item = document.createElement('div');
        item.className = `log-item ${status}`;

        const time = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        item.innerHTML = `
            <span class="log-status">${status.toUpperCase()}</span>
            <span class="log-time">${time} &middot; ${confidence.toFixed(1)}%</span>
        `;

        eventLog.prepend(item);

        const items = eventLog.querySelectorAll('.log-item');
        if (items.length > 20) {
            items[items.length - 1].remove();
        }
    }

    // -- Alert History (reads the SQLite `alerts` table via /alerts) --
    async function fetchAlertHistory() {
        try {
            const response = await fetch('/alerts?limit=20');
            const data = await response.json();
            renderAlertHistory(data.alerts || []);
        } catch (error) {
            console.error('Error fetching alert history:', error);
        }
    }

    function renderAlertHistory(alerts) {
        if (!alerts.length) {
            alertHistory.innerHTML = '<div class="log-placeholder">No alerts recorded yet...</div>';
            return;
        }

        alertHistory.innerHTML = '';
        alerts.forEach((a) => {
            const item = document.createElement('div');
            item.className = `log-item ${a.status}`;

            const time = new Date(a.timestamp).toLocaleTimeString([], {
                hour: '2-digit', minute: '2-digit', second: '2-digit',
            });
            const confidencePct = (a.confidence * 100).toFixed(1);
            const action = a.acknowledged
                ? '<span class="log-ack-badge">&check; Acknowledged</span>'
                : `<button class="log-ack-btn" data-alert-id="${a.id}">Acknowledge</button>`;

            item.innerHTML = `
                <div class="log-item-main">
                    <span class="log-status">${a.status.toUpperCase()}</span>
                    <span class="log-time">${time} &middot; ${confidencePct}%</span>
                </div>
                ${action}
            `;
            alertHistory.appendChild(item);
        });
    }

    // Event delegation: alert rows are re-rendered on every poll, so the
    // click listener is attached once to the container rather than per-button.
    alertHistory.addEventListener('click', async (e) => {
        const btn = e.target.closest('.log-ack-btn');
        if (!btn) return;
        const alertId = btn.dataset.alertId;
        btn.disabled = true;
        btn.textContent = 'Acknowledging...';
        try {
            await fetch(`/alerts/${alertId}/acknowledge`, { method: 'POST' });
            fetchAlertHistory();
        } catch (error) {
            console.error('Failed to acknowledge alert:', error);
            btn.disabled = false;
            btn.textContent = 'Acknowledge';
        }
    });

    function applyMuteState(muted) {
        muteToggleBtn.classList.toggle('paused', muted);
        muteToggleIcon.textContent = muted ? '🔕' : '🔔';
        muteToggleLabel.textContent = muted ? 'Muted' : 'Mute';
    }

    function updateStatusUI(data) {
        const status = data.status || 'normal';
        const confidence = data.confidence || 0;

        statusText.textContent = status.toUpperCase();
        confidenceVal.textContent = `${confidence.toFixed(1)}%`;
        confidenceFill.style.width = `${confidence}%`;

        // Update Card Colors
        statusCard.classList.remove('warning', 'danger');
        if (status === 'warning') statusCard.classList.add('warning');
        if (status === 'fall') statusCard.classList.add('danger');

        // Handle Fall Overlay
        if (status === 'fall') {
            fallOverlay.style.display = 'block';
        } else {
            fallOverlay.style.display = 'none';
        }
    }

    function triggerFallUI() {
        // Play Alert Sound
        try {
            alertSound.play();
        } catch (e) {
            console.warn('Audio playback blocked by browser policies. Interaction required.');
        }

        // Add visual flash class to body or dashboard
        document.body.style.animation = 'none';
        setTimeout(() => {
            document.body.style.animation = 'dangerFlash 2s 3';
        }, 10);
    }

    // -- Video Feed Toggle --
    // Removing the <img> src drops the streaming connection, which stops the
    // server from processing frames for this client — that's what silences
    // the buzzer (it's only triggered from inside the frame-processing loop).
    feedToggleBtn.addEventListener('click', () => {
        feedPaused = !feedPaused;

        if (feedPaused) {
            videoFeed.src = '';
            feedPausedOverlay.classList.add('active');
            feedToggleBtn.classList.add('paused');
            feedToggleIcon.textContent = '▶';
            feedToggleLabel.textContent = 'Resume';
        } else {
            videoFeed.src = '/video_feed?t=' + Date.now();
            feedPausedOverlay.classList.remove('active');
            feedToggleBtn.classList.remove('paused');
            feedToggleIcon.textContent = '⏸';
            feedToggleLabel.textContent = 'Pause';
        }
    });

    // -- Mute Toggle --
    // Keeps video/detection running but tells the server to suppress the
    // buzzer and popup alert (events are still logged).
    muteToggleBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/alerts/mute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            const data = await response.json();
            applyMuteState(data.muted);
        } catch (error) {
            console.error('Failed to toggle mute:', error);
        }
    });

    // -- Initialization --

    // Add dynamic animation for fall alert
    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes dangerFlash {
            0% { background-color: var(--bg-color); }
            50% { background-color: rgba(255, 71, 87, 0.2); }
            100% { background-color: var(--bg-color); }
        }
    `;
    document.head.appendChild(style);

    // Initial update
    updateTime();
    setInterval(updateTime, 60000);

    // API Polling every 500ms
    setInterval(fetchStatus, 500);

    // Alert history doesn't need sub-second refresh; poll every 3s
    fetchAlertHistory();
    setInterval(fetchAlertHistory, 3000);

    console.log('Premium Fall Detection Dashboard Initialized');
});
