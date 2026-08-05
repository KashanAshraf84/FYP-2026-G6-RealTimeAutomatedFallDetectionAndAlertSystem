"""
Fall Detection System - Alert System
=====================================
Handles fall notifications via:
  - Console alerts with colored output
  - Sound alarms
  - Email notifications
  - JSON event logging with timestamps and frame captures
"""

import os
import json
import time
import smtplib
import threading
import cv2
import numpy as np
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Optional, Dict, List
from pathlib import Path

from config import AlertConfig
from database import Database


class AlertSystem:
    """
    Multi-channel alert system for fall detection events.
    Supports sound, email, and logging with configurable cooldowns.
    """

    def __init__(self, config: Optional[AlertConfig] = None, db: Optional[Database] = None):
        self.config = config or AlertConfig()
        self.db = db
        self._last_alert_time = 0
        self._alert_count = 0
        self._muted = False

        # Ensure log directory exists
        os.makedirs(self.config.log_dir, exist_ok=True)

        # Event log file
        self._log_file = os.path.join(
            self.config.log_dir,
            f"fall_events_{datetime.now().strftime('%Y%m%d')}.json",
        )
        self._events: List[Dict] = []
        self._load_existing_log()

    def _load_existing_log(self):
        """Load existing events from today's log file."""
        if os.path.exists(self._log_file):
            try:
                with open(self._log_file, "r") as f:
                    self._events = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._events = []

    def set_muted(self, muted: bool) -> bool:
        """Mute/unmute the buzzer and popup alert (events are still logged)."""
        self._muted = muted
        return self._muted

    def is_muted(self) -> bool:
        return self._muted

    def trigger_alert(
        self,
        status: str,
        confidence: float,
        frame: Optional[np.ndarray] = None,
        person_id: int = 0,
        extra_info: Optional[Dict] = None,
        event_id: Optional[int] = None,
    ) -> bool:
        """
        Trigger a fall detection alert.

        Args:
            status: "normal", "warning", or "fall"
            confidence: Detection confidence (0-1)
            frame: Current video frame for capture
            person_id: ID of the person who fell
            extra_info: Additional info to log

        Returns:
            True if alert was triggered, False if suppressed by cooldown
        """
        current_time = time.time()

        # Cooldown check
        if current_time - self._last_alert_time < self.config.alert_cooldown_seconds:
            return False

        self._last_alert_time = current_time
        self._alert_count += 1
        timestamp = datetime.now().isoformat()

        # Create event record
        event = {
            "event_id": self._alert_count,
            "timestamp": timestamp,
            "status": status,
            "confidence": round(confidence * 100, 2),
            "person_id": person_id,
        }
        if extra_info:
            event.update(extra_info)

        # Save frame capture
        frame_path = None
        if frame is not None:
            frame_dir = os.path.join(self.config.log_dir, "captures")
            os.makedirs(frame_dir, exist_ok=True)
            frame_filename = f"fall_{self._alert_count}_{datetime.now().strftime('%H%M%S')}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            event["frame_capture"] = frame_path

        # --- Console Alert ---
        self._console_alert(status, confidence, timestamp, person_id)

        # Desktop-bound channels are unavailable on a headless host (no display,
        # no audio device), so they are skipped entirely in that profile.
        desktop_alerts = not self._muted and not self.config.headless

        # --- Visual Alert (Popup) ---
        if status in ["warning", "fall"] and desktop_alerts:
            # Run visual alert in a separate thread to avoid blocking main loop
            threading.Thread(
                target=self._visual_alert,
                args=(status, confidence),
                daemon=True
            ).start()

        # --- Desktop Notification (visible even if the dashboard isn't open) ---
        if status in ["warning", "fall"] and desktop_alerts:
            threading.Thread(
                target=self._desktop_notification,
                args=(status, confidence),
                daemon=True,
            ).start()

        # --- Sound Alert ---
        if self.config.enable_sound and status == "fall" and desktop_alerts:
            # Run in a separate thread — winsound.Beep() blocks the calling
            # thread, which would otherwise stall the frame-processing loop.
            threading.Thread(target=self._sound_alert, daemon=True).start()

        # --- Email Alert ---
        if self.config.enable_email and status == "fall":
            threading.Thread(
                target=self._email_alert,
                args=(event, frame_path),
                daemon=True,
            ).start()

        # --- Log Event ---
        if self.config.enable_logging:
            # Strictly formatted log entry for academic requirement
            log_entry = {
                "status": status,
                "confidence": round(confidence * 100, 2),
                "timestamp": timestamp,
                "person_id": person_id
            }
            self._log_event(log_entry)

        # --- Database record ---
        if self.db is not None:
            self.db.log_alert(status=status, confidence=confidence, person_id=person_id, event_id=event_id)

        return True

    def _visual_alert(self, status: str, confidence: float):
        """Display a dedicated high-priority alert window."""
        window_name = f"!!! {status.upper()} ALERT !!!"
        h, w = 400, 600
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background color
        color = (0, 0, 255) if status == "fall" else (0, 200, 255)
        canvas[:] = color
        
        # Border
        cv2.rectangle(canvas, (10, 10), (w-10, h-10), (255, 255, 255), 3)

        # Text
        text = f"INTERNAL ALERT: {status.upper()}"
        conf_text = f"Confidence: {confidence * 100:.1f}%"
        
        cv2.putText(canvas, text, (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(canvas, conf_text, (50, 250), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Check Safety Immediately!", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Show for a configurable duration
        cv2.imshow(window_name, canvas)
        cv2.waitKey(self.config.popup_duration_ms)
        cv2.destroyWindow(window_name)

    def _console_alert(
        self, status: str, confidence: float, timestamp: str, person_id: int
    ):
        """Print colored console alert."""
        colors = {
            "normal": "\033[92m",    # Green
            "warning": "\033[93m",   # Yellow
            "fall": "\033[91m",      # Red
        }
        reset = "\033[0m"
        color = colors.get(status, reset)

        icons = {
            "normal": "✅",
            "warning": "⚠️ ",
            "fall": "🚨",
        }
        icon = icons.get(status, "")

        print(f"\n{color}{'='*50}")
        print(f"{icon}  FALL DETECTION ALERT  {icon}")
        print(f"{'='*50}")
        print(f"  Status:     {status.upper()}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  Person ID:  {person_id}")
        print(f"  Time:       {timestamp}")
        print(f"{'='*50}{reset}\n")

    def _sound_alert(self):
        """Play alarm sound safely across platforms."""
        try:
            import platform
            if platform.system() == "Windows":
                import winsound
                # Demo-friendly alert pattern
                winsound.Beep(1000, 200)
                winsound.Beep(1500, 200)
                winsound.Beep(1000, 200)
            else:
                # Fallback for Linux/Mac
                print("\a" * 3)
        except Exception as e:
            print(f"  ⚠ Sound alert failed (this is normal on some systems): {e}")
            print("\a\a\a")

    def _desktop_notification(self, status: str, confidence: float):
        """Show a native OS notification (Windows Action Center toast, etc.).

        Visible even when the browser dashboard isn't open.
        """
        try:
            from plyer import notification
            icon = "🚨" if status == "fall" else "⚠️"
            notification.notify(
                title=f"{icon} Fall Detection Alert",
                message=f"Status: {status.upper()} ({confidence * 100:.1f}% confidence)",
                app_name="GuardianAI",
                timeout=10,
            )
        except Exception as e:
            print(f"  ⚠ Desktop notification failed: {e}")

    def _email_alert(self, event: Dict, frame_path: Optional[str] = None):
        """Send email notification (runs in background thread)."""
        if not self.config.sender_email or not self.config.recipient_emails:
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.config.sender_email
            msg["To"] = ", ".join(self.config.recipient_emails)
            msg["Subject"] = f"🚨 FALL DETECTED - {event['timestamp']}"

            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background-color: #ff4444; color: white; padding: 20px; border-radius: 10px;">
                    <h1>🚨 Fall Detection Alert</h1>
                </div>
                <div style="padding: 20px;">
                    <table style="border-collapse: collapse; width: 100%;">
                        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Status</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd; color: red; font-weight: bold;">{event['status'].upper()}</td></tr>
                        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Confidence</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{event['confidence']}%</td></tr>
                        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Person ID</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{event.get('person_id', 'N/A')}</td></tr>
                        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Timestamp</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{event['timestamp']}</td></tr>
                    </table>
                    <p style="color: #666; margin-top: 20px;">Please check on the individual immediately.</p>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, "html"))

            # Attach frame capture
            if frame_path and os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename="fall_capture.jpg",
                    )
                    msg.attach(img)

            # Send
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.send_message(msg)

            print("📧 Email alert sent successfully")

        except Exception as e:
            print(f"❌ Email alert failed: {e}")

    def _log_event(self, event: Dict):
        """Log event to JSON file."""
        self._events.append(event)
        try:
            with open(self._log_file, "w") as f:
                json.dump(self._events, f, indent=2, default=str)
        except IOError as e:
            print(f"Warning: Could not write log file: {e}")

    def get_event_summary(self) -> Dict:
        """Get summary of today's events."""
        falls = sum(1 for e in self._events if e["status"] == "fall")
        warnings = sum(1 for e in self._events if e["status"] == "warning")
        return {
            "total_events": len(self._events),
            "falls": falls,
            "warnings": warnings,
            "log_file": self._log_file,
        }
