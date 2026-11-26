import os, time
from django.core.management.base import BaseCommand, CommandError
from django.db import close_old_connections
from PIL import Image
from monitor.models import Camera
from monitor.services import ingest
from ai.runtime import predict


class Command(BaseCommand):
    help = "Read a trusted server USB / RTSP / video source and run the configured detector."

    def add_arguments(self, p):
        p.add_argument("--camera", type=int, required=True)
        p.add_argument(
            "--source-env",
            default="GUARDIAN_CAMERA_SOURCE",
            help="Environment variable holding an index, path, or RTSP URL",
        )
        p.add_argument("--fps", type=float, default=5)

    def handle(self, *args, **o):
        try:
            import cv2
        except ImportError:
            raise CommandError("Install requirements-ai.txt first.")
        if not 0.1 <= o["fps"] <= 30:
            raise CommandError("FPS must be between 0.1 and 30")
        camera = Camera.objects.filter(pk=o["camera"], kind="worker").first()
        if not camera:
            raise CommandError("Select an existing Server camera worker source")
        source = os.environ.get(o["source_env"])
        if not source:
            raise CommandError("Set the source environment variable first")
        # Credentials stay out of the database, UI, and application log.
        source = int(source) if source.isdecimal() else source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise CommandError(
                "Cannot open camera source. Check device access or network connection."
            )
        self.stdout.write(
            "Worker started. Ctrl+C to stop. Do not run two workers for the same camera."
        )
        try:
            while True:
                started = time.monotonic()
                close_old_connections()
                camera.refresh_from_db()
                if not camera.enabled:
                    break
                ok, bgr = cap.read()
                if not ok:
                    self.stdout.write("Source ended or disconnected.")
                    break
                h, w = bgr.shape[:2]
                if w > 960:
                    bgr = cv2.resize(bgr, (960, round(h * 960 / w)))
                frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                ingest(camera, predict(frame, camera.pk))
                time.sleep(max(0, 1 / o["fps"] - (time.monotonic() - started)))
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            Camera.objects.filter(pk=camera.pk).update(
                last_seen=None, last_status="offline", last_result={}
            )
