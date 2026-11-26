from dataclasses import asdict
from datetime import timedelta
from django.db import transaction
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from .models import Camera, DetectionSettings, TrackState, Event, Delivery
from ai.runtime import validate


def notify(event):
    Delivery.objects.create(
        event=event,
        channel="dashboard",
        status="recorded",
        detail="Available to signed-in users",
    )
    cfg = DetectionSettings.current()
    if not cfg.email_enabled or event.is_demo:
        return
    try:
        sent = send_mail(
            f"GuardianAI: {event.label.upper()} / {event.camera.name}",
            f"Event #{event.pk} | {event.created_at.isoformat()}\nCamera: {event.camera.name}\nTrack: {event.track_id}\nModel score: {event.confidence:.1%}\nReview this incident in GuardianAI.",
            settings.DEFAULT_FROM_EMAIL,
            [cfg.email_recipient],
        )
        console = settings.EMAIL_BACKEND.endswith("console.EmailBackend")
        Delivery.objects.create(
            event=event,
            channel="email",
            status="console_only" if console else ("submitted" if sent else "failed"),
            detail="Printed on server; no email sent"
            if console
            else "SMTP submission is not proof of delivery",
        )
    except Exception:
        Delivery.objects.create(
            event=event,
            channel="email",
            status="failed",
            detail="Email failed; check server configuration. Incident is retained.",
        )


def ingest(camera, result):
    validate(result)
    now = timezone.now()
    cfg = DetectionSettings.current()
    created = []
    with transaction.atomic():
        camera = Camera.objects.select_for_update().get(pk=camera.pk)
        if not camera.enabled:
            raise ValueError("Camera is disabled")
        camera.last_seen = now
        camera.last_result = asdict(result)
        labels = {p.label for p in result.predictions}
        camera.last_status = (
            "preview"
            if not result.ready
            else next(
                (x for x in ["fall", "warning", "normal", "unknown"] if x in labels),
                "no_subject",
            )
        )
        camera.save(update_fields=["last_seen", "last_result", "last_status"])
        present = [p.track_id for p in result.predictions] if result.ready else []
        TrackState.objects.filter(camera=camera).exclude(track_id__in=present).update(
            streak=0
        )
        TrackState.objects.filter(
            camera=camera, last_seen__lt=now - timedelta(hours=1)
        ).delete()
        if result.ready:
            for p in result.predictions:
                track, _ = TrackState.objects.get_or_create(
                    camera=camera, track_id=p.track_id, defaults={"last_seen": now}
                )
                if (now - track.last_seen).total_seconds() > 3:
                    track.streak = 0
                track.streak = (
                    track.streak + 1
                    if p.label == "fall" and p.confidence >= cfg.confidence
                    else 0
                )
                track.last_seen = now
                if track.streak >= cfg.consecutive_frames and (
                    not track.last_alert
                    or (now - track.last_alert).total_seconds() >= cfg.cooldown_seconds
                ):
                    event = Event.objects.create(
                        camera=camera,
                        track_id=p.track_id,
                        label="fall",
                        confidence=p.confidence,
                        model_version=result.model_version,
                        evidence=p.evidence,
                    )
                    created.append(event)
                    track.last_alert = now
                track.save()
    for event in created:
        notify(event)
    return created
