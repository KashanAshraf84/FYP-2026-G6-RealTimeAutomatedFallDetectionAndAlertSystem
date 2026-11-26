import csv, io, json, logging, time
from dataclasses import asdict
from datetime import timedelta, datetime, time as datetime_time
from PIL import Image, UnidentifiedImageError
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import RequestDataTooBig
from django.db.models import Count
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.views.decorators.http import require_POST
from .models import Camera, Event, DetectionSettings, Audit, Delivery
from .forms import CameraForm, SettingsForm, ReviewForm
from .services import ingest, notify
from ai.runtime import next_frame_ms, predict

log = logging.getLogger(__name__)


def staff(view):
    from functools import wraps

    @wraps(view)
    @login_required
    def wrapped(request, *args, **kwargs):
        if not request.user.is_staff:
            return HttpResponse("Administrator access required", status=403)
        return view(request, *args, **kwargs)

    return wrapped


def audit(request, action, detail):
    Audit.objects.create(actor=request.user, action=action, detail=detail[:250])


def camera_data(camera):
    fresh = (
        camera.enabled
        and camera.last_seen
        and (timezone.now() - camera.last_seen).total_seconds() < 10
    )
    return {
        "id": camera.pk,
        "name": camera.name,
        "location": camera.location,
        "enabled": camera.enabled,
        "kind": camera.kind,
        "status": camera.last_status if fresh else "offline",
        "result": camera.last_result if fresh else {},
        "last_seen": camera.last_seen.isoformat() if camera.last_seen else None,
    }


@login_required
def wall(request):
    return render(
        request,
        "wall.html",
        {
            "cameras": Camera.objects.all(),
            "alerts": Event.objects.filter(status="open").select_related("camera")[:6],
        },
    )


@login_required
def status(request):
    cameras = [camera_data(c) for c in Camera.objects.all()]
    alerts = Event.objects.filter(status="open").select_related("camera")[:6]
    return JsonResponse(
        {
            "cameras": cameras,
            "open_events": Event.objects.filter(status="open", is_demo=False).count(),
            "demo_events": Event.objects.filter(status="open", is_demo=True).count(),
            "falls_today": Event.objects.filter(
                is_demo=False, created_at__date=timezone.localdate()
            ).count(),
            "alerts": [
                {
                    "id": e.pk,
                    "camera": e.camera.name,
                    "confidence": e.confidence,
                    "demo": e.is_demo,
                    "time": timezone.localtime(e.created_at).strftime("%H:%M:%S"),
                }
                for e in alerts
            ],
        }
    )


@login_required
def cameras(request):
    return render(request, "cameras.html", {"cameras": Camera.objects.all()})


@staff
def camera_edit(request, pk=None):
    camera = get_object_or_404(Camera, pk=pk) if pk else None
    form = CameraForm(request.POST or None, instance=camera)
    if request.method == "POST" and form.is_valid():
        saved = form.save()
        audit(request, "camera_saved", f"Camera #{saved.pk}")
        messages.success(request, "Camera saved.")
        return redirect("cameras")
    return render(
        request,
        "form.html",
        {
            "form": form,
            "title": "Edit camera" if pk else "Add camera",
            "subtitle": "Name the source and choose how frames will reach GuardianAI.",
        },
    )


@login_required
def camera_detail(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    return render(
        request,
        "camera.html",
        {"camera": camera, "events": Event.objects.filter(camera=camera)[:5]},
    )


@require_POST
@staff
def frame(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    if not camera.enabled:
        return JsonResponse({"error": "Camera is disabled"}, status=409)
    if camera.kind == "worker":
        return JsonResponse(
            {"error": "Use the server camera worker for this source"}, status=409
        )
    if request.content_type != "image/jpeg":
        return JsonResponse({"error": "Send an image/jpeg body"}, status=415)
    try:
        body = request.body
        if not body or len(body) > 1500000:
            raise ValueError("Frame must be between 1 byte and 1.5 MB")
        with Image.open(io.BytesIO(body)) as original:
            if original.width > 1920 or original.height > 1080:
                raise ValueError("Maximum frame size is 1920 x 1080")
            original.load()
            rgb = original.convert("RGB")
    except (
        ValueError,
        UnidentifiedImageError,
        OSError,
        RequestDataTooBig,
        Image.DecompressionBombError,
    ):
        return JsonResponse(
            {"error": "Invalid image. Send a JPEG up to 1920 x 1080 and 1.5 MB."},
            status=400,
        )
    started = time.perf_counter()
    try:
        result = predict(rgb, camera.pk)
        created = ingest(camera, result)
        LIVE_FRAMES[camera.pk] = (time.time(), body)
    except Exception:
        log.exception("Inference failed for camera %s", camera.pk)
        Camera.objects.filter(pk=camera.pk).update(
            last_status="error", last_result={}, last_seen=timezone.now()
        )
        return JsonResponse(
            {
                "error": "Model inference failed. Check the server log and adapter contract."
            },
            status=503,
        )
    return JsonResponse(
        {
            **asdict(result),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "next_ms": next_frame_ms(),
            "event_ids": [e.pk for e in created],
        }
    )


# Latest JPEG per camera for the monitor wall preview: kept in this process's memory only,
# overwritten by every new frame and never written to disk.
LIVE_FRAMES = {}
SNAPSHOT_MAX_AGE_S = 6


@login_required
def snapshot(request, pk):
    item = LIVE_FRAMES.get(pk)
    if not item or time.time() - item[0] > SNAPSHOT_MAX_AGE_S:
        LIVE_FRAMES.pop(pk, None)
        return HttpResponse(status=204)
    response = HttpResponse(item[1], content_type="image/jpeg")
    response["Cache-Control"] = "no-store"
    return response


@require_POST
@staff
def demo(request, pk):
    camera = get_object_or_404(Camera, pk=pk, enabled=True)
    recent = Event.objects.filter(
        camera=camera,
        is_demo=True,
        created_at__gte=timezone.now() - timedelta(seconds=10),
    ).first()
    if recent:
        messages.info(request, "A demo event was just created. Open it in Events.")
    else:
        e = Event.objects.create(
            camera=camera,
            track_id="DEMO-P1",
            label="fall",
            confidence=0.94,
            model_version="manual-demo-v1",
            is_demo=True,
            evidence={
                "source": "Manually generated fixture; not model inference",
                "purpose": "Verify review and dashboard workflows",
            },
        )
        notify(e)
        audit(request, "demo_event_created", f"Event #{e.pk}")
        messages.success(
            request,
            "Demo incident created. It is excluded from real reports and external email.",
        )
    return redirect("events")


def filtered_events(request):
    qs = Event.objects.select_related("camera", "updated_by").all()
    status = request.GET.get("status", "")
    if status in dict(Event.STATES):
        qs = qs.filter(status=status)
    mode = request.GET.get("mode", "")
    if mode in {"real", "demo"}:
        qs = qs.filter(is_demo=mode == "demo")
    query = request.GET.get("q", "").strip()[:100]
    if query:
        from django.db.models import Q

        qs = qs.filter(
            Q(camera__name__icontains=query)
            | Q(track_id__icontains=query)
            | Q(notes__icontains=query)
        )
    return qs


@login_required
def events(request):
    from django.core.paginator import Paginator

    page = Paginator(filtered_events(request), 20).get_page(request.GET.get("page"))
    return render(request, "events.html", {"events": page, "states": Event.STATES})


@login_required
def event_detail(request, pk):
    event = get_object_or_404(
        Event.objects.select_related("camera", "updated_by"), pk=pk
    )
    form = ReviewForm(request.POST or None, instance=event)
    if request.method == "POST" and form.is_valid():
        event = form.save(commit=False)
        event.updated_by = request.user
        event.updated_at = timezone.now()
        event.save()
        audit(request, "incident_reviewed", f"Event #{event.pk}: {event.status}")
        messages.success(request, "Incident review saved.")
        return redirect("event", pk=pk)
    return render(
        request,
        "event.html",
        {
            "event": event,
            "form": form,
            "evidence": json.dumps(event.evidence, indent=2),
        },
    )


@login_required
def export(request):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="guardian-events.csv"'
    w = csv.writer(response)
    w.writerow(
        [
            "id",
            "created_at",
            "camera",
            "track_id",
            "label",
            "confidence",
            "model_version",
            "is_demo",
            "review_status",
            "notes",
        ]
    )

    def safe(v):
        s = str(v)
        return (
            "'" + s
            if s.lstrip().startswith(("=", "+", "-", "@"))
            or s.startswith(("\t", "\r", "\n"))
            else s
        )

    for e in filtered_events(request).iterator():
        w.writerow(
            [
                safe(x)
                for x in [
                    e.pk,
                    e.created_at.isoformat(),
                    e.camera.name,
                    e.track_id,
                    e.label,
                    e.confidence,
                    e.model_version,
                    e.is_demo,
                    e.status,
                    e.notes,
                ]
            ]
        )
    return response


@login_required
def reports(request):
    days = request.GET.get("days", "7")
    days = int(days) if days in {"1", "7", "30"} else 7
    start = timezone.make_aware(
        datetime.combine(
            timezone.localdate() - timedelta(days=days - 1), datetime_time.min
        )
    )
    qs = Event.objects.filter(is_demo=False, created_at__gte=start)
    daily = []
    for i in reversed(range(days)):
        d = timezone.localdate() - timedelta(days=i)
        daily.append(
            {
                "label": d.strftime("%d %b"),
                "count": qs.filter(created_at__date=d).count(),
            }
        )
    maximum = max([d["count"] for d in daily] + [1])
    for d in daily:
        d["height"] = max(2, d["count"] / maximum * 100)
    return render(
        request,
        "reports.html",
        {
            "days": days,
            "total": qs.count(),
            "false_alarms": qs.filter(status="false_alarm").count(),
            "resolved": qs.filter(status="resolved").count(),
            "daily": daily,
            "by_camera": qs.values("camera__name")
            .annotate(count=Count("id"))
            .order_by("-count"),
        },
    )


@staff
def settings_view(request):
    form = SettingsForm(request.POST or None, instance=DetectionSettings.current())
    if request.method == "POST" and form.is_valid():
        form.save()
        audit(
            request,
            "settings_saved",
            "Alert thresholds and email configuration updated",
        )
        messages.success(request, "Settings saved. New predictions use these values.")
        return redirect("settings")
    return render(request, "settings.html", {"form": form})


@login_required
def notifications(request):
    from django.conf import settings

    return render(
        request,
        "notifications.html",
        {
            "deliveries": Delivery.objects.select_related(
                "event", "event__camera"
            ).order_by("-created_at")[:50],
            "email_console": settings.EMAIL_BACKEND.endswith("console.EmailBackend"),
            "cfg": DetectionSettings.current(),
        },
    )
