from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator


class Camera(models.Model):
    KINDS = [
        ("browser", "Browser webcam"),
        ("video", "Local video playback"),
        ("worker", "Server camera worker"),
    ]
    name = models.CharField(max_length=80)
    location = models.CharField(max_length=100)
    kind = models.CharField(max_length=12, choices=KINDS, default="browser")
    enabled = models.BooleanField(default=True)
    last_seen = models.DateTimeField(null=True, blank=True)
    last_status = models.CharField(max_length=24, default="offline")
    last_result = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return self.name


class DetectionSettings(models.Model):
    confidence = models.FloatField(
        default=0.80, validators=[MinValueValidator(0.5), MaxValueValidator(1)]
    )
    consecutive_frames = models.PositiveIntegerField(
        default=3, validators=[MinValueValidator(1), MaxValueValidator(15)]
    )
    cooldown_seconds = models.PositiveIntegerField(
        default=30, validators=[MinValueValidator(1), MaxValueValidator(600)]
    )
    email_enabled = models.BooleanField(default=False)
    email_recipient = models.EmailField(blank=True)

    @classmethod
    def current(cls):
        return cls.objects.get_or_create(pk=1)[0]


class TrackState(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    track_id = models.CharField(max_length=80)
    streak = models.PositiveIntegerField(default=0)
    last_seen = models.DateTimeField()
    last_alert = models.DateTimeField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["camera", "track_id"], name="unique_camera_track"
            )
        ]


class Event(models.Model):
    STATES = [
        ("open", "Open"),
        ("acknowledged", "Acknowledged"),
        ("resolved", "Resolved"),
        ("false_alarm", "False alarm"),
    ]
    camera = models.ForeignKey(Camera, on_delete=models.PROTECT)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    track_id = models.CharField(max_length=80)
    label = models.CharField(
        max_length=12, choices=[("fall", "Fall"), ("warning", "Warning")]
    )
    confidence = models.FloatField()
    model_version = models.CharField(max_length=100)
    evidence = models.JSONField(default=dict)
    is_demo = models.BooleanField(default=False, db_index=True)
    status = models.CharField(max_length=16, choices=STATES, default="open")
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL
    )
    updated_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]


class Delivery(models.Model):
    event = models.ForeignKey(
        Event, related_name="deliveries", on_delete=models.CASCADE
    )
    created_at = models.DateTimeField(auto_now_add=True)
    channel = models.CharField(max_length=20)
    status = models.CharField(max_length=30)
    detail = models.CharField(max_length=250, blank=True)


class Profile(models.Model):
    """Role of a workspace account. Saving it keeps Django's is_staff / is_superuser in step."""

    ADMIN, OPERATOR, CAREGIVER = "admin", "operator", "caregiver"
    ROLES = [
        (ADMIN, "Administrator"),
        (OPERATOR, "Operator"),
        (CAREGIVER, "Caregiver"),
    ]
    ROLE_HELP = {
        ADMIN: "Everything, plus user management",
        OPERATOR: "Cameras, sources and detection settings",
        CAREGIVER: "Monitor, review incidents and reports",
    }
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile"
    )
    role = models.CharField(max_length=12, choices=ROLES, default=CAREGIVER)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        staff, superuser = self.role in (self.ADMIN, self.OPERATOR), self.role == self.ADMIN
        if (self.user.is_staff, self.user.is_superuser) != (staff, superuser):
            type(self.user).objects.filter(pk=self.user.pk).update(
                is_staff=staff, is_superuser=superuser
            )
            self.user.is_staff, self.user.is_superuser = staff, superuser

    def __str__(self):
        return f"{self.user} ({self.get_role_display()})"


def role_of(user):
    """Role of a signed-in user (profiles are created automatically)."""
    if not user.is_authenticated:
        return None
    profile = getattr(user, "profile", None)
    if profile is None:
        profile = Profile.objects.create(user=user, role=default_role(user))
    return profile.role


def default_role(user):
    if user.is_superuser:
        return Profile.ADMIN
    return Profile.OPERATOR if user.is_staff else Profile.CAREGIVER


class PasswordResetCode(models.Model):
    """One-time 6-digit code e-mailed for a password reset (stored hashed)."""

    LIFETIME_MINUTES = 15
    MAX_ATTEMPTS = 5
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    code_hash = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)
    attempts = models.PositiveSmallIntegerField(default=0)
    used = models.BooleanField(default=False)


class Audit(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL
    )
    action = models.CharField(max_length=80)
    detail = models.CharField(max_length=250)
