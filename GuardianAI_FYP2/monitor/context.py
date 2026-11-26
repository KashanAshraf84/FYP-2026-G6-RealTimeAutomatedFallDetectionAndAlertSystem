from django.conf import settings
from .models import Profile, role_of


def app_context(request):
    role = role_of(request.user) if hasattr(request, "user") else None
    return {
        "detector_name": settings.DETECTOR_CLASS.rsplit(".", 1)[-1],
        "model_configured": settings.DETECTOR_CLASS
        != "ai.adapters.UnconfiguredDetector",
        "page_name": request.resolver_match.url_name if request.resolver_match else "",
        "user_role": role,
        "user_role_label": dict(Profile.ROLES).get(role, ""),
        "is_admin": role == Profile.ADMIN,
    }
