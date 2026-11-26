from django.urls import path
from . import views as v

urlpatterns = [
    path("", v.wall, name="wall"),
    path("api/status/", v.status, name="status"),
    path("cameras/", v.cameras, name="cameras"),
    path("cameras/new/", v.camera_edit, name="camera_add"),
    path("cameras/<int:pk>/", v.camera_detail, name="camera"),
    path("cameras/<int:pk>/edit/", v.camera_edit, name="camera_edit"),
    path("cameras/<int:pk>/frame/", v.frame, name="frame"),
    path("cameras/<int:pk>/snapshot/", v.snapshot, name="snapshot"),
    path("cameras/<int:pk>/demo/", v.demo, name="demo"),
    path("events/", v.events, name="events"),
    path("events/export/", v.export, name="export"),
    path("events/<int:pk>/", v.event_detail, name="event"),
    path("reports/", v.reports, name="reports"),
    path("settings/", v.settings_view, name="settings"),
    path("notifications/", v.notifications, name="notifications"),
]
