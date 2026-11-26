from django.contrib import admin
from .models import Camera, Event, Delivery, Audit, DetectionSettings

admin.site.register([Camera, Event, Delivery, Audit, DetectionSettings])
