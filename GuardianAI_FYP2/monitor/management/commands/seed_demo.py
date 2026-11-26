from django.core.management.base import BaseCommand
from monitor.models import Camera, DetectionSettings


class Command(BaseCommand):
    help = "Create example camera configuration. Does not invent events or users."

    def handle(self, *args, **options):
        for name, loc, kind in [
            ("Bed bay 1–4", "Ward A", "browser"),
            ("Corridor east", "Ground floor", "worker"),
            ("Day room", "Ward A", "video"),
            ("Bed bay 7", "Ward B", "browser"),
        ]:
            Camera.objects.get_or_create(
                name=name, defaults={"location": loc, "kind": kind}
            )
        DetectionSettings.current()
        self.stdout.write(
            self.style.SUCCESS(
                "Example cameras created. All start offline. Create your user with createsuperuser."
            )
        )
