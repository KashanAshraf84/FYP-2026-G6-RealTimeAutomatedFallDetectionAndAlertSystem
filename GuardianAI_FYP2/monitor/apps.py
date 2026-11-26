import sys
import threading

from django.apps import AppConfig


class MonitorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "monitor"

    def ready(self):
        from . import signals  # noqa: F401  (registers the profile signal)

        # Load the models while the user signs in, so the first camera frame does not wait for them.
        if "runserver" in sys.argv:
            from ai import runtime

            threading.Thread(target=_warm_up, args=(runtime,), daemon=True, name="detector-warm-up").start()


def _warm_up(runtime):
    try:
        runtime.warm_up()
    except Exception:  # a missing ML stack must not stop the web app; frames report the error
        pass
