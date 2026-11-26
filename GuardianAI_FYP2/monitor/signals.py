from django.contrib.auth import get_user_model
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Profile, default_role


@receiver(post_save, sender=get_user_model())
def create_profile(sender, instance, created, raw=False, **kwargs):
    """Every account gets a role; createsuperuser accounts become Administrators."""
    if created and not raw:
        Profile.objects.get_or_create(user=instance, defaults={"role": default_role(instance)})
