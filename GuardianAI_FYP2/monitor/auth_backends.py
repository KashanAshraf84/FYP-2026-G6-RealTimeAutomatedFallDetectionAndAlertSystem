from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend


class UsernameOrEmailBackend(ModelBackend):
    """Sign in with either the username or the account e-mail address (case-insensitive)."""

    def authenticate(self, request, username=None, password=None, **kwargs):
        if username is None or password is None:
            return None
        User = get_user_model()
        user = find_user(username)
        if user is None:
            User().set_password(password)  # same work as a real check, so timing does not reveal accounts
            return None
        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None


def find_user(identifier):
    """Account matching a username or an e-mail address, or None."""
    User = get_user_model()
    identifier = (identifier or "").strip()
    if not identifier:
        return None
    if "@" in identifier:
        matches = list(User.objects.filter(email__iexact=identifier)[:2])
        if len(matches) == 1:
            return matches[0]
    return User.objects.filter(username__iexact=identifier).first()
