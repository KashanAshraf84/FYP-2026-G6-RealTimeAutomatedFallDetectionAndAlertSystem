"""
Accounts: registration, sign-in with lockout, password reset by e-mailed code,
own-account settings and the administrator's user dashboard.

Roles (monitor.models.Profile):
  Administrator  everything, plus this user dashboard
  Operator       cameras, sources and detection settings
  Caregiver      monitor wall, incident review and reports
"""

import hashlib
import hmac
import secrets
from datetime import timedelta
from functools import wraps

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, password_validation, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.core.cache import cache
from django.core.mail import send_mail
from django.db import transaction
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from .auth_backends import find_user
from .models import Audit, PasswordResetCode, Profile, role_of

User = get_user_model()
LOCKOUT_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60
CODE_RESEND_SECONDS = 60


def audit(actor, action, detail):
    Audit.objects.create(actor=actor if actor and actor.is_authenticated else None, action=action, detail=detail[:250])


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(request, *args, **kwargs):
        if role_of(request.user) != Profile.ADMIN:
            return HttpResponse("Administrator access required", status=403)
        return view(request, *args, **kwargs)

    return wrapped


def email_is_console():
    return settings.EMAIL_BACKEND.endswith(("console.EmailBackend", "locmem.EmailBackend"))


def client_ip(request):
    return request.META.get("REMOTE_ADDR", "")


def _active_admins():
    return User.objects.filter(is_active=True, profile__role=Profile.ADMIN)


# ------------------------------------------------------------------ forms
def _check_email_unique(email, exclude=None):
    qs = User.objects.filter(email__iexact=email)
    if exclude is not None:
        qs = qs.exclude(pk=exclude.pk)
    if qs.exists():
        raise forms.ValidationError("An account already uses this e-mail address.")


class RegisterForm(forms.Form):
    username = forms.RegexField(
        regex=r"^[\w.@+-]{3,150}$", max_length=150,
        error_messages={"invalid": "Use 3-150 letters, digits and @ . + - _ only."},
        widget=forms.TextInput(attrs={"autocomplete": "username", "autofocus": True}),
    )
    email = forms.EmailField(widget=forms.EmailInput(attrs={"autocomplete": "email"}),
                             help_text="Used to send password-reset codes.")
    password1 = forms.CharField(label="Password", strip=False,
                                widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))
    password2 = forms.CharField(label="Confirm password", strip=False,
                                widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))

    def clean_username(self):
        username = self.cleaned_data["username"]
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError("This username is taken.")
        return username

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        _check_email_unique(email)
        return email

    def clean(self):
        data = super().clean()
        p1, p2 = data.get("password1"), data.get("password2")
        if p1 and p2 and p1 != p2:
            self.add_error("password2", "The two passwords do not match.")
        elif p1 and data.get("username"):
            try:
                password_validation.validate_password(p1, User(username=data["username"], email=data.get("email", "")))
            except forms.ValidationError as e:
                self.add_error("password1", e)
        return data


class SignInForm(forms.Form):
    username = forms.CharField(label="Username or e-mail", max_length=254)
    password = forms.CharField(strip=False, widget=forms.PasswordInput)
    remember = forms.BooleanField(required=False, initial=True, label="Keep me signed in on this device")


class ResetRequestForm(forms.Form):
    identifier = forms.CharField(label="Username or e-mail", max_length=254,
                                 widget=forms.TextInput(attrs={"autofocus": True}))


class SetNewPasswordForm(forms.Form):
    code = forms.RegexField(regex=r"^\d{6}$", label="6-digit code", max_length=6,
                            error_messages={"invalid": "Enter the 6-digit code from the e-mail."},
                            widget=forms.TextInput(attrs={"inputmode": "numeric", "autocomplete": "one-time-code", "autofocus": True}))
    password1 = forms.CharField(label="New password", strip=False,
                                widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))
    password2 = forms.CharField(label="Confirm new password", strip=False,
                                widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user

    def clean(self):
        data = super().clean()
        p1, p2 = data.get("password1"), data.get("password2")
        if p1 and p2 and p1 != p2:
            self.add_error("password2", "The two passwords do not match.")
        elif p1 and self.user is not None:
            try:
                password_validation.validate_password(p1, self.user)
            except forms.ValidationError as e:
                self.add_error("password1", e)
        return data


class EmailForm(forms.Form):
    email = forms.EmailField(label="E-mail address", help_text="Password-reset codes are sent here.")

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        _check_email_unique(email, exclude=self.user)
        return email


class AdminCreateUserForm(RegisterForm):
    role = forms.ChoiceField(choices=Profile.ROLES, initial=Profile.CAREGIVER)


# ------------------------------------------------------------------ sign in / out / register
def _lock_key(request, username):
    return f"login-fail:{(username or '').strip().lower()}:{client_ip(request)}"


def sign_in(request):
    if request.user.is_authenticated:
        return redirect("wall")
    next_url = request.POST.get("next") or request.GET.get("next") or ""
    if not url_has_allowed_host_and_scheme(next_url, {request.get_host()}, request.is_secure()):
        next_url = ""
    form = SignInForm(request.POST or None)
    error = None
    if request.method == "POST" and form.is_valid():
        key = _lock_key(request, form.cleaned_data["username"])
        failures = cache.get(key, 0)
        if failures >= LOCKOUT_ATTEMPTS:
            error = "Too many failed attempts. Try again in 15 minutes or reset your password."
        else:
            user = authenticate(request, username=form.cleaned_data["username"], password=form.cleaned_data["password"])
            if user is None:
                cache.set(key, failures + 1, LOCKOUT_SECONDS)
                inactive = find_user(form.cleaned_data["username"])
                if inactive is not None and not inactive.is_active and inactive.check_password(form.cleaned_data["password"]):
                    error = "This account is waiting for an administrator to activate it."
                else:
                    left = LOCKOUT_ATTEMPTS - failures - 1
                    error = "Sign-in failed. Check your username or e-mail and password." + (
                        f" {left} attempt{'s' if left != 1 else ''} left before a 15-minute lock." if left <= 2 else "")
            else:
                cache.delete(key)
                login(request, user)
                if not form.cleaned_data.get("remember"):
                    request.session.set_expiry(0)
                role_of(user)
                return redirect(next_url or "wall")
    return render(request, "login.html", {
        "form": form, "error": error, "next": next_url,
        "first_run": not User.objects.exists(),
    })


def register(request):
    if request.user.is_authenticated:
        return redirect("wall")
    first = not User.objects.exists()
    approval = getattr(settings, "GUARDIAN_REGISTRATION_APPROVAL", False) and not first
    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            user = User.objects.create_user(
                form.cleaned_data["username"], email=form.cleaned_data["email"],
                password=form.cleaned_data["password1"], is_active=not approval,
            )
            role = Profile.ADMIN if first else Profile.CAREGIVER
            Profile.objects.update_or_create(user=user, defaults={"role": role})
        audit(user, "account_registered", f"{user.username} registered as {dict(Profile.ROLES)[role]}")
        if approval:
            messages.info(request, "Account created. An administrator must activate it before you can sign in.")
            return redirect("login")
        login(request, user, backend="monitor.auth_backends.UsernameOrEmailBackend")
        messages.success(request, (
            "Welcome! As the first account you are the Administrator." if first else
            "Welcome! Your account has the Caregiver role. An administrator can give you more rights."))
        return redirect("wall")
    return render(request, "register.html", {"form": form, "first": first, "approval": approval})


# ------------------------------------------------------------------ password reset by code
def _hash_code(user, code):
    return hmac.new(settings.SECRET_KEY.encode(), f"{user.pk}:{code}".encode(), hashlib.sha256).hexdigest()


def send_reset_code(user, request=None):
    """Create a new code for `user` (invalidating older ones) and e-mail it. Returns True if sent."""
    code = f"{secrets.randbelow(10**6):06d}"
    PasswordResetCode.objects.filter(user=user, used=False).update(used=True)
    PasswordResetCode.objects.create(user=user, code_hash=_hash_code(user, code))
    send_mail(
        "GuardianAI password reset code",
        f"Hello {user.username},\n\nYour GuardianAI password reset code is: {code}\n\n"
        f"It expires in {PasswordResetCode.LIFETIME_MINUTES} minutes. If you did not ask for it, ignore this e-mail; "
        "your password stays the same.\n\nGuardianAI",
        settings.DEFAULT_FROM_EMAIL, [user.email],
    )
    return True


def forgot_password(request):
    form = ResetRequestForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        identifier = form.cleaned_data["identifier"].strip()
        throttle = f"reset-request:{identifier.lower()}"
        user = find_user(identifier)
        if not cache.get(throttle) and user is not None and user.is_active and user.email:
            try:
                send_reset_code(user, request)
                audit(user, "password_reset_requested", f"Code e-mailed to {user.username}")
            except Exception:
                messages.error(request, "The e-mail could not be sent. Check the server's e-mail settings or ask an administrator.")
                return redirect("forgot_password")
            cache.set(throttle, 1, CODE_RESEND_SECONDS)
        request.session["reset_identifier"] = identifier
        return redirect("reset_password")
    return render(request, "forgot_password.html", {"form": form, "console": email_is_console()})


def reset_password(request):
    identifier = request.session.get("reset_identifier")
    if not identifier:
        return redirect("forgot_password")
    user = find_user(identifier)
    form = SetNewPasswordForm(request.POST or None, user=user)
    if request.method == "POST" and form.is_valid():
        record = None
        if user is not None and user.is_active:
            record = PasswordResetCode.objects.filter(
                user=user, used=False,
                created_at__gte=timezone.now() - timedelta(minutes=PasswordResetCode.LIFETIME_MINUTES),
            ).order_by("-created_at").first()
        if record is None:
            form.add_error("code", "This code is invalid or has expired. Request a new one.")
        else:
            record.attempts += 1
            if hmac.compare_digest(record.code_hash, _hash_code(user, form.cleaned_data["code"])):
                record.used = True
                record.save(update_fields=["attempts", "used"])
                user.set_password(form.cleaned_data["password1"])
                user.save(update_fields=["password"])
                cache.delete_many([k for k in [f"login-fail:{user.username.lower()}:{client_ip(request)}",
                                               f"login-fail:{(user.email or '').lower()}:{client_ip(request)}"]])
                request.session.pop("reset_identifier", None)
                audit(user, "password_reset", f"{user.username} reset the password with an e-mailed code")
                messages.success(request, "Password changed. Sign in with your new password.")
                return redirect("login")
            if record.attempts >= PasswordResetCode.MAX_ATTEMPTS:
                record.used = True
            record.save(update_fields=["attempts", "used"])
            form.add_error("code", "Incorrect code." if not record.used else "Too many wrong codes. Request a new one.")
    return render(request, "reset_password.html", {"form": form, "identifier": identifier, "console": email_is_console()})


# ------------------------------------------------------------------ own account
@login_required
def account(request):
    user = request.user
    email_form = EmailForm(initial={"email": user.email}, user=user)
    password_form = PasswordChangeForm(user)
    if request.method == "POST" and request.POST.get("action") == "email":
        email_form = EmailForm(request.POST, user=user)
        if email_form.is_valid():
            user.email = email_form.cleaned_data["email"]
            user.save(update_fields=["email"])
            audit(user, "email_changed", f"{user.username} updated the e-mail address")
            messages.success(request, "E-mail address saved.")
            return redirect("account")
    elif request.method == "POST" and request.POST.get("action") == "password":
        password_form = PasswordChangeForm(user, request.POST)
        if password_form.is_valid():
            password_form.save()
            update_session_auth_hash(request, password_form.user)
            audit(user, "password_changed", f"{user.username} changed the password")
            messages.success(request, "Password changed.")
            return redirect("account")
    return render(request, "account.html", {
        "email_form": email_form, "password_form": password_form,
        "role": dict(Profile.ROLES)[role_of(user)], "role_help": Profile.ROLE_HELP[role_of(user)],
    })


# ------------------------------------------------------------------ administrator: users
@admin_required
def users(request):
    form = AdminCreateUserForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            user = User.objects.create_user(form.cleaned_data["username"], email=form.cleaned_data["email"],
                                            password=form.cleaned_data["password1"])
            Profile.objects.update_or_create(user=user, defaults={"role": form.cleaned_data["role"]})
        audit(request.user, "user_created", f"{user.username} as {dict(Profile.ROLES)[form.cleaned_data['role']]}")
        messages.success(request, f"Account {user.username} created.")
        return redirect("users")
    accounts = User.objects.select_related("profile").order_by("username")
    for u in accounts:
        role_of(u)
    counts = {key: sum(1 for u in accounts if u.profile.role == key) for key, _ in Profile.ROLES}
    return render(request, "users.html", {
        "accounts": accounts, "roles": Profile.ROLES, "role_help": Profile.ROLE_HELP, "form": form,
        "counts": [(label, counts[key], Profile.ROLE_HELP[key]) for key, label in Profile.ROLES],
        "pending": sum(1 for u in accounts if not u.is_active), "console": email_is_console(),
    })


@require_POST
@admin_required
def user_update(request, pk):
    target = get_object_or_404(User.objects.select_related("profile"), pk=pk)
    role_of(target)
    action = request.POST.get("action")
    is_self = target.pk == request.user.pk
    last_admin = target.profile.role == Profile.ADMIN and target.is_active and _active_admins().count() <= 1

    if action == "save":
        role = request.POST.get("role")
        active = request.POST.get("is_active") == "on"
        if role not in dict(Profile.ROLES):
            messages.error(request, "Choose a valid role.")
        elif is_self and (role != Profile.ADMIN or not active):
            messages.error(request, "You cannot remove your own administrator rights or deactivate yourself.")
        elif last_admin and (role != Profile.ADMIN or not active):
            messages.error(request, "Keep at least one active Administrator.")
        else:
            changes = []
            if role != target.profile.role:
                changes.append(f"role {target.profile.get_role_display()} → {dict(Profile.ROLES)[role]}")
                target.profile.role = role
                target.profile.save()
            if active != target.is_active:
                changes.append("activated" if active else "deactivated")
                target.is_active = active
                target.save(update_fields=["is_active"])
            if changes:
                audit(request.user, "user_rights_changed", f"{target.username}: {', '.join(changes)}")
                messages.success(request, f"{target.username}: {', '.join(changes)}.")
    elif action == "send_code":
        if not target.email:
            messages.error(request, f"{target.username} has no e-mail address.")
        else:
            try:
                send_reset_code(target)
                audit(request.user, "password_reset_sent", f"Reset code e-mailed to {target.username}")
                messages.success(request, f"Reset code e-mailed to {target.email}.")
            except Exception:
                messages.error(request, "The e-mail could not be sent. Check the server's e-mail settings.")
    elif action == "delete":
        if is_self or last_admin:
            messages.error(request, "You cannot delete your own account or the last Administrator.")
        else:
            name = target.username
            target.delete()
            audit(request.user, "user_deleted", name)
            messages.success(request, f"Account {name} deleted.")
    return redirect(reverse("users") + f"#user-{pk}")
