import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Automatically load local .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

INSTANCE = BASE_DIR / "instance"
INSTANCE.mkdir(exist_ok=True)
# A unique local secret is generated once, never packaged with the handoff.
secret_file = INSTANCE / ".secret"
if not os.environ.get("DJANGO_SECRET_KEY") and not secret_file.exists():
    import secrets

    try:
        with secret_file.open("x") as f:
            f.write(secrets.token_urlsafe(48))
    except FileExistsError:
        pass
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or secret_file.read_text().strip()
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get(
    "DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver"
).split(",")
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "monitor",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "config.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "monitor.context.app_context",
            ],
        },
    },
]
WSGI_APPLICATION = "config.wsgi.application"

def _database():
    """
    GUARDIAN_DB=postgres  -> PostgreSQL (GUARDIAN_DB_NAME / _USER / _PASSWORD / _HOST / _PORT)
    otherwise             -> SQLite file in instance/ (zero-setup default), or GUARDIAN_SQLITE_PATH
    """
    if os.environ.get("GUARDIAN_DB", "sqlite").lower() in ("postgres", "postgresql"):
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("GUARDIAN_DB_NAME", "guardianai"),
            "USER": os.environ.get("GUARDIAN_DB_USER", "guardian"),
            "PASSWORD": os.environ.get("GUARDIAN_DB_PASSWORD", ""),
            "HOST": os.environ.get("GUARDIAN_DB_HOST", "127.0.0.1"),
            "PORT": os.environ.get("GUARDIAN_DB_PORT", "5432"),
            "CONN_MAX_AGE": 60,
            "CONN_HEALTH_CHECKS": True,
        }
    # Prefer existing guardian.sqlite3 if present, fallback to guardianai.db
    db_file = os.environ.get("GUARDIAN_SQLITE_PATH") or (
        INSTANCE / "guardian.sqlite3" if (INSTANCE / "guardian.sqlite3").exists() else INSTANCE / "guardianai.db"
    )
    return {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": db_file,
        # WAL + a busy timeout so several cameras writing at once do not hit "database is locked".
        "OPTIONS": {
            "timeout": 20,
            "transaction_mode": "IMMEDIATE",
            "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;",
        },
    }


DATABASES = {"default": _database()}
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {"min_length": 8},
    },
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]
# Sign in with username or e-mail (monitor/auth_backends.py).
AUTHENTICATION_BACKENDS = ["monitor.auth_backends.UsernameOrEmailBackend"]
# 1 = self-registered accounts stay inactive until an Administrator activates them.
GUARDIAN_REGISTRATION_APPROVAL = os.environ.get("GUARDIAN_REGISTRATION_APPROVAL", "0") == "1"
SESSION_COOKIE_AGE = 14 * 24 * 3600  # "Keep me signed in" lasts two weeks
LANGUAGE_CODE = "en-gb"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login/"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
DATA_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024
DETECTOR_CLASS = os.environ.get("GUARDIAN_DETECTOR", "ai.adapters.TeamDetector")
EMAIL_BACKEND = os.environ.get(
    "EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend"
)
EMAIL_HOST = os.environ.get("EMAIL_HOST", "localhost")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "587"))
EMAIL_USE_TLS = os.environ.get("EMAIL_USE_TLS", "1") == "1"
EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "")
EMAIL_TIMEOUT = 10
DEFAULT_FROM_EMAIL = os.environ.get("DEFAULT_FROM_EMAIL", "guardian@localhost")
