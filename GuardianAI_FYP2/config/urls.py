from django.contrib import admin
from django.contrib.auth import views as auth
from django.urls import path, include
from monitor import accounts

urlpatterns = [
    path("admin/", admin.site.urls),
    path("login/", accounts.sign_in, name="login"),
    path("logout/", auth.LogoutView.as_view(), name="logout"),
    path("register/", accounts.register, name="register"),
    path("password/forgot/", accounts.forgot_password, name="forgot_password"),
    path("password/reset/", accounts.reset_password, name="reset_password"),
    path("account/", accounts.account, name="account"),
    path("users/", accounts.users, name="users"),
    path("users/<int:pk>/", accounts.user_update, name="user_update"),
    path("", include("monitor.urls")),
]
