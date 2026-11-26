import csv, io
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch
from PIL import Image
from django.test import TestCase, Client, override_settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from ai.contracts import Prediction, Result
from ai.runtime import validate
from ai.evaluate import evaluate
from .models import Camera, Event, TrackState, DetectionSettings, Delivery
from .services import ingest


class Workflows(TestCase):
    def setUp(self):
        self.admin = get_user_model().objects.create_user(
            "admin", password="test-password", is_staff=True
        )
        self.caregiver = get_user_model().objects.create_user(
            "caregiver", password="test-password"
        )
        self.camera = Camera.objects.create(name="Ward A", location="Test ward")
        self.client.force_login(self.admin)
        self.result = Result(
            [Prediction("p1", "fall", 0.94, evidence={"angle_deg": 15})], "test-v1"
        )

    def test_anonymous_redirects(self):
        c = Client()
        for path in ["/", "/api/status/", "/events/", "/settings/", "/reports/"]:
            self.assertEqual(c.get(path).status_code, 302)

    def test_all_pages_render(self):
        e = Event.objects.create(
            camera=self.camera,
            track_id="p1",
            label="fall",
            confidence=0.9,
            model_version="test",
        )
        for path in [
            "/",
            "/cameras/",
            "/cameras/new/",
            f"/cameras/{self.camera.pk}/",
            f"/cameras/{self.camera.pk}/edit/",
            "/events/",
            f"/events/{e.pk}/",
            "/reports/",
            "/settings/",
            "/notifications/",
            "/api/status/",
            "/events/export/",
        ]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_caregiver_cannot_manage(self):
        self.client.force_login(self.caregiver)
        for path in ["/settings/", "/cameras/new/", f"/cameras/{self.camera.pk}/edit/"]:
            self.assertEqual(self.client.post(path, {}).status_code, 403)
        self.assertEqual(
            self.client.post(f"/cameras/{self.camera.pk}/demo/").status_code, 403
        )
        self.assertEqual(
            self.client.post(f"/cameras/{self.camera.pk}/frame/").status_code, 403
        )

    def test_csrf_enforced(self):
        c = Client(enforce_csrf_checks=True)
        c.force_login(self.admin)
        self.assertEqual(c.post(f"/cameras/{self.camera.pk}/demo/").status_code, 403)

    def test_alert_confirmation_cooldown_and_person_isolation(self):
        self.assertEqual(ingest(self.camera, self.result), [])
        self.assertEqual(ingest(self.camera, self.result), [])
        self.assertEqual(len(ingest(self.camera, self.result)), 1)
        for _ in range(5):
            ingest(self.camera, self.result)
        self.assertEqual(Event.objects.count(), 1)
        other = replace(self.result, predictions=[Prediction("p2", "fall", 0.99)])
        for _ in range(3):
            ingest(self.camera, other)
        self.assertEqual(Event.objects.count(), 2)
        self.assertEqual(Delivery.objects.filter(channel="dashboard").count(), 2)

    def test_missing_low_score_and_gap_reset_streak(self):
        ingest(self.camera, self.result)
        ingest(self.camera, Result([], "test"))
        ingest(self.camera, self.result)
        self.assertEqual(TrackState.objects.get().streak, 1)
        ingest(
            self.camera,
            replace(self.result, predictions=[Prediction("p1", "fall", 0.2)]),
        )
        self.assertEqual(TrackState.objects.get().streak, 0)
        ingest(self.camera, self.result)
        TrackState.objects.update(last_seen=timezone.now() - timedelta(seconds=4))
        ingest(self.camera, self.result)
        self.assertEqual(TrackState.objects.get().streak, 1)

    def test_cameras_do_not_share_track_state(self):
        c2 = Camera.objects.create(name="Other", location="Test")
        ingest(self.camera, self.result)
        ingest(self.camera, self.result)
        ingest(c2, self.result)
        self.assertFalse(Event.objects.exists())

    def test_not_ready_produces_no_alerts(self):
        for _ in range(5):
            ingest(self.camera, replace(self.result, ready=False))
        self.assertFalse(Event.objects.exists())

    def test_invalid_prediction_rejected(self):
        for p in [
            Prediction("p", "fall", float("nan")),
            Prediction("p", "bad", 0.9),
            Prediction("p", "fall", 2),
            Prediction("p", "fall", 0.9, (0.9, 0.2, 0.1, 0.8)),
        ]:
            with self.assertRaises(ValueError):
                validate(Result([p], "test"))
        self.assertFalse(Event.objects.exists())

    @override_settings(DETECTOR_CLASS="ai.adapters.UnconfiguredDetector")
    def test_frame_default_adapter_and_bad_input(self):
        output = io.BytesIO()
        Image.new("RGB", (10, 10)).save(output, "JPEG")
        url = f"/cameras/{self.camera.pk}/frame/"
        response = self.client.post(url, output.getvalue(), content_type="image/jpeg")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])
        self.assertFalse(Event.objects.exists())
        self.assertEqual(
            self.client.post(url, b"bad", content_type="image/jpeg").status_code, 400
        )
        self.assertEqual(
            self.client.post(url, b"bad", content_type="text/plain").status_code, 415
        )
        self.camera.enabled = False
        self.camera.save()
        self.assertEqual(
            self.client.post(
                url, output.getvalue(), content_type="image/jpeg"
            ).status_code,
            409,
        )

    def test_demo_excluded_from_real_reports_and_email(self):
        cfg = DetectionSettings.current()
        cfg.email_enabled = True
        cfg.email_recipient = "a@example.test"
        cfg.save()
        self.client.post(f"/cameras/{self.camera.pk}/demo/")
        self.assertTrue(Event.objects.get().is_demo)
        self.assertFalse(Delivery.objects.filter(channel="email").exists())
        self.assertEqual(self.client.get("/reports/").context["total"], 0)
        self.assertEqual(self.client.get("/api/status/").json()["open_events"], 0)

    def test_review_persists_and_csv_safe(self):
        self.camera.name = "=HYPERLINK(test)"
        self.camera.save()
        e = Event.objects.create(
            camera=self.camera,
            track_id="p",
            label="fall",
            confidence=0.9,
            model_version="test",
        )
        self.client.force_login(self.caregiver)
        self.client.post(
            f"/events/{e.pk}/", {"status": "false_alarm", "notes": "Sitting down"}
        )
        e.refresh_from_db()
        self.assertEqual(e.status, "false_alarm")
        self.assertEqual(e.updated_by, self.caregiver)
        rows = list(
            csv.reader(io.StringIO(self.client.get("/events/export/").content.decode()))
        )
        self.assertTrue(rows[1][2].startswith("'="))

    def test_settings_validation(self):
        response = self.client.post(
            "/settings/",
            {
                "confidence": "1.5",
                "consecutive_frames": 0,
                "cooldown_seconds": 0,
                "email_enabled": "on",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["form"].errors)
        self.assertEqual(DetectionSettings.current().confidence, 0.8)

    @override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
    def test_smtp_failure_does_not_lose_incident(self):
        cfg = DetectionSettings.current()
        cfg.email_enabled = True
        cfg.email_recipient = "a@example.test"
        cfg.save()
        with patch("monitor.services.send_mail", side_effect=RuntimeError("failure")):
            for _ in range(3):
                ingest(self.camera, self.result)
        self.assertEqual(Event.objects.count(), 1)
        self.assertEqual(Delivery.objects.get(channel="email").status, "failed")

    def test_evaluation_metrics_and_leakage_guard(self):
        rows = [
            dict(
                clip_id="a", subject_id="1", split="test", y_true="fall", y_pred="fall"
            ),
            dict(
                clip_id="b",
                subject_id="2",
                split="test",
                y_true="fall",
                y_pred="normal",
            ),
        ]
        result = evaluate(rows)
        self.assertEqual(result["per_class"]["fall"]["recall"], 0.5)
        self.assertEqual(result["accuracy"], 0.5)
        with self.assertRaises(ValueError):
            evaluate(rows + rows)
        with self.assertRaises(ValueError):
            evaluate([{**rows[0], "split": "train"}])

    def test_frame_endpoint_records_valid_model_fall(self):
        output = io.BytesIO()
        Image.new("RGB", (40, 30)).save(output, "JPEG")
        with patch("monitor.views.predict", return_value=self.result):
            responses = [
                self.client.post(
                    f"/cameras/{self.camera.pk}/frame/",
                    output.getvalue(),
                    content_type="image/jpeg",
                )
                for _ in range(3)
            ]
        self.assertTrue(all(r.status_code == 200 for r in responses))
        self.assertEqual(len(responses[-1].json()["event_ids"]), 1)
        self.assertEqual(Event.objects.get().model_version, "test-v1")
        self.assertFalse(Event.objects.get().is_demo)

    def test_stale_telemetry_is_not_reported_as_live(self):
        self.camera.last_seen = timezone.now() - timedelta(seconds=11)
        self.camera.last_status = "fall"
        self.camera.last_result = {"predictions": [{"track_id": "old"}]}
        self.camera.save()
        data = self.client.get("/api/status/").json()["cameras"][0]
        self.assertEqual(data["status"], "offline")
        self.assertEqual(data["result"], {})

    def test_local_calendar_report_boundary(self):
        event = Event.objects.create(
            camera=self.camera,
            track_id="p",
            label="fall",
            confidence=0.9,
            model_version="test",
        )
        yesterday = timezone.now() - timedelta(days=1)
        Event.objects.filter(pk=event.pk).update(created_at=yesterday)
        self.assertEqual(self.client.get("/reports/?days=1").context["total"], 0)
        self.assertEqual(self.client.get("/reports/?days=7").context["total"], 1)


import re
from django.core import mail
from django.core.cache import cache
from .models import Audit, Profile, PasswordResetCode


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class Accounts(TestCase):
    PW = "Calm-River-2026"

    def setUp(self):
        cache.clear()

    def register(self, username, email, pw=None, pw2=None, client=None):
        return (client or self.client).post("/register/", {
            "username": username, "email": email, "password1": pw or self.PW, "password2": pw2 or pw or self.PW})

    def make(self, username, role, email=None):
        u = get_user_model().objects.create_user(username, email=email or f"{username}@example.com", password=self.PW)
        u.profile.role = role
        u.profile.save()
        return u

    def test_first_account_is_admin_then_caregiver(self):
        self.assertContains(self.client.get("/login/"), "Create the first account")
        self.assertEqual(self.register("boss", "boss@example.com").status_code, 302)
        boss = get_user_model().objects.get(username="boss")
        self.assertEqual((boss.profile.role, boss.is_staff, boss.is_superuser), ("admin", True, True))
        self.register("nurse", "nurse@example.com", client=Client())
        nurse = get_user_model().objects.get(username="nurse")
        self.assertEqual((nurse.profile.role, nurse.is_staff), ("caregiver", False))

    def test_registration_validation(self):
        self.make("taken", "caregiver", "taken@example.com")
        for data, field in [
            (("taken", "new@example.com"), "username"),
            (("fresh", "TAKEN@example.com"), "email"),
            (("fresh", "not-an-email"), "email"),
        ]:
            r = self.register(*data)
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.context["form"].has_error(field), field)
        self.assertTrue(self.register("fresh", "f@example.com", "short").context["form"].has_error("password1"))
        self.assertTrue(self.register("fresh", "f@example.com", self.PW, "Other-Pass-99").context["form"].has_error("password2"))

    @override_settings(GUARDIAN_REGISTRATION_APPROVAL=True)
    def test_registration_approval_mode(self):
        self.make("boss", "admin")
        self.register("nurse", "nurse@example.com")
        nurse = get_user_model().objects.get(username="nurse")
        self.assertFalse(nurse.is_active)
        r = self.client.post("/login/", {"username": "nurse", "password": self.PW})
        self.assertContains(r, "waiting for an administrator")

    def test_login_with_email_and_lockout(self):
        self.make("nurse", "caregiver", "nurse@example.com")
        self.assertEqual(self.client.post("/login/", {"username": "NURSE@example.com", "password": self.PW}).status_code, 302)
        c = Client()
        for _ in range(5):
            c.post("/login/", {"username": "nurse", "password": "wrong"})
        r = c.post("/login/", {"username": "nurse", "password": self.PW})
        self.assertContains(r, "Too many failed attempts")

    def test_open_redirect_blocked(self):
        self.make("nurse", "caregiver")
        r = self.client.post("/login/", {"username": "nurse", "password": self.PW, "next": "https://evil.example/"})
        self.assertEqual(r.url, "/")

    def test_password_reset_by_code(self):
        user = self.make("nurse", "caregiver", "nurse@example.com")
        r = self.client.post("/password/forgot/", {"identifier": "nurse"})
        self.assertRedirects(r, "/password/reset/")
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ["nurse@example.com"])
        code = re.search(r"\b(\d{6})\b", mail.outbox[0].body).group(1)
        wrong = "000000" if code != "000000" else "111111"
        r = self.client.post("/password/reset/", {"code": wrong, "password1": "New-Garden-77", "password2": "New-Garden-77"})
        self.assertContains(r, "Incorrect code")
        r = self.client.post("/password/reset/", {"code": code, "password1": "New-Garden-77", "password2": "New-Garden-77"})
        self.assertRedirects(r, "/login/")
        user.refresh_from_db()
        self.assertTrue(user.check_password("New-Garden-77"))
        # A used code cannot be replayed.
        self.client.post("/password/forgot/", {"identifier": "nurse"})  # throttled: no new mail
        self.assertEqual(len(mail.outbox), 1)
        r = self.client.post("/password/reset/", {"code": code, "password1": "Another-Pass-88", "password2": "Another-Pass-88"})
        self.assertContains(r, "invalid or has expired")

    def test_reset_by_email_and_unknown_identifier(self):
        self.make("nurse", "caregiver", "nurse@example.com")
        self.client.post("/password/forgot/", {"identifier": "Nurse@Example.com"})
        self.assertEqual(len(mail.outbox), 1)
        r = Client().post("/password/forgot/", {"identifier": "nobody@example.com"})
        self.assertRedirects(r, "/password/reset/")  # same response: does not reveal accounts
        self.assertEqual(len(mail.outbox), 1)

    def test_reset_code_expires_and_attempts_limited(self):
        user = self.make("nurse", "caregiver")
        self.client.post("/password/forgot/", {"identifier": "nurse"})
        code = re.search(r"\b(\d{6})\b", mail.outbox[0].body).group(1)
        PasswordResetCode.objects.update(created_at=timezone.now() - timedelta(minutes=16))
        r = self.client.post("/password/reset/", {"code": code, "password1": "New-Garden-77", "password2": "New-Garden-77"})
        self.assertContains(r, "invalid or has expired")
        cache.clear()
        self.client.post("/password/forgot/", {"identifier": "nurse"})
        code = re.search(r"\b(\d{6})\b", mail.outbox[1].body).group(1)
        wrong = "000000" if code != "000000" else "111111"
        for _ in range(5):
            self.client.post("/password/reset/", {"code": wrong, "password1": "New-Garden-77", "password2": "New-Garden-77"})
        r = self.client.post("/password/reset/", {"code": code, "password1": "New-Garden-77", "password2": "New-Garden-77"})
        self.assertContains(r, "invalid or has expired")
        user.refresh_from_db()
        self.assertTrue(user.check_password(self.PW))

    def test_role_permissions(self):
        admin, operator, caregiver = self.make("boss", "admin"), self.make("op", "operator"), self.make("nurse", "caregiver")
        expect = {admin: (200, 200), operator: (403, 200), caregiver: (403, 403)}
        for user, (users_page, settings_page) in expect.items():
            self.client.force_login(user)
            self.assertEqual(self.client.get("/users/").status_code, users_page, user.username)
            self.assertEqual(self.client.get("/settings/").status_code, settings_page, user.username)
            self.assertEqual(self.client.get("/account/").status_code, 200)

    def test_admin_gives_and_takes_rights(self):
        admin, nurse = self.make("boss", "admin"), self.make("nurse", "caregiver")
        self.client.force_login(admin)
        self.client.post(f"/users/{nurse.pk}/", {"action": "save", "role": "operator", "is_active": "on"})
        nurse.refresh_from_db()
        self.assertEqual((nurse.profile.role, nurse.is_staff, nurse.is_superuser), ("operator", True, False))
        c = Client(); c.force_login(nurse)
        self.assertEqual(c.get("/settings/").status_code, 200)
        self.client.post(f"/users/{nurse.pk}/", {"action": "save", "role": "caregiver"})  # unchecked = deactivate
        nurse.refresh_from_db()
        self.assertEqual((nurse.profile.role, nurse.is_staff, nurse.is_active), ("caregiver", False, False))
        self.assertEqual(Client().post("/login/", {"username": "nurse", "password": self.PW}).status_code, 200)
        self.assertTrue(Audit.objects.filter(action="user_rights_changed").exists())

    def test_admin_safety_rules(self):
        admin = self.make("boss", "admin")
        self.client.force_login(admin)
        self.client.post(f"/users/{admin.pk}/", {"action": "save", "role": "caregiver", "is_active": "on"})
        self.client.post(f"/users/{admin.pk}/", {"action": "delete"})
        admin.refresh_from_db()
        self.assertEqual(admin.profile.role, "admin")
        second = self.make("boss2", "admin")
        self.client.post(f"/users/{second.pk}/", {"action": "delete"})
        self.assertFalse(get_user_model().objects.filter(pk=second.pk).exists())

    def test_admin_creates_user_and_sends_code(self):
        admin = self.make("boss", "admin")
        self.client.force_login(admin)
        self.client.post("/users/", {"username": "newop", "email": "newop@example.com", "password1": self.PW, "password2": self.PW, "role": "operator"})
        newop = get_user_model().objects.get(username="newop")
        self.assertEqual(newop.profile.role, "operator")
        self.client.post(f"/users/{newop.pk}/", {"action": "send_code"})
        self.assertEqual(mail.outbox[-1].to, ["newop@example.com"])

    def test_account_page_updates_email_and_password(self):
        nurse = self.make("nurse", "caregiver", "old@example.com")
        self.client.force_login(nurse)
        self.client.post("/account/", {"action": "email", "email": "New@Example.com"})
        nurse.refresh_from_db()
        self.assertEqual(nurse.email, "new@example.com")
        r = self.client.post("/account/", {"action": "password", "old_password": self.PW, "new_password1": "New-Garden-77", "new_password2": "New-Garden-77"})
        self.assertRedirects(r, "/account/")
        nurse.refresh_from_db()
        self.assertTrue(nurse.check_password("New-Garden-77"))
        self.assertEqual(self.client.get("/").status_code, 200)  # still signed in


class LiveWall(TestCase):
    def setUp(self):
        self.admin = get_user_model().objects.create_user("boss", password="x", is_staff=True)
        self.viewer = get_user_model().objects.create_user("nurse", password="x")
        self.camera = Camera.objects.create(name="Ward A", location="Test ward")
        self.result = Result([Prediction("P1", "normal", 0.9, (0.1, 0.1, 0.5, 0.9))], "test-v1")

    def test_frame_returns_pacing_and_snapshot_is_served(self):
        from monitor import views

        views.LIVE_FRAMES.clear()
        url = f"/cameras/{self.camera.pk}/snapshot/"
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get(url).status_code, 204)  # nothing live yet
        output = io.BytesIO()
        Image.new("RGB", (64, 48)).save(output, "JPEG")
        staff = Client()
        staff.force_login(self.admin)
        with patch("monitor.views.predict", return_value=self.result):
            r = staff.post(f"/cameras/{self.camera.pk}/frame/", output.getvalue(), content_type="image/jpeg")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(r.json()["next_ms"], 160)
        r = self.client.get(url)  # a caregiver on another device sees the latest frame
        self.assertEqual((r.status_code, r["Content-Type"], r.content), (200, "image/jpeg", output.getvalue()))
        views.LIVE_FRAMES[self.camera.pk] = (views.LIVE_FRAMES[self.camera.pk][0] - 60, b"old")
        self.assertEqual(self.client.get(url).status_code, 204)  # stale frames are dropped
        self.assertEqual(Client().get(url).status_code, 302)  # sign-in required

    def test_status_reports_camera_kind(self):
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get("/api/status/").json()["cameras"][0]["kind"], "browser")
