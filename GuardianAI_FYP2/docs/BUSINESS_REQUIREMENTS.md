# Business Requirements — GuardianAI

*Derived from code and existing docs. Items marked (unconfirmed) were not stated in the code.*

## 1. Business purpose
Detect falls of people in camera view, alert staff quickly, and give them a workflow to review, annotate and report
incidents. Context: university Final Year Project (FYP-2026-G6, team lead/owner Kashan Ashraf) that **may be piloted in a
hospital/care setting** later. Seed data uses ward names ("Ward A", "Day room", "Bed bay").
Not a certified medical device; there are no validated clinical accuracy claims.

Success measures stated in the project: fall recall and false-alarm rate on held-out subjects, latency, and a live demo
(webcam fall + a "sit quickly" false-alarm challenge). Current measured result (GMDCSA-24, leave-one-subject-out, 5 FPS):
69/79 falls detected (87.3%), 7/81 normal clips with a false alarm (8.6%), median alert latency 1.71 s.
The ROADMAP's targets (>=92% F1, <=50 ms/frame, AUC >= 0.95) are goals, not achieved results.

## 2. Users and roles
| Role | Rights (enforced in code) |
|---|---|
| **Administrator** (`is_superuser`, `is_staff`) | Everything, plus Users page: create/delete users, change role, activate/deactivate, send reset codes |
| **Operator** (`is_staff`) | Add/edit cameras, send camera frames (start sources), change detection settings, create demo incidents |
| **Caregiver** | Monitor wall, view cameras, review incidents, export CSV, reports, notifications |

- The first registered account becomes Administrator; later self-sign-ups are Caregivers (optionally inactive until approved: `GUARDIAN_REGISTRATION_APPROVAL=1`).
- Safeguards: cannot demote/deactivate/delete yourself or the last active Administrator.
- Note: caregivers cannot start browser camera capture (frame endpoint requires staff).

## 3. Main features
1. **Sign-in / accounts**: username or e-mail login, "keep me signed in" 14 days, lockout after 5 failures (15 min), password reset by 6-digit e-mailed code (15 min, 5 tries), own-account page.
2. **Camera management**: three source kinds — browser webcam, local video playback (analysed in browser), server worker (USB/RTSP/file via `camera_worker`). Enable/disable.
3. **Live monitor wall**: per-camera status (offline after 10 s without frames), person boxes with IDs (P1, P2…), NORMAL / WARNING / FALL / WARMING UP / PARTIAL VIEW, snapshot previews, open-alert panel. Polls every 3 s.
4. **AI detection**: YOLOv8-Pose + tracker + FallNet v2; evidence (probabilities, trunk angle, velocities) stored with each incident.
5. **Incident workflow**: Event list with filters/search/paging; review states Open -> Acknowledged -> Resolved / False alarm, with notes; CSV export (formula-injection safe).
6. **Notifications**: dashboard record always; optional e-mail (SMTP) to one configured recipient; opt-in browser notifications. Demo incidents never e-mail.
7. **Reports**: real incidents per day over 1/7/30 days, false-alarm and resolved counts, per-camera counts.
8. **Settings**: confidence gate (0.5–1.0), consecutive frames (1–15), cooldown (1–600 s), e-mail on/off and recipient.
9. **Audit log**: camera/settings edits, demo creation, reviews, account events (viewable in Django admin).
10. **Demo incident**: labelled `is_demo`, excluded from reports and e-mail, throttled.
11. **Public browser demo**: same workspace in the browser, data local to the visitor.

## 4. Business rules
- **Alert rule**: a `fall` prediction with score >= confidence threshold (default 0.80) for N consecutive frames (default 3) of the same camera+person creates an Event; per-person cooldown (default 30 s) suppresses repeats; a gap > 3 s, a missing person, low score or non-fall label resets the streak.
- Only confirmed **falls** are persisted. `warning` is shown live only (the Event model allows it but no code creates it).
- Person with only head/shoulders visible = `PARTIAL VIEW`; cannot raise an alert. Needs ~1.5 s of history before predicting.
- Track IDs are per camera, never reused while the server runs.
- Demo events are excluded from reports/status counts of real incidents and from e-mail.
- Raw frames are never stored; only the latest JPEG per camera is kept in memory for ~6 s (wall preview).
- Email "submitted" does not mean "delivered"; console backend is labelled `console_only`.
- Frames: JPEG only, <= 1.5 MB, <= 1920x1080; browser sends at most ~5 FPS; server paces cameras to keep model busy <= 65%.
- Reports use calendar days in the server timezone (currently UTC).
- Deleting a camera with events is blocked (PROTECT).

## 5. Workflows
**Fall detection**: frame -> detector -> per-person labels -> confirmation (streak + threshold) -> Event + dashboard delivery (+ optional e-mail) -> caregiver opens event -> Acknowledge / Resolve / False alarm + notes -> appears in reports and CSV.

**Onboarding**: first user registers (Administrator) -> admin creates operators/caregivers or approves sign-ups -> add cameras -> start source -> tune settings.

**Password reset**: request code (throttled 60 s) -> e-mail -> enter code + new password.

## 6. Out of scope / not implemented (per code and docs)
SMS, buzzer/GPIO, background mobile push, escalation, evidence video clips, face blurring, retention controls, RTSP video streamed to browser, auto-reconnect and supervision of workers, bed/sofa exclusion zones, post-fall immobility check (roadmap items).

## 7. Open business questions (unconfirmed)
- Real hospital target, number of cameras/wards, legal/privacy approval (consent, retention, location of data).
- Who receives alerts after hours; escalation expectations; required uptime/SLA.
- Whether a single alert e-mail recipient is sufficient.
