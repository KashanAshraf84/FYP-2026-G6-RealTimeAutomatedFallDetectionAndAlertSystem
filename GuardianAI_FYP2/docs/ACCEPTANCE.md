# Verification and acceptance

Verified on 21 September 2026, Windows, Python 3.14.6, Django 5.2.17, Pillow 12.3.0.

## Completed

- Django system check: no issues.
- 18 Django tests pass, covering authentication, staff/caregiver permissions, CSRF, all main page renders, frame input rejection, valid detector output through the HTTP endpoint, consecutive-frame confirmation, cooldown, camera/person isolation, gaps/missing/low-score resets, not-ready models, invalid predictions, demo/report/email separation, review persistence, CSV formula escaping, settings validation, notification failure, stale telemetry, report date boundaries and evaluation validation.
- Python static checks: Ruff F checks pass; Python sources formatted.
- JavaScript syntax checks: Node `--check` passes on both scripts.
- Real browser sign-in, monitor wall, source page, demo creation and false-alarm review with notes: verified.
- Synthetic 15-second WebM playback -> sampled JPEG requests -> Django -> default adapter -> live measured processing time and Preview only state: verified. Stop releases playback and stops requests.
- OpenCV camera worker processed the same synthetic video through EOF and marked its camera offline.
- Desktop light/dark dashboard and 390-pixel mobile layout inspected. Mobile sign-out and single-column camera tiles included.
- Browser console inspection during the synthetic video test: no errors/warnings.

## Not claimed as tested

No physical webcam/RTSP device was opened, no trained model was evaluated, and no external email/SMS/push/buzzer was sent. Synthetic video validates plumbing, not fall-detection accuracy. GPU dependencies and the teammate's future model require their own verification. Public deployment, concurrent multi-process inference and large camera counts are not supported by this local prototype configuration.

## Teammate acceptance checklist

1. Extract the clean ZIP; follow README in a fresh environment; create your own superuser.
2. Sign in, add/edit a camera, disable/re-enable it, and verify permissions with a non-staff caregiver.
3. Create a labelled demo, acknowledge it, then resolve or mark it a false alarm; check notes and CSV export.
4. Connect an actual webcam on localhost and verify Start / Stop and permission-denied behaviour.
5. Load your model adapter; validate stable person identity and different cameras with known clips.
6. Confirm real falls require the configured frame streak and repeated scores respect cooldown.
7. Verify report counts exclude demos; do not interpret them as model recall/accuracy.
8. If email is wanted, configure an approved SMTP account/recipient and verify actual receipt independently of the submitted record.
9. Evaluate on approved held-out data and fill MODEL_CARD_TEMPLATE.md.

The distributed archive excludes the preview test account/database, local secrets, synthetic test video and virtual environment.
