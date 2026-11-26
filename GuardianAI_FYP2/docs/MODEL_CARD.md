# GuardianAI model card — FallNet v2

All numbers below were measured by `training/train_fallnet.py` on 3 November 2026 and are stored in
`training/results/report.json`. None come from the UI design or synthetic data.

## Summary

| | |
|---|---|
| Model name/version | `fallnet-v2-cnnlstm-9385bb3524` — CNN-LSTM temporal pose classifier (normal / warning / fall) |
| Owner and date | GuardianAI FYP-2 team (Kashan Ashraf); trained 2026-11-03 14:07 |
| Weights | `weights/fallnet_v2.pt`, 932,182 bytes, SHA-256 `c17148a6fcab88b54d067a78c087ed49d2b130123d8eacfa7c26c5547d1327ed`. Tensors and plain values only: load with `torch.load(..., weights_only=True)` |
| Pose model | `weights/yolov8n-pose.pt` (Ultralytics YOLOv8n-pose, COCO-17 keypoints) |
| Runtime | Python 3.14.6, torch 2.14.1 (CPU build), ultralytics 8.4.183, numpy 2.5.4, opencv 5.0.0, scikit-learn 1.9.1 — `requirements-ai-lock.txt` |
| Training entry point | `python -m training.extract_keypoints` then `python -m training.train_fallnet` (seed 42) |
| Configuration | hidden 64, 2-layer bidirectional LSTM after 2×Conv1d(64), dropout 0.3, 224k parameters; AdamW lr 1e-3, weight decay 1e-3, batch 64, ≤60 epochs, early stopping patience 12 on validation macro-F1; focal loss γ=2 with square-root inverse-frequency class weights, label smoothing 0.05 |

## Data

- **Dataset:** GMDCSA-24 — 4 subjects, 79 fall clips and 81 activities-of-daily-living (ADL) clips, 640×480 and
  1280×720, 15 and 30 FPS, 21.5 minutes in total. The falls are staged by the recorded subjects.
  Check the dataset's own licence before redistributing the videos; they are not bundled with the weights.
- **Labels** (class index order `0 normal, 1 warning, 2 fall`), assigned per 2-second window from the time ranges
  in `Fall.csv` / `ADL.csv`: `fall` if at least 50% of the window is inside the annotated fall range, `warning` if
  0.25 s to 1 s is, otherwise `normal`. All ADL windows are `normal`, including lying on a bed or floor,
  push-ups and sitting on the floor.
  Annotation repairs: typing errors such as a missing bracket or a missing "to" are tolerated. One row without a fall
  range (Subject 4, Fall 15) is assumed to fall from 1.7 s, when the annotated sitting ends.
- **Windows:** 5,037 in total (3,688 normal, 222 warning, 1,127 fall), one every 0.2 s per clip.

## Split and protocol

Leave-one-subject-out (LOSO): 4 folds, each holding out one person entirely.

| Test subject | Test clips | Test windows (normal / warning / fall) | Training clips | Validation clips |
|---|---|---|---|---|
| S1 | 32 | 491 / 42 / 223 | 109 | 19 |
| S2 | 48 | 1365 / 78 / 400 | 95 | 17 |
| S3 | 43 | 969 / 58 / 304 | 99 | 18 |
| S4 | 37 | 863 / 44 / 200 | 105 | 18 |

- Validation is a stratified 15% clip-level split of the training subjects. It is used for early stopping,
  temperature calibration and architecture choice. Feature normalisation is fitted on the training split only.
- The architecture was chosen by mean validation macro-F1 (CNN-LSTM 0.873, Transformer 0.859, Bi-LSTM 0.858). The
  test folds were not used for any choice.
- The deployed model was retrained on all 4 subjects (136 training clips, 24 validation clips; best epoch 12).
  Its expected performance on new people is the LOSO result below, not its own validation score.

## Preprocessing (shared code: `ai/pose_features.py`)

- **Input:** the app passes RGB frames, which are converted to BGR for YOLO (imgsz 640, detection confidence 0.25).
  Keypoints are divided by frame height on both axes, so 4:3 and 16:9 sources keep body proportions.
- **Tracking during training and LOSO evaluation:** single subject. The detection overlapping the previous choice
  (IoU ≥ 0.3 within 1 s) is kept; otherwise the largest box × confidence is chosen.
- **Tracking in the live app:** multi-person (`ai/tracking.py`). Every person gets a stable ID (`P1`, `P2`, …, never
  reused while the server runs), their own pose history and their own prediction and alert streak. See
  *Multi-person tracking* below. Tracker state resets after a 3 s gap in frames (`ai/runtime.py`).
- **Sampling:** training poses were cached at 15 Hz. At inference, any frame rate is accepted: the last 2.0 s of
  timestamped poses are linearly resampled to 30 steps, so the browser's ~5 FPS and the worker's 5–15 FPS match
  training. Training also simulates 4–15 FPS capture.
- **Missing poses:** frames without a person are skipped and the gaps are interpolated. The last window is reused
  for at most 1.0 s if the person disappears. At least 1.5 s of history is needed before any prediction; until then
  the adapter reports `unknown`.
- **Features:** 62 per step:
  - 34 hip-centred keypoint coordinates
  - 4 hip and head displacements within the window
  - 3 hip and head velocities, clipped at ±8 body-sizes/s
  - torso angle (sin and cos)
  - box aspect ratio and box height
  - 17 keypoint confidences

  Positions are in body-size units (the median keypoint-box diagonal).
- **Calibration:** temperature scaling, T = 0.365, fitted on validation data (per-fold T 0.29–0.47).
  Focal loss otherwise left fall scores capped near 0.80.

## Results on held-out subjects (LOSO, pooled over 4 folds)

### Window level (2-second windows, CNN-LSTM)

Confusion matrix (rows = true, columns = predicted): `training/results/confusion_matrix_loso.png`

| True \ Predicted | Normal | Warning | Fall |
|---|---|---|---|
| Normal | 3426 | 84 | 178 |
| Warning | 43 | 147 | 32 |
| Fall | 189 | 39 | 899 |

| Class | Precision | Recall | F1 | Windows |
|---|---|---|---|---|
| normal | 0.937 | 0.929 | 0.933 | 3,688 |
| warning | 0.544 | 0.662 | 0.598 | 222 |
| fall | 0.811 | 0.798 | 0.804 | 1,127 |

- **Overall:** accuracy 0.888, macro-F1 0.778.
- **Fall versus rest:** sensitivity 0.798, specificity 0.946, ROC-AUC 0.937, average precision 0.828.
- **Calibration:** expected calibration error 0.043.
- **Per fold:** window macro-F1 0.830 / 0.781 / 0.804 / 0.679 for S1–S4.

### Event level (each held-out clip replayed through the live code path)

Alert rule = the app defaults in `monitor/services.py`: label `fall` with score ≥ 0.80 on 3 consecutive frames,
30 s cooldown. An alert counts as a detection if it comes at most 0.5 s before the annotated fall start.

| Stream rate | Fall clips detected | ADL clips with a false alarm | Event precision | Event F1 | Latency after fall start (median / 90th percentile) |
|---|---|---|---|---|---|
| 5 FPS (browser) | **69 / 79 (87.3%)** | **7 / 81 (8.6%)** | 0.908 | 0.890 | 1.71 s / 2.38 s |
| 15 FPS (worker) | 71 / 79 (89.9%) | 8 / 81 (9.9%) | 0.899 | 0.899 | 1.36 s / 1.72 s |

- **Per subject at 5 FPS:**
  - S1: 16 of 16 falls detected, 1 of 16 ADL clips with a false alarm
  - S2: 22 of 25 falls, 2 of 23 ADL clips
  - S3: 17 of 21 falls, 0 of 22 ADL clips
  - S4: 14 of 17 falls, 4 of 20 ADL clips
- **Before the fall:** 1 fall clip had an alert earlier than its annotated start.
- **False alarms per hour:** 33.7 per hour of ADL footage. This is 7 alarms in 12.5 minutes of staged activity that is
  dense in lying down and exercise. It does not represent continuous home monitoring, which has not been measured.
- **Threshold trade-off at 5 FPS** (for information only; the threshold was not tuned on these results):

  | Threshold | Fall recall | ADL clips without an alarm |
  |---|---|---|
  | 0.5 | 0.924 | 0.877 |
  | 0.6 | 0.924 | 0.901 |
  | 0.7 | 0.899 | 0.914 |
  | 0.8 | 0.873 | 0.914 |
  | 0.9 | 0.848 | 0.914 |

- **Independent check:** `ai/evaluate.py` on `training/results/cv_clip_predictions_cnn_lstm.csv` gives the same
  clip-level confusion: 69 of 79 falls and 74 of 81 ADL clips correct.

### Architecture comparison (same protocol): `training/results/architecture_comparison.png`

| | Bi-LSTM | **CNN-LSTM (selected)** | Transformer |
|---|---|---|---|
| Mean validation macro-F1 (selection criterion) | 0.858 | **0.873** | 0.859 |
| Window macro-F1 / fall F1 | 0.804 / 0.833 | 0.778 / 0.804 | 0.782 / 0.816 |
| Fall ROC-AUC | 0.934 | 0.937 | 0.951 |
| Fall clips detected (5 FPS) | 69 / 79 | 69 / 79 | 70 / 79 |
| ADL clips with a false alarm (5 FPS) | 11 / 81 | **7 / 81** | 9 / 81 |

The three architectures are within a few points of each other on 4 subjects. The differences are not statistically
established. CNN-LSTM was selected on validation data and has the fewest false alarms on test.

## Multi-person tracking (live app)

`ai/tracking.py` matches people between frames with Hungarian assignment on box overlap plus centre distance, so a
box that changes from tall to wide during a fall keeps its ID. Matching runs in stages:

1. Confident detections continue existing people first.
2. Low-confidence detections may only continue existing people (the ByteTrack idea), because a fallen person is
   often detected at confidence 0.3–0.5.
3. A recovery stage lets a person lost in the last 1.5 s take a nearby leftover detection.

**Ghost control.** In GMDCSA-24, YOLO reports a second "person" in 22.5% of frames: hanging clothes, pillows and
duplicate boxes. A new ID therefore needs:
- box confidence ≥ 0.5,
- mean keypoint confidence ≥ 0.5,
- 3 confident re-detections over ≥ 0.3 s,
- and it must not sit on top of someone already tracked.

**Regression check.** All 160 single-person clips were replayed through the tracker and compared with the
single-subject path. This used the final model, which saw these clips in training, so it measures the tracker's
effect only, not accuracy.

| Stream rate | Clips with exactly one ID | Falls caught (tracker / single subject) | False-alarm clips (tracker / single subject) | Alarms raised by ghost IDs |
|---|---|---|---|---|
| 5 FPS | 157 / 160 | 78 / 78 | 4 / 5 | 0 |
| 15 FPS | 151 / 160 | 79 / 79 | 5 / 4 | 0 |

The extra false alarm at 15 FPS is S2-adl-22 (push-ups). The tracker thresholds were chosen with this same check,
so the check is not independent.

**Two-person scenes.** Clips were placed side by side and run through `TeamDetector` at 5 FPS:
- one person falls while one walks: only the faller alerts (1.7 s after the fall), and the walker stays normal;
- two people fall at different times: two separate alerts with separate IDs;
- two people move normally: no alerts.

There were no ID switches in these scenes, and every result passed `ai/runtime.validate`.

**Not yet tested:** real footage of several people in one room. In particular, people crossing or hiding each other
can swap or reset IDs. A reset ID re-warms its history for 1.5 s.

## Hardware and throughput

- **Machine:** Intel Core i7-10510U (4 cores / 8 threads), 32 GB RAM, Intel UHD 620 iGPU; training on CPU only.
- **Live adapter (PyTorch CPU):** median 52 ms per frame (90th percentile 62 ms) for YOLO, features and classifier
  together, with 640-pixel JPEG frames like the browser sends.
- **Live adapter (auto-selected runtime, `ai/pose_backend.py`):** frames are letterboxed into the same stride-32
  rectangle Ultralytics uses (480x640 for 4:3, 384x640 for 16:9). On 46 test frames DirectML and OpenVINO-CPU returned
  exactly the PyTorch detections; OpenVINO on the Intel GPU (fp16) found the same people in 45 of 46 frames with
  keypoints within 2.7 px. Measured per frame on this laptop: OpenVINO Intel UHD 21 ms (8 ms of CPU), DirectML on the
  NVIDIA Quadro P520 43 ms, OpenVINO CPU 50 ms (155 ms of CPU). The selector kept OpenVINO on the Intel GPU.
- **Server load, browser webcam at 640x480:** one camera 6.0 frames/s with the server process at 16% of one core
  (28% before, with PyTorch-style square input and a busy-waiting GPU queue). Four cameras at once ran at 4.2 frames/s each (16.8 in total)
  with the server at 45% of one core, all shown live on the wall with no dropped page frames.
- **Frame rate and detection:** replaying all 160 clips through the tracker and the final model (which saw these
  clips in training, so only the trend is meaningful), falls caught were 78 at 5 frames/s, 73 at 4 and 70 at 3. Pacing
  therefore keeps each camera near 5-6 frames/s and never below 2.
- **Partial view:** a track cut off by the bottom of the frame whose knees and ankles are not visible (median
  confidence < 0.2) while head and shoulders are, is reported as `unknown / partial_view` and cannot alert. Close-up
  crops of dataset clips were classified "fall" before this rule. On the dataset replay it changed no clip's outcome.
- **Browser workspace:** WebGPU on the Intel UHD 620 takes ~90-110 ms per frame at 480 px, so one source gets ~6
  analysed frames a second and each of three simultaneous sources about 2; WebAssembly is slower. The server version is
  the one to use for several cameras. The 480 px setting was not separately evaluated.
- **Classifier alone:** 2.8 ms per window.
- **One-off costs:** pose extraction for the whole dataset took 26.3 min; cross-validation plus final training took 19.3 min.

## Known failures (from the held-out clip results, `cv_clip_results_cnn_lstm_5fps.csv`)

- **Missed falls onto or off a bed:** S2-03 and S3-10 (forward onto the bed); S4-08, S4-09 and S4-17 (from
  sitting on the bed); S2-11 (from the bed to the floor). In 2D pose these resemble lying down on purpose.
- **Missed falls that start at the first frame or from sitting on the floor:** S3-06, S3-13. There is little or no
  motion before the fall.
- **Missed fall in low light:** S3-16 (night; a person was detected in only 62 of 91 frames).
- **Missed because the video ends 1.75 s after the fall starts:** S2-18. That leaves almost no time for the 3-frame
  confirmation.
- **False alarms when lying down quickly on a bed:** S1-adl-02, S2-adl-12, S4-adl-10, S4-adl-15.
- **False alarms during push-ups:** S2-adl-22, S4-adl-05, S4-adl-07.
- **Not evaluated:**
  - real multi-person footage (tracking was checked only with side-by-side composites)
  - CCTV ceiling angles
  - elderly people and real (not staged) falls
  - slow slides longer than a few seconds
  - people other than these 4 adults
  - rooms beyond the dataset's recording locations

## Intended use

- **Intended:** a supervised FYP demonstration of indoor fall alerts for one or a few people, with human review of
  every incident in GuardianAI.
- **Out of scope:**
  - unattended clinical or care deployment
  - crowded scenes
  - outdoor use
  - any claim of accuracy beyond the subjects and conditions above

A post-fall immobility check (ROADMAP 3.1) is the obvious next step against push-up and lying-down false alarms.

## Reproduction

```powershell
.venv\Scripts\python -m training.gmdcsa              # annotation check
.venv\Scripts\python -m training.extract_keypoints   # pose cache (data/keypoints/)
.venv\Scripts\python -m training.train_fallnet       # LOSO CV + final weights + training/results/
.venv\Scripts\python -m ai.evaluate training\results\cv_clip_predictions_cnn_lstm.csv --output training\results\clip_metrics_ai_evaluate_cnn_lstm.json
```

The project folder is not a Git repository, so there is no evaluation commit. The run is identified by the model
version and the weights SHA-256 above. Training is deterministic for a fixed seed on the same hardware and library
versions.
