# FallNet v2 training (FYP-2)

Trains the fall classifier that the Django app uses (`ai.adapters.TeamDetector`) on the real
GMDCSA-24 videos in `data/dataset/` and measures it on people the model has never seen.

## Pipeline

```text
video (15/30 FPS) --YOLOv8n-pose--> 17 keypoints per person, sampled at 15 Hz   (extract_keypoints.py, run once)
                  --SubjectSelector--> one subject per frame                    (ai/pose_features.py)
                  --resample_window--> last 2.0 s as 30 evenly spaced steps     (same code live and offline)
                  --window_features--> 62 features per step                     (pose shape, motion, torso angle, confidences)
                  --FallNet v2------> P(normal), P(warning), P(fall)            (Bi-LSTM / CNN-LSTM / Transformer)
                  --temperature T---> calibrated scores (T fitted on validation data only)
                  --app alert rule--> fall score >= 0.80 on 3 consecutive frames (monitor/services.py)
```

Window labels come from the time ranges in `Fall.csv` / `ADL.csv` (`gmdcsa.py`):

| Label | Meaning (2 s window) |
|---|---|
| `normal` | No overlap with the annotated fall (all ADL clips, and the part of fall clips before the fall) |
| `warning` | Fall has started: 0.25 s to 1 s of the window is inside the fall range |
| `fall` | At least half of the window is inside the fall range |

## Run

```powershell
.venv\Scripts\python -m training.gmdcsa              # check annotation parsing (prints every fall range)
.venv\Scripts\python -m training.extract_keypoints   # ~20 min on a laptop CPU, resumable, writes data/keypoints/
.venv\Scripts\python -m training.train_fallnet       # LOSO cross-validation of 3 architectures + final model
.venv\Scripts\python -m training.train_fallnet --quick   # 1-fold, 3-epoch smoke test
```

Outputs:

- `weights/fallnet_v2.pt` — final model (loaded with `torch.load(..., weights_only=True)`)
- `training/results/report.json` — every metric, per fold and pooled
- `training/results/confusion_matrix_loso.png`, `architecture_comparison.png`, `final_training_curves.png`
- `training/results/cv_clip_results_<arch>_<fps>fps.csv` — per-clip alarms, latency, misses
- `training/results/cv_clip_predictions_<arch>.csv` — the same results in `ai/evaluate.py` format

## How the evaluation avoids inflated numbers

- **Subject-disjoint:** leave-one-subject-out. Each of the 4 people is the test set once, so every clip is predicted
  by a model that never saw that person. Overlapping windows from one clip never sit on both sides of a split.
- **No test-set tuning:** early stopping uses a clip-level validation split of the training subjects; the architecture
  is chosen by mean validation macro-F1. Feature normalisation is fitted on the training split only.
- **Calibrated scores:** focal loss leaves probabilities under-confident (fall scores topped out near 0.80 in a
  3-epoch test, so the app's 0.80 gate almost never fired). A temperature fitted on the validation split corrects the
  scale without changing any predicted class; expected calibration error is reported.
- **Deployment replay:** event metrics replay each held-out clip as a 5 FPS stream (the browser rate) through
  `ai/fallnet.py` and the app's alert rule, so they include warm-up, frame-rate effects and the confirmation streak.

## Google Colab

Not required: the whole run fits on a laptop CPU. If you do use Colab, YOLO extraction picks up the GPU
automatically; the classifier itself is small and trains on CPU. Upload the project (or just the code plus the
`data/keypoints/` cache, which skips YOLO entirely), `pip install -r requirements-ai.txt`, then run the same
`python -m training.train_fallnet` command from the project root.
