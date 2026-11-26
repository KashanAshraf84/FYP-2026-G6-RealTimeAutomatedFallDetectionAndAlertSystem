# AI teammate handoff

## The boundary

Kashan's application side is ready for integration: pages, authentication, cameras, frame ingestion, incident persistence/review, confirmation logic and reporting. Your side is the trained detector, preprocessing, stable person tracking, temporal modelling and evidence-based evaluation.

Do not treat the original design's 92.4% / 94.1% figures, skeletons, speeds, connection tests or delivery statuses as measured results. They were hard-coded design examples. No performance claim from that artifact is carried into this project.

## First working session

1. Follow README and run the tests without installing PyTorch/YOLO.
2. Generate a demo incident and mark it a false alarm to understand the downstream workflow.
3. Read `ai/contracts.py`, `ai/runtime.py`, and `monitor/services.py`.
4. Create a Python environment compatible with your ML runtime; pin its packages in `requirements-ai.txt`.
5. Implement `TeamDetector` in `ai/adapters.py` or your own importable class.
6. Set the environment variable below, restart the server, and open a camera source.

```powershell
$env:GUARDIAN_DETECTOR = 'ai.adapters.TeamDetector'
$env:DJANGO_DEBUG = '1'
.venv\Scripts\python manage.py runserver 127.0.0.1:8000 --noreload
```

`TeamDetector` deliberately inherits the unconfigured behaviour until you implement it. Merely changing the environment variable does not load a model.

## Required Python interface

```python
from ai.contracts import Result, Prediction


class YourDetector:
    def __init__(self):
        # Load the selected model ONCE here.
        # Track state must be indexed by (camera_id, stable_person_id).
        self.buffers = {}

    def reset(self, camera_id):
        # Clear all temporal/tracker state belonging to this source.
        self.buffers.pop(camera_id, None)

    def predict(self, frame, *, camera_id, timestamp):
        # frame: PIL.Image in RGB, not OpenCV BGR.
        # timestamp: server receipt/processing Unix time in seconds.
        # Run preprocessing, tracking, temporal classifier and calibration here.
        # Return ready=False until weights are loaded and sequences are usable.
        return Result(
            [],
            "your-model-version",
            ready=False,
            message="Temporal sequence is warming up.",
        )
```

When ready, return one `Prediction` per tracked person:

- `track_id`: stable string, 1–80 characters, unique within the frame. Never use array position as identity. Namespace/epoch track IDs if your tracker reuses numeric IDs.
- `label`: `normal`, `warning`, `fall` or `unknown`.
- `confidence`: finite number in `[0,1]`, a prediction score, not dataset accuracy.
- `bbox`: optional normalized `(x1,y1,x2,y2)` relative to the submitted full frame, all in `[0,1]` with positive area. Undo letterboxing before returning.
- `evidence`: JSON object. Include interpretable measured features and units, not generic fabricated explanations.
- `Result.model_version`: stable version string (1–100 characters), retained on every event.
- `Result.ready`: false while unconfigured/warming up; no alert can be created in that state.

Example return shape (illustrative values only):

```python
Result(
    predictions=[
        Prediction(
            track_id="session-3-person-7",
            label="fall",
            confidence=0.91,
            bbox=(0.1, 0.3, 0.7, 0.8),
            evidence={
                "trunk_angle_deg": 18.2,
                "hip_velocity_frame_heights_per_second": 0.72,
                "sequence_length": 30,
                "classifier": "cnn-lstm-v2",
            },
        )
    ],
    model_version="cnn-lstm-v2-checkpoint-SHA256-prefix",
)
```

The runtime rejects invalid labels, NaN/infinite scores, invalid boxes, repeated IDs and oversized results. A failure returns HTTP 503, marks the source as errored, and does not invent a fall. Adapter calls are serialized per process. A source inactivity gap longer than 3 seconds triggers `reset(camera_id)` on its next frame. Implement reset correctly and avoid unbounded buffers.

## Frame rate and units — resolve before training

Browser input downsizes to at most 640 pixels wide, uses JPEG quality 0.8, and sends one frame after the preceding request completes plus a 200 ms delay. Input FPS therefore varies and cannot exceed about 5 FPS. The server accepts frames up to 1920×1080 / 1.5 MB.

A 30-frame window at 5 FPS spans roughly six seconds; the same window at 30 FPS spans one second. Do not reuse a FYP1 model trained at one cadence while pretending it receives the other. Choose a fixed sampling policy, resample correctly, or modify the capture pipeline and load-test it. For local video the current browser route is a playback smoke test, not reproducible offline evaluation. The worker also uses processing time, not original file presentation timestamps. For rigorous testing, build a dataset evaluator using clip timestamps.

Monocular normalized image motion is not metres per second without camera/scene calibration. FYP1 contains mixed comments and feature conventions; verify tensors and units rather than copying thresholds by name.

## FYP1 context inspected

Local reference: `Desktop/Kashan_FYP/FYP-2026-G6-RealTimeAutomatedFallDetectionAndAlertSystem` and `Desktop/Kashan_FYP/FYP1`.

- README describes YOLOv8 pose -> engineered features -> rule/CNN-LSTM hybrid -> alerts.
- Model source includes LSTM, CNN-LSTM and Transformer candidates.
- Config describes 13 selected YOLO keypoints × 3 coordinates plus 12 engineered features = 51 features per time step, and a 30-frame sequence.
- Feature code includes normalized geometry and stateful movement checks; some comments/configuration still mention MediaPipe.
- FYP1 README acknowledges quick-movement false positives, slow-slide misses, and a standalone multi-person demo not integrated into its dashboard.

This is context, not proof that a saved checkpoint matches those dimensions. Explicitly verify feature ordering, COCO keypoint subset, normalization, class order, architecture, checkpoint compatibility and training split before porting. Keep preprocessing in versioned Python modules shared by training and inference. Reuse code only after those checks; the new application does not depend on the old folder.

## Suggested training work

1. Establish licensed/approved datasets and subject-disjoint train/validation/test splits. Keep all clips from a person/session together to avoid leakage. Preserve raw annotations and provenance.
2. Include ordinary actions that resemble falls: sitting quickly, kneeling, bending, lying down deliberately, reaching, occlusion and camera shake. Evaluate slow slides, partial visibility, lighting changes and multiple people.
3. Version the pose model and feature schema. Test keypoint indexing and RGB/BGR conversion on known frames.
4. Train a baseline and temporal alternatives. Tune thresholds on validation data only. Fix thresholds before the final test run.
5. Record class imbalance, per-class precision/recall/F1, confusion matrix, fall-event recall, false alarms per hour and time-to-detection. Report hardware, FPS, resolution and latency percentiles.
6. Integrate the chosen detector through the contract. Test normal motion -> warning -> fall -> cooldown -> reviewed event. Track IDs must stay isolated across cameras and people.
7. Deliver model weights separately with checksum, reproducible dependency versions, training entry point/config, dataset/split references, model card and results.

## Included evaluation utility

```powershell
.venv\Scripts\python -m ai.evaluate data/example_predictions.csv --output instance/example-metrics.json
```

Required CSV columns: `clip_id,subject_id,split,y_true,y_pred`. Allowed classes: `normal,warning,fall`; `split` must be `test`. One independent labelled clip/sequence per `(clip_id,subject_id)`. The tool rejects duplicates and train/validation rows. Undefined precision/recall is `null`, not a misleading 100% or zero.

Replace the invented example with predictions from your actual held-out dataset. The tool cannot establish that subjects never occurred during training and does not compute event latency or false alarms per hour. Those require separately annotated event-time and exposure records. Human false-alarm reviews are useful feedback but are not a complete ground-truth dataset: missed falls are absent from alert logs.

## Alert policy already implemented

A `fall` prediction must meet the confidence threshold for N consecutive processed frames for the same camera/person. Missing persons, low confidence, a non-fall label, not-ready state, or gaps over 3 seconds reset the streak. A per-person cooldown limits repeat events. Persistent falls can generate another alert once cooldown expires. These are application policy, separate from the classifier's temporal model; document both.

Warnings are displayed live but are not persisted by this service. Falls are retained with score, model version, evidence, review status and notification records. Demo events bypass inference explicitly and are marked `is_demo=True`.

## Definition of ready for integration review

- A clean machine can install the pinned ML stack and locate weights without hard-coded personal paths.
- RGB frame input produces validated output with real, stable per-person identities.
- The same preprocessing and sampling policy run in training and inference.
- Real clips generate reviewed incidents; missing subjects and ordinary activities do not generate a flood.
- Held-out metrics and failure examples exist with dataset/split and model versions.
- No claimed accuracy, camera FPS, metric units or delivery status is invented.
