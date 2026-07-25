# Deployment

Two deployment variants of GuardianAI are maintained here. The local edge
application in [`../src`](../src) remains the primary system; both variants
below exist to make the project demonstrable from a URL.

---

## `huggingface-static/` — live demo (deployed)

**URL:** https://kashan84-guardianai.static.hf.space

Runs **entirely in the visitor's browser** using ONNX Runtime Web. There is no
server-side component: pose estimation and fall classification both execute in
the browser tab, and no frame is ever uploaded.

| | |
|---|---|
| Hosting | Hugging Face **static** Space (free tier) |
| Pose model | `yolov8n-pose-480.onnx` — 12.7 MB |
| Classifier | `fall_detector.onnx` (CNN-LSTM) — 2.9 MB |
| Runtime | ONNX Runtime Web (WebGPU where available, WASM otherwise) |
| Detection logic | `pipeline.js` — a line-for-line port of `feature_extractor.py` and the fusion/smoothing logic in `inference.py` |

### Why in-browser rather than server-side

Beyond being the only option on Hugging Face's free tier, this preserves the
project's core privacy property: GuardianAI is specified as an edge system in
which video never leaves the premises. A server-side demo would have inverted
that; running in the browser keeps it intact.

### Verification

The JavaScript port was validated against the Python implementation on a
45-frame synthetic fall sequence spanning body angles from 10° to 90°.
Rule verdicts, fused status, confidence, angle, speed and the first five
feature-vector elements matched to within 1e-4 on **all 45 frames**.

The ONNX decode (channel-major `[1, 56, 4725]` indexing plus letterbox
inversion) was verified by rendering the decoded keypoints onto a test frame
and confirming they align with the subject.

### Regenerating the ONNX models

```bash
cd src
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='onnx', imgsz=480, opset=12)"
python main.py export --model models/fall_detector_best.pth --output models/fall_detector.onnx
```

### Redeploying

```bash
hf auth login
python -c "from huggingface_hub import upload_folder; upload_folder(repo_id='Kashan84/guardianai', repo_type='space', folder_path='deploy/huggingface-static')"
```

---

## `huggingface-space/` — server-side variant (built, not deployed)

A Docker-based Flask deployment in which the browser sends frames to the
server for inference. It imports the detection pipeline **unmodified** from the
local application and relies on the `headless` configuration profile added to
`AlertConfig` to suppress the buzzer, OpenCV popup and OS notification on hosts
with no desktop or audio device.

**Not currently deployed:** Hugging Face now requires a PRO subscription to
host Docker or Gradio Spaces; only static Spaces remain free. The variant is
retained because it is verified working and can be deployed to any Docker host
(a PRO Space, Render, Railway, Fly.io, Cloud Run, or a self-hosted machine).

Locally verified at ~8.5 fps steady state, with per-visitor detector instances
isolating concurrent viewers' temporal buffers.

```bash
cd deploy/huggingface-space
pip install -r requirements.txt
python app.py        # http://127.0.0.1:7860
```

---

## Comparison

| | Local (`src/`) | Static demo | Docker variant |
|---|---|---|---|
| Camera | Server-attached webcam | Visitor's browser | Visitor's browser |
| Inference | Local Python | Visitor's browser | Server |
| Video leaves device | Never | Never | Yes, to the server |
| Alerts | Buzzer, popup, OS notification, e-mail, JSON log | Audio tone + on-screen | On-screen + JSON log |
| Throughput | 25–30 FPS | Depends on client hardware | ~8.5 fps |
| Hosting cost | — | Free | Requires a paid host |
