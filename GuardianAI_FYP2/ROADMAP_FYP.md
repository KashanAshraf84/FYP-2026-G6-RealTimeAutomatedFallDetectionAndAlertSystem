# 🛡️ GuardianAI — FYP-2 Engineering & Research Roadmap
> **Real-Time AI Fall Detection & CCTV Monitoring System**  
> *Target Milestones, Architecture Upgrades, Team Task Breakdown & Evaluation Plan*

---

## 📌 Executive Summary

Our Final Year Project brings together **Computer Vision (YOLOv8-Pose)**, **Temporal Sequence Modeling (CNN-LSTM / Transformer)**, and a **Production Web Dashboard (GuardianAI)**. 

To achieve high academic grades and a production-grade system, our core focus is divided into **four pillars**:
1. **AI Training & Dataset Engineering**
2. **Advanced Model Architectures & Multi-Person Tracking**
3. **Accuracy, False-Alarm Reduction & Clinical Immobility Verification**
4. **CCTV / RTSP Hardware Connectivity & Real-Time Video Streaming**

---

## 🗺️ High-Level System Architecture

```mermaid
flowchart TD
    subgraph Inputs ["1. Video Sources"]
        CCTV["CCTV / IP Camera\n(RTSP Stream: 15-30 FPS)"]
        Webcam["Browser Webcam\n(Local WebRTC / Canvas)"]
        Files["Test Video Files\n(MP4 / AVI Clips)"]
    end

    subgraph Ingestion ["2. High-Performance Stream Buffer"]
        Reader["Threaded RTSP Reader\n(Zero-Lag FIFO Queue)"]
        FrameResize["Preprocessing & Resize\n(960px max width)"]
    end

    subgraph AIPipeline ["3. AI Fall Detection Engine"]
        YOLO["YOLOv8-Pose Estimator\n(17 Keypoints per person)"]
        Tracker["Multi-Person Tracker\n(ByteTrack / Unique Person IDs)"]
        Features["Feature Extractor\n(Angles, Velocities, Aspect Ratios)"]
        TemporalModel["Temporal Classifier\n(Bi-LSTM / CNN-LSTM / Transformer)"]
        PostFall["Post-Fall Verification\n(Lying Duration & Immobility Check)"]
    end

    subgraph Platform ["4. GuardianAI Platform"]
        Ingest["Event Ingestion Service"]
        DB[("Database\n(Incidents & Audit Logs)")]
        UI["Web Dashboard\n(Live Telemetry & Mosaic)"]
        Alerts["Alert Dispatch\n(Audio, Desktop & Email)"]
    end

    CCTV --> Reader --> FrameResize
    Webcam --> FrameResize
    Files --> FrameResize

    FrameResize --> YOLO --> Tracker --> Features --> TemporalModel --> PostFall
    PostFall --> Ingest
    Ingest --> DB
    Ingest --> UI
    Ingest --> Alerts
```

---

## 🎯 The 4 Core Technical Pillars

### Pillar 1: AI Training & Dataset Engineering
**Goal:** Build a robust, reproducible dataset pipeline that eliminates overfitting and generalizes to different camera angles.

- [ ] **1.1 Dataset Standardization & Sourcing:**
  - Collect and format standard public benchmarks:
    - **UR Fall Detection Dataset (URFD)** (front and overhead angles).
    - **Le2i Fall Detection Dataset** (covers home, coffee room, office, lecture room).
    - **Multiple Cameras Fall Dataset (MCFD)**.
  - Convert all raw video datasets into standardized NumPy keypoint sequence arrays (`(N, 30, 51)`: 30 frames, 17 keypoints $(x, y, c)$).
- [ ] **1.2 Data Augmentation:**
  - **Temporal Jittering:** Randomly skip/duplicate 1–2 frames to simulate FPS fluctuations.
  - **Horizontal Flipping:** Mirror keypoint coordinates horizontally (swapping left and right joints).
  - **Scale & Shift Noise:** Inject Gaussian noise $(\mu=0, \sigma=0.02)$ into keypoint coordinates to simulate tracking noise.
- [ ] **1.3 Class Imbalance Resolution:**
  - Normal everyday actions (walking, sitting, standing) vastly outnumber falls (10:1 ratio).
  - Implement **Focal Loss** ($\gamma=2.0$) or class-weighted cross-entropy to penalize missed falls heavily without causing false alarms.
- [ ] **1.4 Experiment Tracking:**
  - Log training loss, validation loss, F1-score, and learning rate curves into JSON/TensorBoard for project presentation graphs.

---

### Pillar 2: Model Architectures & Multi-Person Tracking
**Goal:** Transition from single-person heuristics to robust multi-person temporal deep learning.

- [ ] **2.1 Multi-Person Tracking Integration (Crucial for CCTV):**
  - Current limitation: The baseline takes only the largest single person.
  - Solution: Integrate **ByteTrack** or YOLO's built-in tracker (`persist=True`).
  - Maintain a separate 30-frame rolling buffer for *each tracked individual* (`track_id`).
- [ ] **2.2 Architecture Comparison Study:**
  - Benchmark three models on the same test set:
    1. **Baseline Bi-LSTM:** Fast, low compute, good baseline.
    2. **CNN-LSTM (Hybrid):** 1D CNNs extract spatial-temporal joint features; LSTM models sequence history (Recommended).
    3. **Spatial-Temporal Transformer (ST-Transformer):** Self-attention over joint tokens across time.
- [ ] **2.3 ONNX & TensorRT Export for Real-Time Inference:**
  - Export PyTorch models (`.pth`) to ONNX (`.onnx`) and FP16 format for $\ge 30\text{ FPS}$ throughput on laptops and edge devices.

---

### Pillar 3: Accuracy & False-Alarm Elimination
**Goal:** Ensure the system does not trigger false alerts when subjects sit down quickly, tie their shoes, or lie on a couch.

- [ ] **3.1 Two-Stage Verification Logic:**
  - **Stage 1 (Kinematic Trigger):** Sudden spike in downward vertical velocity $(v_y > \text{threshold})$ + torso tilt angle dropping below $30^\circ$ relative to horizontal.
  - **Stage 2 (Post-Fall Immobility Check):**
    - When Stage 1 fires, enter a `WARNING` state.
    - If the subject recovers/stands within $3\text{ seconds}$ $\rightarrow$ **Cancel alert** (classified as "Sitting" or "Tripped and Recovered").
    - If the subject remains prone/immobile on the floor for $> 5\text{ seconds}$ $\rightarrow$ **Confirm `FALL` incident**.
- [ ] **3.2 Bed / Sofa Exclusion Zones (ROI Masking):**
  - Allow drawing bounding polygons in the UI to designate beds or sofas where lying down is normal behavior.
- [ ] **3.3 Quantitative Evaluation Report:**
  - Generate Confusion Matrix, Precision-Recall Curve, ROC curve ($AUC \ge 0.95$).
  - Measure inference latency per frame (ms).

---

### Pillar 4: CCTV & RTSP Camera Connectivity
**Goal:** Connect commercial IP cameras / CCTV systems directly into GuardianAI without frame delay.

- [ ] **4.1 Zero-Lag Threaded RTSP Reader:**
  - Standard `cv2.VideoCapture` accumulates network buffers and creates multi-second video lag.
  - Build a background worker thread (`ThreadedRTSPStream`) that continuously reads frames and discards stale buffer, ensuring inference always gets the *latest* frame.
- [ ] **4.2 Auto-Reconnect with Exponential Backoff:**
  - Handle camera power cuts, Wi-Fi drops, and RTSP stream resets gracefully without crashing the server.
- [ ] **4.3 UI-Configurable RTSP Streams:**
  - Add fields in the Django Camera form for:
    - RTSP Stream URL (`rtsp://user:pass@192.168.1.100:554/stream1`)
    - Frame sampling rate (e.g., 5 FPS, 10 FPS, 15 FPS)
    - Camera zone / location tag
- [ ] **4.4 Web Dashboard Live MJPEG Stream:**
  - Provide a live video preview feed directly on the web monitor wall alongside telemetry data.

---

## 👥 FYP Team Member Responsibilities & Task Distribution

| Team Member | Primary Role | Key Deliverables |
|---|---|---|
| **Member 1: AI & ML Lead** | Model Architecture & Training | • Train CNN-LSTM and Transformer models<br>• Implement Focal Loss & Data Augmentation<br>• Model export to ONNX & benchmark reports |
| **Member 2: Computer Vision Lead** | Multi-Person Tracking & Biomechanics | • ByteTrack integration for multi-person tracking<br>• Post-fall immobility verification algorithm<br>• False positive reduction (sitting/bending filters) |
| **Member 3: System & Hardware Lead** | CCTV / RTSP Ingestion & Edge Deployment | • Zero-lag threaded RTSP video stream worker<br>• Network auto-reconnect logic<br>• Camera hardware integration & testing |
| **Member 4: Full-Stack Platform Lead** | GuardianAI Web App & Alerting | • Live MJPEG video stream integration on dashboard<br>• Camera CRUD & settings UI<br>• Incident audit logs & audio/email alert dispatch |

---

## 📅 Sprint Schedule & Milestones

```mermaid
gantt
    title FYP-2 Project Timeline & Milestones
    dateFormat  YYYY-MM-DD
    section Sprint 1: Foundation
    Multi-Person Tracking (ByteTrack)   :s1, 2026-10-21, 7d
    Threaded RTSP CCTV Stream Worker     :s2, 2026-10-21, 7d
    section Sprint 2: Accuracy & AI
    Dataset Pipeline & Augmentation     :s3, 2026-10-28, 7d
    Train CNN-LSTM with Focal Loss      :s4, 2026-10-28, 7d
    Post-Fall Immobility Logic          :s5, 2026-11-04, 7d
    section Sprint 3: Integration
    Live CCTV Stream on Web Dashboard   :s6, 2026-11-04, 7d
    Settings & Confidence Tuning UI     :s7, 2026-11-11, 7d
    section Sprint 4: Testing & Final
    End-to-End Stress & Latency Tests   :s8, 2026-11-18, 7d
    Academic Paper & Presentation Demo  :s9, 2026-11-25, 7d
```

---

## 🎓 Academic Defense & Presentation Checklist

For our project evaluation committee and external examiners, we must prepare:
1. **Live Demonstration:**
   - Real-time webcam demonstration showing both a **fall** (alert triggered) and a **false-alarm challenge** (e.g., sitting quickly on a chair $\rightarrow$ no false alarm).
   - Live CCTV / IP camera connected via RTSP.
2. **Metrics & Charts:**
   - Confusion Matrix (Normal vs. Warning vs. Fall).
   - Precision, Recall, and F1-Score breakdown ($\ge 92\%$ target on held-out test sets).
   - Inference latency chart ($\le 50\text{ ms}$ per frame on GPU/CPU).
3. **Architecture Diagrams:**
   - Pipeline flow diagram (from raw CCTV frame to emergency notification).
   - System deployment diagram.

---

*Document generated for the GuardianAI FYP Engineering Team. Keep updated as sprints progress.*
