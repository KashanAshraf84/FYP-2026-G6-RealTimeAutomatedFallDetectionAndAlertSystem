/*
 * GuardianAI - Detection pipeline, browser port.
 *
 * A faithful JavaScript port of the Python detection pipeline:
 *   feature_extractor.py  -> FeatureExtractor
 *   inference.py          -> fusePredictions / smoothPredictions
 *   config.py             -> CONFIG
 *
 * Every threshold, weight and counter rule below mirrors the corresponding
 * value in config.py so that the browser demo and the local edge application
 * classify identically. Where the Python code has a known defect (see the
 * fall-trigger note in getRuleBasedStatus) the defect is reproduced rather
 * than silently fixed, so the demo reflects the real system.
 */

// ---------------------------------------------------------------- config
export const CONFIG = {
  // PoseConfig
  minDetectionConfidence: 0.5,
  jointDrawConfidence: 0.4,
  // Indices into YOLOv8's 17 COCO keypoints, matching PoseConfig.CRITICAL_KEYPOINTS
  criticalKeypoints: [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
  maxLostFrames: 10,

  // FeatureConfig
  sequenceLength: 30,
  numRawFeatures: 39,          // 13 keypoints x 3
  numEngineeredFeatures: 12,   // 5 active + 7 reserved
  virtualWidth: 640,
  virtualHeight: 480,

  // Rule thresholds
  warningAngleThreshold: 60,
  lyingAngle: 50,
  groundProximityRatio: 0.6,
  dropSpeedThreshold: 20,
  fallCounterThreshold: 2,
  lyingCounterThreshold: 5,
  hipConfidenceThreshold: 0.4,
  // Below the fall drop-speed threshold but still a sudden jerk/lunge —
  // flags "warning" for a single frame instead of waiting on the 3-of-5
  // smoothing consensus, so brief jerks stay visible.
  jerkSpeedThreshold: 12,

  // SystemConfig
  confidenceThreshold: 0.7,
  nnWeight: 0.7,
  ruleOverrideConfidence: 0.8,
  alertCooldownSeconds: 30,
};

const LABEL_NAMES = ['normal', 'warning', 'fall'];
const STATUS_PRIORITY = { normal: 0, warning: 1, fall: 2 };

// Skeleton edges for the 17-keypoint COCO layout (drawing only)
export const SKELETON = [
  [0, 5], [0, 6], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

// ------------------------------------------------------- feature extractor
export class FeatureExtractor {
  constructor() { this.reset(); }

  reset() {
    this.buffer = [];
    this.prevHeadY = null;       // used for the neural feature vector
    this.prevHeadYRule = null;   // used by the rule engine (kept separate)
    this.prevHeadXRule = null;
    this.fallCounter = 0;
    this.lyingCounter = 0;
    this.lastAngle = 90.0;
    this.lastSpeed = 0.0;
    this.lastMovementSpeed = 0.0;
  }

  /** Port of feature_extractor.extract_frame_features() */
  extractFrameFeatures(kp) {
    const f = [];
    const head = [kp[0][0], kp[0][1]];
    // Torso centre = midpoint of the left/right hips (indices 7 and 8 of the
    // 13 critical keypoints). Using a single hip is far noisier.
    const hip = [(kp[7][0] + kp[8][0]) / 2, (kp[7][1] + kp[8][1]) / 2];

    const dx = hip[0] - head[0];
    const dy = hip[1] - head[1];
    const bodyAngle = Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);
    f.push(bodyAngle / 180.0);

    f.push(head[1]);

    const headYPx = head[1] * CONFIG.virtualHeight;
    const headVelocity = this.prevHeadY !== null ? headYPx - this.prevHeadY : 0.0;
    this.prevHeadY = headYPx;
    f.push(headVelocity / 100.0);

    const xs = kp.map(p => p[0]);
    const ys = kp.map(p => p[1]);
    const wNorm = Math.max(...xs) - Math.min(...xs);
    const hNorm = Math.max(...ys) - Math.min(...ys);
    f.push(Math.min((wNorm / (hNorm + 1e-6)) / 2.0, 1.0));
    f.push(hNorm);

    // 7 reserved dimensions, zero-filled (see SDD Figure 6.1)
    for (let i = 0; i < 7; i++) f.push(0.0);
    return f;
  }

  /** Port of feature_extractor.update(). Returns a Float32Array(30*51) or null. */
  update(kp) {
    let combined;
    if (kp === null) {
      combined = new Array(CONFIG.numRawFeatures + CONFIG.numEngineeredFeatures).fill(0);
    } else {
      const raw = [];
      for (const p of kp) raw.push(p[0], p[1], p[2]);
      combined = raw.concat(this.extractFrameFeatures(kp));
    }

    this.buffer.push(combined);
    if (this.buffer.length > CONFIG.sequenceLength) this.buffer.shift();
    if (this.buffer.length < CONFIG.sequenceLength) return null;

    const flat = new Float32Array(CONFIG.sequenceLength * 51);
    for (let t = 0; t < CONFIG.sequenceLength; t++) {
      for (let j = 0; j < 51; j++) flat[t * 51 + j] = this.buffer[t][j];
    }
    return flat;
  }

  /**
   * Port of feature_extractor.get_rule_based_status().
   *
   * Returns [status, confidence, isInstant] — isInstant is true when
   * "warning" was triggered by a single-frame speed jerk rather than the
   * angle/lying rules, so the caller can surface it immediately instead of
   * waiting on temporal smoothing.
   */
  getRuleBasedStatus(kp) {
    const W = CONFIG.virtualWidth, H = CONFIG.virtualHeight;
    const T = CONFIG.hipConfidenceThreshold;
    const head = [kp[0][0] * W, kp[0][1] * H];

    // --- Hip confidence guard (SRS FR-2.8) ---
    // YOLO estimates hip positions even when the hips are off-frame, which
    // corrupts the body-angle calculation. Fall back to the shoulder
    // midpoint when the hips are not reliably visible.
    const lHipConf = kp[7][2], rHipConf = kp[8][2];
    let hip, hipsReliable;
    if (lHipConf >= T && rHipConf >= T) {
      hip = [((kp[7][0] + kp[8][0]) / 2) * W, ((kp[7][1] + kp[8][1]) / 2) * H];
      hipsReliable = true;
    } else if (lHipConf >= T) {
      hip = [kp[7][0] * W, kp[7][1] * H];
      hipsReliable = true;
    } else if (rHipConf >= T) {
      hip = [kp[8][0] * W, kp[8][1] * H];
      hipsReliable = true;
    } else {
      const lShConf = kp[1][2], rShConf = kp[2][2];
      if (lShConf >= T && rShConf >= T) {
        hip = [((kp[1][0] + kp[2][0]) / 2) * W, ((kp[1][1] + kp[2][1]) / 2) * H];
      } else if (lShConf >= T) {
        hip = [kp[1][0] * W, kp[1][1] * H];
      } else if (rShConf >= T) {
        hip = [kp[2][0] * W, kp[2][1] * H];
      } else {
        hip = head; // last resort: no anchor at all
      }
      hipsReliable = false;
    }

    const angle = Math.abs(Math.atan2(hip[1] - head[1], hip[0] - head[0]) * 180 / Math.PI);
    const headY = head[1], headX = head[0];
    const dropSpeed = this.prevHeadYRule !== null ? headY - this.prevHeadYRule : 0;
    const horizSpeed = this.prevHeadXRule !== null ? headX - this.prevHeadXRule : 0;
    this.prevHeadYRule = headY;
    this.prevHeadXRule = headX;

    // Combined (any-direction) movement magnitude — catches sideways/upward
    // jerks that a vertical-only drop_speed would miss.
    const movementSpeed = Math.hypot(dropSpeed, horizSpeed);
    this.lastMovementSpeed = movementSpeed;

    const nearGround = headY > H * CONFIG.groundProximityRatio;
    const isLying = angle < CONFIG.lyingAngle && nearGround && hipsReliable;

    // NOTE: this reproduces a known defect in the current build - the fall
    // verdict is reachable through the speed trigger alone, without the
    // ground-proximity/angle confirmation the specification requires.
    // Documented in SDD 12.2 and SRS v3 9.4; scheduled for FYP-2.
    const fallTrigger = dropSpeed > CONFIG.dropSpeedThreshold && hipsReliable;

    this.fallCounter = fallTrigger
      ? this.fallCounter + 1
      : Math.max(0, this.fallCounter - 1);
    this.lyingCounter = isLying ? this.lyingCounter + 1 : 0;

    this.lastAngle = angle;
    this.lastSpeed = dropSpeed;

    let status, confidence;
    if (this.fallCounter >= CONFIG.fallCounterThreshold) {
      status = 'fall';
      confidence = Math.min(0.6 + this.fallCounter * 0.1, 1.0);
    } else if (this.lyingCounter >= CONFIG.lyingCounterThreshold) {
      status = 'warning'; confidence = 0.8;
    } else if (angle < CONFIG.warningAngleThreshold && hipsReliable) {
      status = 'warning'; confidence = 0.6;
    } else {
      status = 'normal'; confidence = hipsReliable ? 1.0 : 0.7;
    }

    // --- Instant jerk override ---
    // A sudden movement below the fall drop-speed threshold still counts as
    // "warning" for this frame, even if it lasts a single frame and would
    // otherwise be voted out by temporal smoothing.
    let isInstant = false;
    if (status !== 'fall' && movementSpeed > CONFIG.jerkSpeedThreshold) {
      status = 'warning';
      confidence = Math.max(confidence, 0.65);
      isInstant = true;
    }

    return [status, confidence, isInstant];
  }
}

// ------------------------------------------------------------- fusion
/** Port of inference._fuse_predictions() */
export function fusePredictions(ruleStatus, ruleConf, nnStatus, nnConf) {
  if (nnConf === 0.0) return [ruleStatus, ruleConf];

  // --- Guardrail: physics rule engine says "normal" ---
  // When the physics rule engine detects a normal upright stance (angle
  // above the warning threshold, no rapid drop), trust it over the NN,
  // which may be poorly calibrated or out-of-distribution.
  if (ruleStatus === 'normal') return ['normal', ruleConf];

  const w = CONFIG.nnWeight;
  const rw = 1.0 - w;
  const rulePri = STATUS_PRIORITY[ruleStatus];
  const nnPri = STATUS_PRIORITY[nnStatus];

  let status, conf;
  if (nnPri >= rulePri) {
    status = nnStatus;
    conf = nnConf * w + ruleConf * rw;
  } else if (ruleConf > CONFIG.ruleOverrideConfidence) {
    // Interpretable rules see greater severity and are confident: escalate.
    status = ruleStatus;
    conf = ruleConf * 0.5 + nnConf * 0.5;
  } else {
    status = nnStatus;
    conf = nnConf * w + ruleConf * rw;
  }
  return [status, Math.min(conf, 1.0)];
}

/** Port of inference._smooth_predictions() */
export function smoothPredictions(history, status, confidence) {
  history.push([status, confidence]);
  if (history.length > 10) history.shift();
  if (history.length < 3) return [status, confidence];

  const recent = history.slice(-5);
  const counts = {};
  for (const [s, c] of recent) {
    if (!counts[s]) counts[s] = { count: 0, total: 0 };
    counts[s].count += 1;
    counts[s].total += c;
  }

  if (counts.fall && counts.fall.count >= 3) {
    return ['fall', counts.fall.total / counts.fall.count];
  }
  if (counts.warning && counts.warning.count >= 3) {
    if (!counts.fall || counts.fall.count < 3) {
      return ['warning', counts.warning.total / counts.warning.count];
    }
  }
  let best = null;
  for (const [s, v] of Object.entries(counts)) {
    if (!best || v.count > counts[best].count) best = s;
  }
  return [best, counts[best].total / counts[best].count];
}

// --------------------------------------------------- YOLOv8-pose decoding
/**
 * Decode YOLOv8-pose ONNX output.
 *
 * Output is [1, 56, 4725] laid out channel-major: value(c, i) = data[c*N + i].
 * Per anchor: 0-3 = cx,cy,w,h · 4 = person confidence · 5-55 = 17 keypoints
 * as (x, y, confidence) triples, all in letterboxed input-image space.
 *
 * Only the single highest-confidence anchor is kept, matching the Python
 * single-person estimator which takes the top-scoring detection.
 */
export function decodePose(data, numAnchors, letterbox) {
  let bestIdx = -1, bestConf = CONFIG.minDetectionConfidence;
  for (let i = 0; i < numAnchors; i++) {
    const conf = data[4 * numAnchors + i];
    if (conf > bestConf) { bestConf = conf; bestIdx = i; }
  }
  if (bestIdx < 0) return null;

  const { scale, padX, padY, srcW, srcH } = letterbox;
  const keypoints = [];
  for (let k = 0; k < 17; k++) {
    const base = (5 + k * 3) * numAnchors + bestIdx;
    const x = (data[base] - padX) / scale;
    const y = (data[base + numAnchors] - padY) / scale;
    const c = data[base + 2 * numAnchors];
    // Clamp to the frame, matching Ultralytics' behaviour in the Python
    // pipeline. Without this, a subject cut off by the frame edge yields
    // normalised coordinates above 1.0 and skews the angle calculation.
    keypoints.push([
      Math.min(Math.max(x, 0), srcW),
      Math.min(Math.max(y, 0), srcH),
      c,
    ]);
  }
  return { keypoints, confidence: bestConf, srcW, srcH };
}

/** Reduce 17 keypoints to the 13 critical ones, normalised to [0, 1]. */
export function toCriticalNormalised(keypoints, srcW, srcH) {
  return CONFIG.criticalKeypoints.map(idx => {
    const [x, y, c] = keypoints[idx];
    return [x / srcW, y / srcH, c];
  });
}

/** Softmax over the CNN-LSTM logits. */
export function softmaxArgmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => v / sum);
  let arg = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[arg]) arg = i;
  return [LABEL_NAMES[arg], probs[arg]];
}
