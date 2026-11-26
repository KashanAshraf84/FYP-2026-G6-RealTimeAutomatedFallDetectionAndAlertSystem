"""
FallNet v2 - Training & Evaluation (FYP-2, Pillar 1)
=====================================================
Trains the temporal pose classifier (normal / warning / fall) on the
GMDCSA-24 keypoint cache and measures it honestly:

  1. Leave-one-subject-out cross-validation (4 folds). The test person is never
     seen in training; early stopping uses a clip-level validation split of the
     training subjects only.
  2. Architecture comparison: Bi-LSTM, CNN-LSTM, Transformer
     (fall_detector_model.py). The architecture is chosen by mean VALIDATION
     macro-F1 - never by test results. Scores are calibrated with temperature
     scaling fitted on the validation split (focal loss leaves them under-confident).
  3. Metrics on the held-out subjects:
       - window level: 3-class confusion matrix, per-class P/R/F1, fall ROC-AUC
       - event level: every test clip is replayed as a live stream (5 FPS, the
         browser rate) through ai/fallnet.py with the application's alert rule
         (fall score >= 0.80 on 3 consecutive frames) -> fall-clip recall,
         false alarms on ADL clips, detection latency
  4. Final model trained on all 4 subjects -> weights/fallnet_v2.pt

    .venv\\Scripts\\python -m training.extract_keypoints      # once, ~20 min CPU
    .venv\\Scripts\\python -m training.train_fallnet          # CV + final model
    .venv\\Scripts\\python -m training.train_fallnet --quick  # smoke test
"""

import argparse
import copy
import csv
import hashlib
import io
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ai.fallnet import LABELS, FallNet, SubjectStream, build_model
from ai.pose_features import FEATURE_NAMES, FEATURE_VERSION, MIN_HISTORY_S, NUM_FEATURES, SAMPLE_HZ, SEQ_LEN, WINDOW_S
from focal_loss import FocalLoss
from training.extract_keypoints import CACHE_DIR
from training.gmdcsa import PROJECT_DIR, load_clips
from training.windows import WindowDataset, label_counts, load_track, raw_features, window_index

RESULTS_DIR = PROJECT_DIR / "training" / "results"
WEIGHTS_PATH = PROJECT_DIR / "weights" / "fallnet_v2.pt"
ARCH_NAMES = {"lstm": "Bi-LSTM", "cnn_lstm": "CNN-LSTM", "transformer": "Transformer"}
EARLY_TOLERANCE_S = 0.5   # an alert this long before the annotated fall start still counts
THRESHOLD_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9]


def model_config(arch, hidden):
    return dict(
        architecture=arch, input_size=NUM_FEATURES, hidden_size=hidden, num_layers=2,
        num_classes=len(LABELS), dropout=0.3, bidirectional=True, cnn_filters=[64, 64],
        cnn_kernel_size=3, num_heads=4, ff_dim=hidden * 2, num_transformer_layers=2,
    )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def clip_split(tracks, val_frac, seed):
    """Stratified (fall / ADL) clip-level split: no clip contributes to both sides."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for kind in ("fall", "adl"):
        group = [t for t in tracks if t.clip.kind == kind]
        n_val = max(1, int(round(val_frac * len(group))))
        for j, gi in enumerate(rng.permutation(len(group))):
            (val if j < n_val else train).append(group[gi])
    return train, val


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_model(arch, train_tracks, train_idx, mean, std, x_val, y_val, args, seed, tag):
    set_seed(seed)
    cfg = model_config(arch, args.hidden)
    model = build_model(cfg)
    # Square-root inverse-frequency class weights: lifts the rare warning/fall
    # windows without making false alarms on normal activity nearly free.
    counts = np.bincount([lbl for *_, lbl in train_idx], minlength=len(LABELS))
    alpha = 1.0 / np.sqrt(np.maximum(counts, 1))
    alpha = alpha / alpha.mean()
    criterion = FocalLoss(alpha=torch.tensor(alpha, dtype=torch.float32), gamma=args.gamma, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    ds = WindowDataset(train_tracks, train_idx, mean, std, augment=True, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=gen)
    xv = torch.from_numpy((x_val - mean) / std).float()
    yv = torch.from_numpy(y_val).long()

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_macro_f1", "val_fall_f1", "lr")}
    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    stale = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        model.eval()
        with torch.no_grad():
            logits = model(xv)
            val_loss = criterion(logits, yv).item()
            pred = logits.argmax(1).numpy()
        macro = f1_score(y_val, pred, average="macro", labels=[0, 1, 2], zero_division=0)
        fall_f1 = f1_score(y_val == 2, pred == 2, zero_division=0)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(macro)
        for k, v in zip(history, (total / seen, 100.0 * correct / seen, val_loss, macro, fall_f1, lr)):
            history[k].append(float(v))

        improved = macro > best["f1"] + 1e-4 or (abs(macro - best["f1"]) <= 1e-4 and val_loss < best["loss"])
        if improved:
            best = {"f1": macro, "loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            stale = 0
        else:
            stale += 1
        if improved or epoch % 5 == 0 or epoch == 1:
            print(f"  [{tag}] ep {epoch:3d} train {total / seen:.4f}/{100.0 * correct / seen:5.1f}% "
                  f"val {val_loss:.4f} macroF1 {macro:.3f} fallF1 {fall_f1:.3f} lr {lr:.1e}"
                  f"{'  *best*' if improved else ''}", flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(best["state"])
    model.eval()
    print(f"  [{tag}] best epoch {best['epoch']} val macroF1 {best['f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return model, cfg, history, best


def fit_temperature(model, x_val, y_val, mean, std):
    """
    Temperature scaling (Guo et al., 2017) on the validation split: one scalar T
    minimising validation NLL. Focal loss leaves scores under-confident; T < 1
    sharpens them. The predicted class never changes.
    """
    with torch.no_grad():
        logits = model(torch.from_numpy((x_val - mean) / std).float()).double()
    y = torch.from_numpy(y_val).long()
    grid = np.exp(np.linspace(np.log(0.05), np.log(5.0), 300))
    nll = [nn.functional.cross_entropy(logits / t, y).item() for t in grid]
    return float(grid[int(np.argmin(nll))])


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def expected_calibration_error(y, probs, bins=15):
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def window_metrics(y, probs):
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    yb, pb = y == 2, pred == 2
    tp, fp = int((yb & pb).sum()), int((~yb & pb).sum())
    fn, tn = int((yb & ~pb).sum()), int((~yb & ~pb).sum())
    both = yb.any() and (~yb).any()
    return {
        "windows": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "expected_calibration_error": expected_calibration_error(y, probs),
        "confusion_matrix_true_rows_pred_cols": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "per_class": {
            LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in range(3)
        },
        "fall_vs_rest": {
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "roc_auc": float(roc_auc_score(yb, probs[:, 2])) if both else None,
            "average_precision": float(average_precision_score(yb, probs[:, 2])) if both else None,
        },
    }


def stream_clip(fallnet, clip, stream_fps):
    """Replay a cached clip through the live code path at `stream_fps`. Returns per-frame (t, probs|None)."""
    with np.load(CACHE_DIR / f"{clip.clip_id}.npz") as d:
        times, det_frame, kps, boxes = d["times"], d["det_frame"], d["kps"], d["boxes"]
        height = int(d["height"])
        duration = float(d["src_frames"] / d["fps"])
    bins = np.floor(times * stream_fps + 1e-6)
    keep = np.r_[True, bins[1:] != bins[:-1]] if stream_fps < SAMPLE_HZ else np.ones(len(times), bool)
    stream = SubjectStream()
    frame_t, windows = [], []
    for f in np.flatnonzero(keep):
        rows = np.flatnonzero(det_frame == f)
        t = float(times[f])
        stream.push(t, kps[rows], boxes[rows], height)
        frame_t.append(t)
        windows.append(stream.window(t))
    have = [i for i, w in enumerate(windows) if w is not None]
    probs = [None] * len(windows)
    if have:
        out = fallnet.predict_windows(np.stack([windows[i] for i in have]))
        for i, p in zip(have, out):
            probs[i] = p
    return frame_t, probs, duration


def alert_times(frame_t, probs, threshold, consecutive, cooldown_s=30.0):
    """monitor/services.ingest rule: label fall with score >= threshold on N consecutive frames."""
    streak, last, alarms = 0, None, []
    for t, p in zip(frame_t, probs):
        if p is not None and int(p.argmax()) == 2 and p[2] >= threshold:
            streak += 1
        else:
            streak = 0
        if streak >= consecutive and (last is None or t - last >= cooldown_s):
            alarms.append(t)
            last = t
    return alarms


def clip_outcome(clip, frame_t, probs, duration, threshold, consecutive):
    alarms = alert_times(frame_t, probs, threshold, consecutive)
    row = {
        "clip_id": clip.clip_id, "subject": clip.subject, "kind": clip.kind,
        "duration_s": round(duration, 2), "alarms_s": [round(a, 2) for a in alarms],
        "max_fall_score": round(max((float(p[2]) for p in probs if p is not None), default=0.0), 4),
    }
    if clip.kind == "fall":
        fs = clip.fall_intervals[0][0]
        on_time = [a for a in alarms if a >= fs - EARLY_TOLERANCE_S]
        row.update(
            fall_start_s=fs,
            detected=bool(on_time),
            early_alarm=any(a < fs - EARLY_TOLERANCE_S for a in alarms),
            latency_s=round(on_time[0] - fs, 2) if on_time else None,
            short_after_fall=duration - fs < 1.6,
        )
    else:
        row.update(false_alarm=bool(alarms))
    return row


def event_metrics(rows):
    falls = [r for r in rows if r["kind"] == "fall"]
    adls = [r for r in rows if r["kind"] == "adl"]
    tp = sum(r["detected"] for r in falls)
    fp = sum(r["false_alarm"] for r in adls)
    lat = sorted(r["latency_s"] for r in falls if r["latency_s"] is not None)
    adl_hours = sum(r["duration_s"] for r in adls) / 3600.0
    pct = lambda q: float(np.percentile(lat, q)) if lat else None
    return {
        "fall_clips": len(falls), "fall_clips_detected": tp,
        "fall_event_recall": tp / max(len(falls), 1),
        "adl_clips": len(adls), "adl_clips_with_false_alarm": fp,
        "adl_specificity": (len(adls) - fp) / max(len(adls), 1),
        "event_precision": tp / max(tp + fp, 1),
        "event_f1": 2 * tp / max(2 * tp + fp + (len(falls) - tp), 1),
        "false_alarms_per_hour_adl": sum(len(r["alarms_s"]) for r in adls) / max(adl_hours, 1e-9),
        "fall_clips_with_alarm_before_fall": sum(r["early_alarm"] for r in falls),
        "latency_s_p50": pct(50), "latency_s_p90": pct(90), "latency_s_mean": float(np.mean(lat)) if lat else None,
        "missed_clips": [r["clip_id"] for r in falls if not r["detected"]],
        "false_alarm_clips": [r["clip_id"] for r in adls if r["false_alarm"]],
    }


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------

def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.55 else "black", fontsize=10)
    ax.set_xticks(range(3), [l.title() for l in LABELS])
    ax.set_yticks(range(3), [l.title() for l in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_history(history, best_epoch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    a.plot(ep, history["train_loss"], label="Train loss")
    a.plot(ep, history["val_loss"], label="Validation loss")
    a.set_xlabel("Epoch")
    a.set_ylabel("Focal loss")
    a.legend()
    b.plot(ep, history["val_macro_f1"], label="Validation macro-F1")
    b.plot(ep, history["val_fall_f1"], label="Validation fall F1")
    b.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    b.set_xlabel("Epoch")
    b.set_ylim(0, 1.02)
    b.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_arch_comparison(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = list(summary)
    metrics = [
        ("Window macro-F1", lambda s: s["window"]["macro_f1"]),
        ("Window fall F1", lambda s: s["window"]["fall_vs_rest"]["f1"]),
        ("Fall-clip recall\n(5 FPS stream)", lambda s: s["event_5fps"]["fall_event_recall"]),
        ("ADL clips without\nfalse alarm", lambda s: s["event_5fps"]["adl_specificity"]),
    ]
    x = np.arange(len(metrics))
    width = 0.8 / len(archs)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    for k, arch in enumerate(archs):
        vals = [fn(summary[arch]) for _, fn in metrics]
        bars = ax.bar(x + (k - (len(archs) - 1) / 2) * width, vals, width, label=ARCH_NAMES[arch])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, [m for m, _ in metrics], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("Leave-one-subject-out results by architecture (held-out subjects)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def state_sha256(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_clip_csv(rows, path):
    keys = ["clip_id", "subject", "kind", "duration_s", "fall_start_s", "detected", "early_alarm",
            "false_alarm", "latency_s", "short_after_fall", "max_fall_score", "alarms_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "alarms_s": " ".join(str(a) for a in r["alarms_s"])})


def write_ai_evaluate_csv(rows, path):
    """Clip-level predictions in the format of ai/evaluate.py (one row per test clip)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "subject_id", "split", "y_true", "y_pred"])
        for r in rows:
            alarm = r["detected"] if r["kind"] == "fall" else r["false_alarm"]
            w.writerow([r["clip_id"], f"S{r['subject']}", "test",
                        "fall" if r["kind"] == "fall" else "normal", "fall" if alarm else "normal"])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def fold_arch_job(job):
    """Train and evaluate one architecture on one LOSO fold (may run in a worker process)."""
    torch.set_num_threads(job["threads"])
    args, arch, s = job["args"], job["arch"], job["subject"]
    model, _, _, best = train_model(
        arch, job["train"], job["train_idx"], job["mean"], job["std"],
        job["x_val"], job["y_val"], args, args.seed + s, f"S{s}/{arch}")
    temperature = fit_temperature(model, job["x_val"], job["y_val"], job["mean"], job["std"])
    fallnet = FallNet.from_parts(model, job["mean"], job["std"], temperature)
    streams = {fps: [(tr.clip, *stream_clip(fallnet, tr.clip, fps)) for tr in job["test"]] for fps in (5, SAMPLE_HZ)}
    return {"arch": arch, "subject": s, "best_epoch": best["epoch"], "val_macro_f1": float(best["f1"]),
            "temperature": temperature, "probs": fallnet.predict_features(job["x_te"]),
            "y_te": job["y_te"], "streams": streams}


def run(args):
    global RESULTS_DIR
    if args.quick:  # never overwrite the real results with a smoke test
        RESULTS_DIR = RESULTS_DIR / "quick"
    torch.set_num_threads(args.threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    clips = load_clips()
    missing = [c.clip_id for c in clips if not (CACHE_DIR / f"{c.clip_id}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} clips have no keypoint cache; run training.extract_keypoints first.")
    tracks = [load_track(c) for c in clips]
    subjects = sorted({c.subject for c in clips})
    all_idx = window_index(tracks)
    print(f"[data] {len(clips)} clips, {sum(len(t.times) for t in tracks)} pose samples, "
          f"{len(all_idx)} windows {label_counts(all_idx)}", flush=True)

    folds = subjects[:1] if args.quick else subjects
    cv = {arch: {"folds": [], "test_y": [], "test_probs": [], "stream": {}} for arch in args.archs}

    # ---------------- Leave-one-subject-out cross-validation ----------------
    if not args.skip_cv:
        jobs = []
        for s in folds:
            test = [t for t in tracks if t.clip.subject == s]
            train, val = clip_split([t for t in tracks if t.clip.subject != s], args.val_frac, args.seed + s)
            train_idx, val_idx, test_idx = window_index(train), window_index(val), window_index(test)
            x_tr, _ = raw_features(train, train_idx)
            flat = x_tr.reshape(-1, NUM_FEATURES)
            mean, std = flat.mean(0), flat.std(0) + 1e-6
            x_val, y_val = raw_features(val, val_idx)
            x_te, y_te = raw_features(test, test_idx)
            print(f"[fold S{s}] test subject S{s}: {len(test)} clips {label_counts(test_idx)} | "
                  f"train {len(train)} clips {label_counts(train_idx)} | val {len(val)} clips", flush=True)
            for arch in args.archs:
                jobs.append(dict(arch=arch, subject=s, train=train, train_idx=train_idx, mean=mean, std=std,
                                 x_val=x_val, y_val=y_val, x_te=x_te, y_te=y_te, test=test, args=args,
                                 threads=max(1, args.threads // max(args.jobs, 1))))

        if args.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
                results = list(pool.map(fold_arch_job, jobs))
        else:
            results = [fold_arch_job(job) for job in jobs]

        for res in results:
            arch, s = res["arch"], res["subject"]
            wm = window_metrics(res["y_te"], res["probs"])
            cv[arch]["test_y"].append(res["y_te"])
            cv[arch]["test_probs"].append(res["probs"])
            for fps, streams in res["streams"].items():
                cv[arch]["stream"].setdefault(fps, []).extend(streams)
            em = event_metrics([clip_outcome(c, ft, pr, du, args.alert_conf, args.alert_frames)
                                for c, ft, pr, du in res["streams"][5]])
            cv[arch]["folds"].append({
                "test_subject": s, "best_epoch": res["best_epoch"], "val_macro_f1": res["val_macro_f1"],
                "temperature": res["temperature"],
                "test_window_macro_f1": wm["macro_f1"], "test_fall_f1": wm["fall_vs_rest"]["f1"],
                "test_fall_event_recall_5fps": em["fall_event_recall"],
                "test_adl_specificity_5fps": em["adl_specificity"],
            })
            print(f"[S{s}/{arch}] TEST window macroF1 {wm['macro_f1']:.3f} fallF1 {wm['fall_vs_rest']['f1']:.3f} | "
                  f"5FPS stream: falls detected {em['fall_clips_detected']}/{em['fall_clips']}, "
                  f"ADL false alarms {em['adl_clips_with_false_alarm']}/{em['adl_clips']}", flush=True)

        summary = {}
        for arch in args.archs:
            y = np.concatenate(cv[arch]["test_y"])
            p = np.concatenate(cv[arch]["test_probs"])
            entry = {
                "mean_val_macro_f1": float(np.mean([f["val_macro_f1"] for f in cv[arch]["folds"]])),
                "folds": cv[arch]["folds"],
                "window": window_metrics(y, p),
            }
            for fps, streams in cv[arch]["stream"].items():
                rows = [clip_outcome(c, ft, pr, du, args.alert_conf, args.alert_frames) for c, ft, pr, du in streams]
                entry[f"event_{fps}fps"] = event_metrics(rows)
                write_clip_csv(rows, RESULTS_DIR / f"cv_clip_results_{arch}_{fps}fps.csv")
                if fps == 5:
                    entry["threshold_sweep_5fps"] = {
                        str(th): {k: v for k, v in event_metrics(
                            [clip_outcome(c, ft, pr, du, th, args.alert_frames) for c, ft, pr, du in streams]
                        ).items() if not k.endswith("_clips")}
                        for th in THRESHOLD_SWEEP
                    }
                    write_ai_evaluate_csv(rows, RESULTS_DIR / f"cv_clip_predictions_{arch}.csv")
            summary[arch] = entry

        chosen = max(args.archs, key=lambda a: summary[a]["mean_val_macro_f1"])
        print("\n[CV] mean validation macro-F1 (selection criterion): " +
              ", ".join(f"{a} {summary[a]['mean_val_macro_f1']:.3f}" for a in args.archs) + f" -> {chosen}", flush=True)
        if len(args.archs) > 1:
            plot_arch_comparison(summary, RESULTS_DIR / "architecture_comparison.png")
        plot_confusion(
            summary[chosen]["window"]["confusion_matrix_true_rows_pred_cols"],
            f"{ARCH_NAMES[chosen]} - held-out subjects (LOSO), 2 s windows",
            RESULTS_DIR / "confusion_matrix_loso.png",
        )
    else:
        summary, chosen = {}, args.archs[0]

    if args.quick:
        print(f"\n[quick] smoke test finished in {(time.time() - started) / 60:.1f} min", flush=True)
        return

    # ---------------- Final model on all subjects ----------------
    train, val = clip_split(tracks, args.val_frac, args.seed)
    train_idx, val_idx = window_index(train), window_index(val)
    x_tr, _ = raw_features(train, train_idx)
    flat = x_tr.reshape(-1, NUM_FEATURES)
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    x_val, y_val = raw_features(val, val_idx)
    print(f"\n[final] {ARCH_NAMES[chosen]} on all subjects: train {len(train)} clips {label_counts(train_idx)}, "
          f"val {len(val)} clips {label_counts(val_idx)}", flush=True)
    model, cfg, history, best = train_model(chosen, train, train_idx, mean, std, x_val, y_val, args, args.seed, "final")
    temperature = fit_temperature(model, x_val, y_val, mean, std)
    val_probs = FallNet.from_parts(model, mean, std, temperature).predict_features(x_val)

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    sha = state_sha256(state)
    model_version = f"fallnet-v2-{chosen.replace('_', '')}-{sha[:10]}"

    # Latency on this machine (feature extraction + classifier for one window).
    fallnet = FallNet.from_parts(model, mean, std, temperature)
    win = np.stack([np.zeros((SEQ_LEN, 17, 3), np.float32) + 0.5])
    for _ in range(10):
        fallnet.predict_windows(win)
    t0 = time.perf_counter()
    for _ in range(200):
        fallnet.predict_windows(win)
    classifier_ms = (time.perf_counter() - t0) / 200 * 1000

    chosen_summary = summary.get(chosen, {})
    report = {
        "model_version": model_version,
        "architecture": chosen,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "selection": "architecture with highest mean validation macro-F1 across LOSO folds (test folds not used)",
        "dataset": "GMDCSA-24: 4 subjects, 79 fall + 81 ADL clips (data/dataset)",
        "protocol": "leave-one-subject-out; early stopping on clip-level validation split of training subjects",
        "labels": LABELS,
        "window_s": WINDOW_S, "seq_len": SEQ_LEN, "sample_hz": SAMPLE_HZ, "min_history_s": MIN_HISTORY_S,
        "alert_rule": {"confidence": args.alert_conf, "consecutive_frames": args.alert_frames, "stream_fps": 5},
        "final_model": {"best_epoch": best["epoch"], "val_macro_f1": best["f1"], "temperature": temperature,
                        "val_window_metrics": window_metrics(y_val, val_probs),
                        "train_windows": label_counts(train_idx), "val_windows": label_counts(val_idx)},
        "classifier_latency_ms_per_window_cpu": round(classifier_ms, 2),
        "hardware": f"{platform.processor() or platform.machine()}, torch {torch.__version__}, {torch.get_num_threads()} threads",
        "cv": summary,
        "args": vars(args),
        "elapsed_min": round((time.time() - started) / 60, 1),
    }
    ckpt = {
        "model_state_dict": state,
        "model_config": cfg,
        "norm_mean": torch.from_numpy(mean.astype(np.float32)),
        "norm_std": torch.from_numpy(std.astype(np.float32)),
        "temperature": temperature,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "labels": LABELS,
        "window_s": WINDOW_S,
        "seq_len": SEQ_LEN,
        "min_history_s": MIN_HISTORY_S,
        "model_version": model_version,
        "state_sha256": sha,
        "trained_at": report["trained_at"],
        "report_json": json.dumps(report),
    }
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    WEIGHTS_PATH.write_bytes(buf.getvalue())
    file_sha = hashlib.sha256(buf.getvalue()).hexdigest()
    report["weights_file"] = str(WEIGHTS_PATH.relative_to(PROJECT_DIR))
    report["weights_file_sha256"] = file_sha

    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS_DIR / "final_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, best["epoch"], RESULTS_DIR / "final_training_curves.png")

    # Sanity check: the saved file loads through the live runtime and reproduces the predictions.
    reloaded = FallNet(WEIGHTS_PATH)
    assert np.allclose(reloaded.predict_features(x_val[:64]), val_probs[:64], atol=1e-5)
    print(f"\n[final] saved {WEIGHTS_PATH} ({model_version}, sha256 {file_sha[:16]}...)", flush=True)
    print(f"[done] {(time.time() - started) / 60:.1f} min. Report: {RESULTS_DIR / 'report.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", nargs="+", default=["lstm", "cnn_lstm", "transformer"], choices=list(ARCH_NAMES))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--alert-conf", type=float, default=0.80, help="Matches DetectionSettings.confidence default")
    p.add_argument("--alert-frames", type=int, default=3, help="Matches DetectionSettings.consecutive_frames default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4, help="Total CPU threads (split across --jobs)")
    p.add_argument("--jobs", type=int, default=3, help="Parallel worker processes for the CV runs")
    p.add_argument("--skip-cv", action="store_true", help="Train only the final model (first --archs entry)")
    p.add_argument("--quick", action="store_true", help="Smoke test: one fold, no final model")
    args = p.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 3)
    run(args)


if __name__ == "__main__":
    main()
