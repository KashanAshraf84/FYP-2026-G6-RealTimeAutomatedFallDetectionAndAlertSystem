"""
GMDCSA-24 Clip Index
====================
Lists every clip in data/dataset with its subject and the annotated fall
interval(s) parsed from the per-subject Fall.csv / ADL.csv files.

Expected layout (as shipped with the dataset):
    data/dataset/falls/Subject<N>/Fall.csv  + falls/Subject<N>/Fall/<NN>.mp4
    data/dataset/adl/Subject<N>/ADL.csv     + adl/Subject<N>/ADL/<NN>.mp4

The "Classes" column holds hand-typed time ranges in seconds, e.g.
"Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]". A few rows contain typos
("Falling (BW[5.9 to 10]", "[2.3 6]") or omit the fall range entirely; the
parser tolerates the typos and records an explicit note for the omission.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "dataset"

# "Falling (SW)[3.4 to 6]", "Fall (FW)[0 to 3]", "Falling (BW[5.9 to 10]", "Falling (BW) [0.3 to 5]"
_FALL_SEGMENT = re.compile(r"\bFall\w*\s*(?:\([^\[\];]*?\)?)?\s*\[([^\]\[]*)\]", re.IGNORECASE)
# "3.4 to 6" and the shorthand "2.3 6"
_INTERVAL = re.compile(r"(\d+(?:\.\d+)?)\s*(?:to\s*|\s+)(\d+(?:\.\d+)?)")

# A fall range ending this close to the annotated clip length runs to the end of the video.
_END_TOLERANCE_S = 0.15


@dataclass(frozen=True)
class Clip:
    clip_id: str                # "S2-fall-10"
    subject: int                # 1..4
    kind: str                   # "fall" or "adl"
    path: Path
    annotated_length_s: float
    description: str
    annotation: str             # raw "Classes" cell
    # (start_s, end_s); end None = until the end of the video. Empty for ADL clips.
    fall_intervals: Tuple[Tuple[float, Optional[float]], ...] = ()
    notes: str = ""


def parse_fall_intervals(classes: str, annotated_length: float):
    """Return merged fall intervals and a note describing any assumption made."""
    intervals = []
    for seg in _FALL_SEGMENT.finditer(classes):
        for a, b in _INTERVAL.findall(seg.group(1)):
            intervals.append((float(a), float(b)))

    notes = ""
    if not intervals and re.search(r"\bfall", classes, re.IGNORECASE):
        # Subject4 Fall 15: "Falling (FW); Sitting[0 to 1.7]" -> fall begins when the last activity ends.
        others = [float(b) for _, b in _INTERVAL.findall(classes)]
        start = max(others) if others else 0.0
        intervals.append((start, annotated_length))
        notes = f"Fall range missing in CSV; assumed {start:g}s to clip end."

    intervals.sort()
    merged: List[Tuple[float, float]] = []
    for a, b in intervals:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    out = tuple(
        (a, None if b >= annotated_length - _END_TOLERANCE_S else b) for a, b in merged
    )
    return out, notes


def load_clips(dataset_dir: Path = DATASET_DIR) -> List[Clip]:
    """Index every annotated clip. Raises if a video has no annotation row."""
    clips = []
    layout = (
        ("fall", "falls", "Fall", "Fall.csv"),
        ("adl", "adl", "ADL", "ADL.csv"),
    )
    for kind, top, video_dir, csv_name in layout:
        for subj_dir in sorted((dataset_dir / top).glob("Subject*")):
            subject = int(subj_dir.name.replace("Subject", ""))
            rows = {}
            with open(subj_dir / csv_name, newline="", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.reader(f)
                next(reader)  # header
                for row in reader:
                    if row and row[0].strip():
                        rows[row[0].strip()] = row

            for video in sorted((subj_dir / video_dir).glob("*.mp4")):
                row = rows.get(video.name)
                if row is None:
                    raise ValueError(f"No annotation row for {video}")
                length = float(row[1].strip())
                classes = row[-1].strip()
                description = ",".join(row[4:-1]).strip()
                intervals, notes = (
                    parse_fall_intervals(classes, length) if kind == "fall" else ((), "")
                )
                if kind == "fall" and not intervals:
                    raise ValueError(f"Could not parse a fall range for {video}: {classes!r}")
                clips.append(
                    Clip(
                        clip_id=f"S{subject}-{kind}-{video.stem}",
                        subject=subject,
                        kind=kind,
                        path=video,
                        annotated_length_s=length,
                        description=description,
                        annotation=classes,
                        fall_intervals=intervals,
                        notes=notes,
                    )
                )
    return clips


if __name__ == "__main__":
    all_clips = load_clips()
    for c in all_clips:
        if c.kind == "fall":
            print(f"{c.clip_id:<14} {c.fall_intervals}  {c.notes}")
    n_fall = sum(c.kind == "fall" for c in all_clips)
    print(f"\n{len(all_clips)} clips: {n_fall} fall, {len(all_clips) - n_fall} ADL, "
          f"subjects {sorted({c.subject for c in all_clips})}")
