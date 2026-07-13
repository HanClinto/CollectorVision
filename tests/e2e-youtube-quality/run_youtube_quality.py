#!/usr/bin/env python3
"""Run deterministic CollectorVision quality comparisons on YouTube-sampled frames."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class RunSpec:
    label: str
    cornelius_source: str | None
    catalog_source: str


@dataclass(frozen=True)
class RecognitionResult:
    orientation: str | None
    top_hits: list[dict[str, Any]]
    orientation_results: list[dict[str, Any]]

    @property
    def best_card_id(self) -> str | None:
        return self.top_hits[0]["card_id"] if self.top_hits else None

    @property
    def best_score(self) -> float | None:
        return self.top_hits[0]["score"] if self.top_hits else None


def parse_run(spec: str) -> RunSpec:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Bad --run {spec!r}; expected LABEL:CORNELIUS:CATALOG"
        )
    label, cornelius, catalog = parts
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    if not catalog:
        raise argparse.ArgumentTypeError("Catalog source cannot be empty")
    return RunSpec(label=label, cornelius_source=cornelius or None, catalog_source=catalog)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def source_to_url(video_id_or_url: str) -> tuple[str, str]:
    if "://" in video_id_or_url:
        video_id = video_id_or_url.rstrip("/").split("/")[-1]
        if "watch?v=" in video_id_or_url:
            video_id = video_id_or_url.split("watch?v=", 1)[1].split("&", 1)[0]
        return video_id, video_id_or_url
    return video_id_or_url, f"https://www.youtube.com/watch?v={video_id_or_url}"


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Required tool not found on PATH: {name}")
    return path


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def command_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cornelius_path(source: str | None) -> Path:
    if source is None:
        from collector_vision import weights

        return weights.CORNER_DETECTOR
    path = repo_path(source)
    if not path.exists():
        raise SystemExit(f"Cornelius model not found: {path}")
    return path


def format_hits(catalog: Any, hits: list[tuple[float, str]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "card_id": card_id,
            "score": float(score),
            "oracle_id": catalog.card_to_oracle.get(card_id),
        }
        for rank, (score, card_id) in enumerate(hits, start=1)
    ]


def recognize_crop(catalog: Any, crop: Any, top_k: int, rot_invariant: bool) -> RecognitionResult:
    import collector_vision as cvg

    candidates = [("upright", crop)]
    if rot_invariant:
        candidates.append(("rotated_180", cvg.rotate_card_180(crop)))

    embeddings = catalog.embedder.embed([image for _, image in candidates])
    if len(candidates) == 1:
        embeddings = [embeddings]

    orientation_results: list[dict[str, Any]] = []
    for (orientation, _image), embedding in zip(candidates, embeddings):
        top_hits = format_hits(catalog, catalog.search(embedding, top_k=top_k))
        orientation_results.append(
            {
                "orientation": orientation,
                "top_k": top_hits,
                "best_card_id": top_hits[0]["card_id"] if top_hits else None,
                "best_score": top_hits[0]["score"] if top_hits else None,
            }
        )

    best = max(
        orientation_results,
        key=lambda result: result["best_score"] if result["best_score"] is not None else -1.0,
    )
    return RecognitionResult(
        orientation=best["orientation"],
        top_hits=best["top_k"],
        orientation_results=orientation_results,
    )


def prepare_video_frames(args: argparse.Namespace, video_dir: Path) -> tuple[str, list[Path], dict[str, Any]]:
    frames_dir = video_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_meta_path = frames_dir / "frames_meta.json"

    if args.frames_dir:
        source_dir = repo_path(args.frames_dir)
        if not source_dir.exists():
            raise SystemExit(f"Frames directory not found: {source_dir}")
        source_frames = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not source_frames:
            raise SystemExit(f"No image frames found in {source_dir}")
        copied: list[Path] = []
        for index, src in enumerate(source_frames[: args.max_frames or None], start=1):
            dst = frames_dir / f"frame_{index:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            copied.append(dst)
        return "local-frames", copied, {"source_frames_dir": str(source_dir), "commands": []}

    if not args.video_id:
        raise SystemExit("Either --video-id or --frames-dir is required")

    video_id, url = source_to_url(args.video_id)
    downloads_dir = video_dir / "downloads"
    output_template = str(downloads_dir / "source.%(ext)s")
    download_cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        url,
    ]

    existing_frames = sorted(frames_dir.glob("frame_*.jpg"))
    if existing_frames and not args.force_frames:
        frames = existing_frames[: args.max_frames or None]
        reuse_meta = {
            "source_url": url,
            "commands": [download_cmd],
            "reused_frames": True,
            "frame_count": len(frames),
        }
        if frame_meta_path.exists():
            reuse_meta["frames_meta"] = json.loads(frame_meta_path.read_text(encoding="utf-8"))
        else:
            reuse_meta["frames_meta"] = "missing; reused existing frames by filename"
        return video_id, frames, reuse_meta

    require_tool("yt-dlp")
    require_tool("ffmpeg")

    downloads_dir.mkdir(parents=True, exist_ok=True)
    before = set(downloads_dir.glob("source.*"))
    existing = sorted(before)
    if not existing:
        run_command(download_cmd)
    video_files = sorted(downloads_dir.glob("source.*"))
    if not video_files:
        raise SystemExit(f"yt-dlp did not produce a source video in {downloads_dir}")
    video_path = video_files[0]

    for old in existing_frames:
        old.unlink()
    extract_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={args.fps}",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    if args.max_frames:
        extract_cmd[-1:-1] = ["-frames:v", str(args.max_frames)]
    run_command(extract_cmd)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit("ffmpeg did not extract any frames")
    frame_meta_path.write_text(
        json.dumps(
            {
                "source_url": url,
                "video_path": str(video_path),
                "fps": args.fps,
                "max_frames": args.max_frames,
                "frame_count": len(frames),
                "command": extract_cmd,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.discard_video:
        # Keep deterministic extracted frames but allow users to drop large media.
        try:
            video_path.unlink()
        except OSError:
            pass
    return video_id, frames, {"source_url": url, "commands": [download_cmd, extract_cmd]}


def corners_pixels(corners: np.ndarray | None, width: int, height: int) -> list[list[float]] | None:
    if corners is None:
        return None
    scaled = corners * np.array([width, height], dtype=np.float32)
    return [[float(x), float(y)] for x, y in scaled]


def draw_overlay(
    bgr: np.ndarray,
    detection: Any,
    best_card_id: str | None,
    best_score: float | None,
    corner_quality: dict[str, Any] | None = None,
) -> np.ndarray:
    out = bgr.copy()
    detector_card_present = bool(detection.extra.get("detector_card_present", detection.card_present))
    if detector_card_present and detection.corners is not None:
        h, w = out.shape[:2]
        pts = (detection.corners * [w, h]).astype(np.int32)
        quad_rejected = corner_quality is not None and not corner_quality.get("accepted", True)
        color = (0, 128, 255) if quad_rejected else (0, 255, 0)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)
        for i, pt in enumerate(pts, start=1):
            cv2.circle(out, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(out, str(i), tuple(pt + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if quad_rejected:
            label = f"REJECTED quality={corner_quality.get('score'):.2f} {corner_quality.get('reason')}"
            cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif best_card_id:
            label = f"{best_card_id}  {best_score:.3f}" if best_score is not None else best_card_id
            cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        sharpness = detection.sharpness
        label = f"NO CARD sharpness={sharpness:.4f}" if sharpness is not None else "NO CARD"
        cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def rel(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def run_one(
    run_spec: RunSpec,
    frames: list[Path],
    video_id: str,
    fps: float,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    import collector_vision as cvg

    cornelius_path = load_cornelius_path(run_spec.cornelius_source)
    catalog_source = run_spec.catalog_source
    catalog = cvg.Catalog.load(catalog_source)
    detector = cvg.NeuralCornerDetector(cornelius_path, presence_threshold=args.presence_threshold)

    run_dir = output_dir / "runs" / run_spec.label
    overlays_dir = run_dir / "overlays"
    crops_dir = run_dir / "crops"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "label": run_spec.label,
        "cornelius_source": run_spec.cornelius_source,
        "cornelius_path": str(cornelius_path),
        "cornelius_sha256": file_sha256(cornelius_path),
        "catalog_source": catalog_source,
        "catalog_loaded_source": catalog.source,
        "catalog_size": len(catalog),
        "catalog_algo_key": catalog.algo_key,
        "min_sharpness": args.min_sharpness,
        "presence_threshold": args.presence_threshold,
        "min_corner_quality": args.min_corner_quality,
        "top_k": args.top_k,
        "min_score": args.min_score,
        "rot_invariant": args.rot_invariant,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    frame_log_path = run_dir / "frames.jsonl"
    seen: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    with frame_log_path.open("w", encoding="utf-8") as log_f:
        for frame_index, frame_path in enumerate(frames, start=1):
            frame_start = time.perf_counter()
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                record = {
                    "run_label": run_spec.label,
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "frame_path": rel(frame_path, output_dir),
                    "timestamp_sec": (frame_index - 1) / fps,
                    "card_present": False,
                    "error": "cv2.imread returned None",
                }
                log_f.write(json.dumps(record) + "\n")
                records.append(record)
                continue

            h, w = bgr.shape[:2]
            detect_start = time.perf_counter()
            detection = detector.detect(bgr, min_sharpness=args.min_sharpness, min_corner_quality=args.min_corner_quality)
            detect_ms = 1000 * (time.perf_counter() - detect_start)
            corner_quality = {
                "score": detection.extra.get("corner_quality"),
                "accepted": detection.extra.get("corner_quality_accepted"),
                "reason": detection.extra.get("corner_quality_reason"),
                "metrics": detection.extra.get("corner_quality_metrics"),
            }
            detector_card_present = bool(detection.extra.get("detector_card_present", detection.card_present))

            crop_path: Path | None = None
            top_hits: list[dict[str, Any]] = []
            orientation: str | None = None
            orientation_results: list[dict[str, Any]] = []
            best_card_id: str | None = None
            best_score: float | None = None
            timings = {"detect": detect_ms, "dewarp": 0.0, "embed": 0.0, "search": 0.0}
            error: str | None = None
            if detection.card_present:
                try:
                    dewarp_start = time.perf_counter()
                    crop = detection.dewarp(bgr)
                    timings["dewarp"] = 1000 * (time.perf_counter() - dewarp_start)

                    if args.save_crops:
                        crop_path = crops_dir / f"frame_{frame_index:06d}.jpg"
                        crop.save(crop_path)

                    embed_start = time.perf_counter()
                    recognition = recognize_crop(catalog, crop, args.top_k, args.rot_invariant)
                    timings["embed"] = 1000 * (time.perf_counter() - embed_start)
                    top_hits = recognition.top_hits
                    orientation = recognition.orientation
                    orientation_results = recognition.orientation_results
                    best_card_id = recognition.best_card_id
                    best_score = recognition.best_score
                except Exception as exc:
                    error = repr(exc)

            overlay_path = overlays_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(overlay_path), draw_overlay(bgr, detection, best_card_id, best_score, corner_quality))

            total_ms = 1000 * (time.perf_counter() - frame_start)
            timings["total"] = total_ms
            record = {
                "run_label": run_spec.label,
                "video_id": video_id,
                "frame_index": frame_index,
                "frame_path": rel(frame_path, output_dir),
                "timestamp_sec": (frame_index - 1) / fps,
                "card_present": bool(detection.card_present),
                "detector_card_present": detector_card_present,
                "confidence": float(detection.confidence),
                "sharpness": None if detection.sharpness is None else float(detection.sharpness),
                "presence": detection.extra.get("presence"),
                "corner_quality": corner_quality,
                "rejection_reason": None if detection.card_present else corner_quality.get("reason"),
                "corners_normalized": None
                if detection.corners is None
                else [[float(x), float(y)] for x, y in detection.corners],
                "corners_pixels": corners_pixels(detection.corners, w, h),
                "crop_path": rel(crop_path, output_dir),
                "overlay_path": rel(overlay_path, output_dir),
                "orientation": orientation,
                "orientation_results": orientation_results,
                "top_k": top_hits,
                "best_card_id": best_card_id,
                "best_score": best_score,
                "duration_ms": timings,
                "error": error,
            }
            log_f.write(json.dumps(record) + "\n")
            records.append(record)

            if best_card_id and best_score is not None and best_score >= args.min_score:
                entry = seen.get(best_card_id)
                if entry is None:
                    seen[best_card_id] = {
                        "run_label": run_spec.label,
                        "card_id": best_card_id,
                        "oracle_id": catalog.card_to_oracle.get(best_card_id),
                        "first_seen_frame": frame_index,
                        "first_seen_timestamp_sec": record["timestamp_sec"],
                        "last_seen_frame": frame_index,
                        "last_seen_timestamp_sec": record["timestamp_sec"],
                        "count_frames_best_match": 1,
                        "scores": [best_score],
                        "max_score": best_score,
                        "best_orientation": orientation,
                        "best_frame_path": record["frame_path"],
                        "best_overlay_path": record["overlay_path"],
                        "best_crop_path": record["crop_path"],
                    }
                else:
                    entry["last_seen_frame"] = frame_index
                    entry["last_seen_timestamp_sec"] = record["timestamp_sec"]
                    entry["count_frames_best_match"] += 1
                    entry["scores"].append(best_score)
                    if best_score > entry["max_score"]:
                        entry["max_score"] = best_score
                        entry["best_orientation"] = orientation
                        entry["best_frame_path"] = record["frame_path"]
                        entry["best_overlay_path"] = record["overlay_path"]
                        entry["best_crop_path"] = record["crop_path"]

    rows = []
    for entry in sorted(seen.values(), key=lambda e: (e["first_seen_frame"], e["card_id"])):
        scores = entry.pop("scores")
        entry["mean_score"] = sum(scores) / len(scores)
        rows.append(entry)
    write_seen_csv(run_dir / "seen_cards.csv", rows)

    return {"config": run_config, "records": records, "seen_rows": rows}


def write_seen_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_label",
        "card_id",
        "oracle_id",
        "first_seen_frame",
        "first_seen_timestamp_sec",
        "last_seen_frame",
        "last_seen_timestamp_sec",
        "count_frames_best_match",
        "max_score",
        "mean_score",
        "best_orientation",
        "best_frame_path",
        "best_overlay_path",
        "best_crop_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_compare_csv(path: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    combined: dict[str, dict[str, Any]] = {}
    for label, rows in all_rows.items():
        for row in rows:
            card_id = row["card_id"]
            entry = combined.setdefault(
                card_id,
                {"card_id": card_id, "oracle_id": row.get("oracle_id"), "first_seen_any": None},
            )
            first_seen = row.get("first_seen_timestamp_sec")
            if entry["first_seen_any"] is None or first_seen < entry["first_seen_any"]:
                entry["first_seen_any"] = first_seen
            entry[f"{label}_first_seen_timestamp_sec"] = first_seen
            entry[f"{label}_first_seen_frame"] = row.get("first_seen_frame")
            entry[f"{label}_count_frames_best_match"] = row.get("count_frames_best_match")
            entry[f"{label}_max_score"] = row.get("max_score")
            entry[f"{label}_best_orientation"] = row.get("best_orientation")

    labels = list(all_rows)
    fields = ["card_id", "oracle_id", "first_seen_any"]
    for label in labels:
        fields.extend(
            [
                f"{label}_first_seen_timestamp_sec",
                f"{label}_first_seen_frame",
                f"{label}_count_frames_best_match",
                f"{label}_max_score",
                f"{label}_best_orientation",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(combined.values(), key=lambda r: (r["first_seen_any"], r["card_id"])):
            writer.writerow(row)


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    video_id: str,
    frames: list[Path],
    prep_meta: dict[str, Any],
    run_results: dict[str, dict[str, Any]],
) -> None:
    import collector_vision as cvg

    manifest = {
        "video_id": video_id,
        "fps": args.fps,
        "frame_count": len(frames),
        "python": sys.version,
        "collectorvision_version": cvg.__version__,
        "yt_dlp_version": command_output(["yt-dlp", "--version"]),
        "ffmpeg_version": command_output(["ffmpeg", "-version"]),
        "preparation": prep_meta,
        "runs": {label: result["config"] for label, result in run_results.items()},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_viewer(output_dir: Path) -> None:
    viewer_path = Path(__file__).with_name("viewer.py")
    if viewer_path.exists():
        run_command([sys.executable, str(viewer_path), str(output_dir)])


def main() -> None:
    from collector_vision.geometry import DEFAULT_MIN_CORNER_QUALITY

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", help="YouTube video ID or URL")
    parser.add_argument("--frames-dir", type=Path, help="Use an existing image-frame directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate (default: 2)")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=parse_run,
        required=True,
        metavar="LABEL:CORNELIUS:CATALOG",
        help="Scanner run definition. Use an empty Cornelius field for bundled weights.",
    )
    parser.add_argument("--min-sharpness", type=float, default=0.02)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-corner-quality",
        type=float,
        default=DEFAULT_MIN_CORNER_QUALITY,
        help="Minimum 0-1 geometric quality score for detected corners. Use 0 to disable the quality gate.",
    )
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum best score for seen_cards.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-rot-invariant",
        dest="rot_invariant",
        action="store_false",
        help="Only search the dewarped crop instead of also trying a 180-degree rotated copy.",
    )
    parser.add_argument("--max-frames", type=int, help="Optional frame limit for smoke runs")
    parser.add_argument("--force-frames", action="store_true", help="Re-extract frames even when cached frames already exist")
    parser.add_argument("--discard-video", action="store_true", help="Delete downloaded source media after frame extraction")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--no-viewer", action="store_true", help="Skip viewer.html generation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/e2e-youtube-quality")
    parser.set_defaults(rot_invariant=True)
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")

    video_key = args.video_id or f"frames-{repo_path(args.frames_dir).name}"
    video_id, _url = source_to_url(video_key) if args.video_id else (video_key, "")
    output_dir = repo_path(args.output_dir) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id, frames, prep_meta = prepare_video_frames(args, output_dir)
    print(f"Frames: {len(frames)} -> {output_dir / 'frames'}")

    run_results: dict[str, dict[str, Any]] = {}
    seen_by_run: dict[str, list[dict[str, Any]]] = {}
    for run_spec in args.runs:
        if run_spec.label in run_results:
            raise SystemExit(f"Duplicate run label: {run_spec.label}")
        print(f"\nRun: {run_spec.label}")
        result = run_one(run_spec, frames, video_id, args.fps, args, output_dir)
        run_results[run_spec.label] = result
        seen_by_run[run_spec.label] = result["seen_rows"]
        print(f"  seen cards: {len(result['seen_rows'])}")

    write_compare_csv(output_dir / "compare_seen_cards.csv", seen_by_run)
    write_manifest(output_dir / "manifest.json", args, video_id, frames, prep_meta, run_results)
    if not args.no_viewer:
        generate_viewer(output_dir)

    print(f"\nDone: {output_dir}")
    print(f"  compare CSV: {output_dir / 'compare_seen_cards.csv'}")
    viewer_html = output_dir / "viewer.html"
    if viewer_html.exists():
        print(f"  viewer: {viewer_html}")


if __name__ == "__main__":
    main()
