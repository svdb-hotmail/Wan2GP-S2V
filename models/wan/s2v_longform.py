import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

def _resolve_media_binary(name: str) -> str | None:
    env_map = {"ffmpeg": "FFMPEG_BINARY", "ffprobe": "FFPROBE_BINARY"}
    env_value = os.environ.get(env_map.get(name, ""), "").strip()
    if env_value:
        return env_value
    return shutil.which(name)


@dataclass
class S2VChunk:
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass
class S2VLongformPlan:
    total_duration_seconds: float
    chunk_seconds: float
    overlap_seconds: float
    chunks: list[S2VChunk]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_ffmpeg_available() -> tuple[str, str]:
    ffmpeg_path = _resolve_media_binary("ffmpeg")
    ffprobe_path = _resolve_media_binary("ffprobe")
    if ffmpeg_path is None or ffprobe_path is None:
        raise RuntimeError("ffmpeg/ffprobe not found in PATH.")
    return ffmpeg_path, ffprobe_path


def get_audio_duration_seconds(audio_path: str) -> float:
    _, ffprobe_path = ensure_ffmpeg_available()
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        os.fspath(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or f"ffprobe failed for {audio_path}")
    payload = json.loads(result.stdout or "{}")
    fmt = payload.get("format", {}) if isinstance(payload, dict) else {}
    duration = fmt.get("duration", None)
    if duration is None:
        raise RuntimeError(f"Unable to read duration for audio file: {audio_path}")
    return float(duration)


def plan_s2v_chunks(total_duration_seconds: float, chunk_seconds: float, overlap_seconds: float) -> S2VLongformPlan:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    chunks: list[S2VChunk] = []
    step = chunk_seconds - overlap_seconds
    start = 0.0
    idx = 1
    while start < total_duration_seconds:
        end = min(total_duration_seconds, start + chunk_seconds)
        chunks.append(
            S2VChunk(
                index=idx,
                start_seconds=start,
                end_seconds=end,
                duration_seconds=max(0.0, end - start),
            )
        )
        idx += 1
        start += step

    return S2VLongformPlan(
        total_duration_seconds=total_duration_seconds,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        chunks=chunks,
    )


def extract_audio_chunk(source_audio: str, output_audio: str, start_seconds: float, duration_seconds: float) -> None:
    ffmpeg_path, _ = ensure_ffmpeg_available()
    cmd = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        os.fspath(source_audio),
        "-map",
        "0:a:0",
        os.fspath(output_audio),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "ffmpeg failed while splitting audio")


def estimate_required_disk_bytes(total_duration_seconds: float, width: int, height: int, fps: float = 24.0, bits_per_pixel: float = 0.15) -> int:
    # Conservative compressed video estimate + audio + metadata overhead.
    bits = total_duration_seconds * fps * width * height * bits_per_pixel
    return int(bits / 8.0) + int(total_duration_seconds * 32000) + 128 * 1024 * 1024


def concat_chunks(chunk_paths: list[str], output_path: str) -> None:
    if len(chunk_paths) == 0:
        raise RuntimeError("No chunk videos available to concatenate")
    ffmpeg_path, _ = ensure_ffmpeg_available()
    list_path = output_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in chunk_paths:
            f.write(f"file '{os.path.abspath(p).replace("'", "'\\''")}'\n")
    cmd = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback to re-encode when stream copy is incompatible.
        fallback = [
            ffmpeg_path,
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            output_path,
        ]
        retry = subprocess.run(fallback, capture_output=True, text=True)
        if retry.returncode != 0:
            raise RuntimeError(retry.stderr or result.stderr or "ffmpeg concat failed")


def run_longform_job(
    *,
    output_root: str,
    source_audio: str,
    prompt: str,
    model_type: str,
    resolution: str,
    fps: float,
    chunk_seconds: float,
    overlap_seconds: float,
    requested_duration_seconds: float,
    continuity_mode: str,
    stop_on_chunk_failure: bool,
    resume: bool,
    do_concat: bool,
    preserve_audio_chunks: bool,
    dry_run: bool,
    render_chunk: Callable[[S2VChunk, str, dict], str],
    render_context: dict,
) -> dict:
    os.makedirs(output_root, exist_ok=True)
    chunks_dir = os.path.join(output_root, "chunks")
    previews_dir = os.path.join(output_root, "previews")
    final_dir = os.path.join(output_root, "final")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(previews_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    ensure_ffmpeg_available()

    audio_duration = get_audio_duration_seconds(source_audio)
    total_duration = min(audio_duration, requested_duration_seconds) if requested_duration_seconds > 0 else audio_duration
    plan = plan_s2v_chunks(total_duration, chunk_seconds, overlap_seconds)

    state_path = os.path.join(output_root, "job_state.json")
    config_path = os.path.join(output_root, "job_config.json")

    state = {
        "job_id": os.path.basename(output_root),
        "total_duration_seconds": total_duration,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "total_chunks": len(plan.chunks),
        "current_chunk": 0,
        "completed_chunks": [],
        "failed_chunks": [],
        "final_concat_status": "pending" if do_concat else "disabled",
        "resume": bool(resume),
        "updated_at": _now_iso(),
    }

    if resume and os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            state.update(loaded)

    config = {
        "model_type": model_type,
        "prompt": prompt,
        "source_audio": source_audio,
        "resolution": resolution,
        "fps": fps,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "continuity_mode": continuity_mode,
        "stop_on_chunk_failure": stop_on_chunk_failure,
        "preserve_audio_chunks": preserve_audio_chunks,
        "dry_run": dry_run,
        "created_at": _now_iso(),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    completed_videos: list[str] = []
    completed_set = set(state.get("completed_chunks", []))

    for chunk in plan.chunks:
        name = f"chunk_{chunk.index:04d}"
        chunk_video = os.path.join(chunks_dir, f"{name}.mp4")
        chunk_audio = os.path.join(chunks_dir, f"{name}.wav")
        chunk_json = os.path.join(chunks_dir, f"{name}.json")

        if chunk.index in completed_set and os.path.isfile(chunk_video):
            completed_videos.append(chunk_video)
            continue

        started_at = _now_iso()
        chunk_meta = {
            "chunk_index": chunk.index,
            "start_seconds": chunk.start_seconds,
            "end_seconds": chunk.end_seconds,
            "audio_segment": chunk_audio,
            "video_output": chunk_video,
            "prompt": prompt,
            "model_task": model_type,
            "resolution": resolution,
            "fps": fps,
            "settings": {
                "chunk_seconds": chunk_seconds,
                "overlap_seconds": overlap_seconds,
                "continuity_mode": continuity_mode,
            },
            "status": "running",
            "start_timestamp": started_at,
            "end_timestamp": None,
            "error_message": None,
        }

        try:
            extract_audio_chunk(source_audio, chunk_audio, chunk.start_seconds, chunk.duration_seconds)
            if dry_run:
                time.sleep(0.01)
                with open(chunk_video, "wb") as f:
                    f.write(b"")
            else:
                rendered_path = render_chunk(chunk, chunk_audio, render_context)
                if rendered_path != chunk_video and os.path.isfile(rendered_path):
                    shutil.copyfile(rendered_path, chunk_video)

            chunk_meta["status"] = "completed"
            chunk_meta["end_timestamp"] = _now_iso()
            state["completed_chunks"] = sorted(set(state.get("completed_chunks", []) + [chunk.index]))
            completed_videos.append(chunk_video)
        except Exception as exc:
            chunk_meta["status"] = "failed"
            chunk_meta["error_message"] = str(exc)
            chunk_meta["end_timestamp"] = _now_iso()
            state["failed_chunks"] = sorted(set(state.get("failed_chunks", []) + [chunk.index]))
            with open(chunk_json, "w", encoding="utf-8") as f:
                json.dump(chunk_meta, f, indent=2)
            state["current_chunk"] = chunk.index
            state["updated_at"] = _now_iso()
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            if stop_on_chunk_failure:
                raise
            continue

        with open(chunk_json, "w", encoding="utf-8") as f:
            json.dump(chunk_meta, f, indent=2)
        state["current_chunk"] = chunk.index
        state["updated_at"] = _now_iso()
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        if not preserve_audio_chunks and os.path.isfile(chunk_audio):
            os.remove(chunk_audio)

    if do_concat and len(completed_videos) > 0:
        final_video = os.path.join(final_dir, "final_concat.mp4")
        try:
            state["final_concat_status"] = "running"
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            concat_chunks(completed_videos, final_video)
            state["final_concat_status"] = "completed"
            state["final_output"] = final_video
        except Exception as exc:
            state["final_concat_status"] = "failed"
            state["final_concat_error"] = str(exc)
    state["updated_at"] = _now_iso()
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return state
