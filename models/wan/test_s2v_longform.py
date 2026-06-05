import json
import math
import os
import tempfile
import unittest
import wave
import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("s2v_longform.py")
_SPEC = importlib.util.spec_from_file_location("s2v_longform", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

ensure_ffmpeg_available = _MOD.ensure_ffmpeg_available
plan_s2v_chunks = _MOD.plan_s2v_chunks
run_longform_job = _MOD.run_longform_job


def _write_sine_wav(path: str, seconds: float = 2.0, sample_rate: int = 16000) -> None:
    total = int(seconds * sample_rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for n in range(total):
            value = int(32767 * math.sin(2.0 * math.pi * 220.0 * (n / sample_rate)))
            wf.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


class S2VLongformTests(unittest.TestCase):
    def test_plan_two_hour_audio(self):
        plan = plan_s2v_chunks(7200.0, 120.0, 2.0)
        self.assertGreaterEqual(len(plan.chunks), 55)
        self.assertLessEqual(len(plan.chunks), 65)
        self.assertEqual(plan.chunks[0].index, 1)
        self.assertAlmostEqual(plan.chunks[0].start_seconds, 0.0)

    def test_dry_run_job_writes_state_and_chunks(self):
        try:
            ensure_ffmpeg_available()
        except Exception:
            self.skipTest("ffmpeg/ffprobe not available")

        with tempfile.TemporaryDirectory() as td:
            audio_path = os.path.join(td, "input.wav")
            out_dir = os.path.join(td, "job")
            _write_sine_wav(audio_path, seconds=6.0)

            state = run_longform_job(
                output_root=out_dir,
                source_audio=audio_path,
                prompt="test prompt",
                model_type="s2v-14B",
                resolution="1024x704",
                fps=24.0,
                chunk_seconds=2.0,
                overlap_seconds=0.0,
                requested_duration_seconds=0.0,
                continuity_mode="independent",
                stop_on_chunk_failure=True,
                resume=True,
                do_concat=False,
                preserve_audio_chunks=True,
                dry_run=True,
                render_chunk=lambda *_args, **_kwargs: "",
                render_context={},
            )

            self.assertGreaterEqual(len(state.get("completed_chunks", [])), 3)
            state_path = os.path.join(out_dir, "job_state.json")
            self.assertTrue(os.path.isfile(state_path))
            with open(state_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded.get("total_chunks"), len(loaded.get("completed_chunks", [])))


if __name__ == "__main__":
    unittest.main()
