# src/orchestrator.py
import os
import time
import datetime as dt
import threading
import traceback

from utils import load_config
from audio_capture import ContinuousAudioRecorder
from stt import transcribe, save_transcript
from indexer import index_transcript
from safety_checks import run_checks
from forgetting import apply_forgetting
from tts import speak
from summarize import mmr_summarize


def log(msg: str):
    print(msg, flush=True)


def warmup_camera():
    """Ensure macOS camera permission happens on main thread (headless)."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        if opened:
            cap.read()
            cap.release()
        log(f"[warmup] camera opened={opened}")
    except Exception as e:
        log(f"[warmup] camera warmup skipped: {e}")


class Orchestrator:
    def __init__(self):
        self.cfg = load_config()
        self._stop = threading.Event()
        self._rec: ContinuousAudioRecorder | None = None
        self._th_proc: threading.Thread | None = None
        self._th_safety: threading.Thread | None = None
        self._th_forget: threading.Thread | None = None

    def start(self):
        self._stop.clear()

        # --- Start continuous recorder (producer) ---
        win_sec = int(self.cfg.get("audio", {}).get("window_seconds", 120))
        self._rec = ContinuousAudioRecorder(
            window_sec=win_sec,
            device=None,
            keep_webcam=True  # headless; no file saved
        )
        self._rec.start()

        # --- Start consumer and aux threads ---
        self._th_proc = threading.Thread(target=self._loop_consume_and_process, daemon=True)
        self._th_safety = threading.Thread(target=self._loop_safety, daemon=True)
        self._th_forget = threading.Thread(target=self._loop_forgetting, daemon=True)

        log("[MnemoSense] starting threads…")
        self._th_proc.start()
        self._th_safety.start()
        self._th_forget.start()
        log("[MnemoSense] live mode is running (continuous).")

    def stop(self):
        self._stop.set()
        if self._rec:
            self._rec.stop()

    def _loop_consume_and_process(self):
        model_size = self.cfg.get("stt", {}).get("model_size", "base")
        max_chunk_len = self.cfg.get("policies", {}).get("max_chunk_len", 512)
        store_full_text = self.cfg.get("summary", {}).get("store_full_text", False)
        max_sent = self.cfg.get("summary", {}).get("max_sentences", 4)

        log(f"[Live] whisper={model_size} • store_full_text={store_full_text}")

        while not self._stop.is_set():
            try:
                # Wait for next recorded chunk (non-blocking recording continues meanwhile)
                wav = self._rec.get_chunk(timeout=1.0) if self._rec else None
                if not wav:
                    continue

                log(f"[Live] processing {os.path.basename(wav)} …")
                txt, _ = transcribe(wav, model_size=model_size)
                summary = mmr_summarize(txt, max_sentences=max_sent)

                to_store = txt if store_full_text else summary
                log(f"[Live] saving text only (len={len(to_store)})…")
                save_transcript(wav, to_store, [])  # deletes wav internally

                if summary and summary.strip():
                    log("[Live] indexing summary…")
                    index_transcript(summary, source="", tags=[], max_chunk_len=max_chunk_len)
                else:
                    log("[Live] summary empty; skipping index.")
                log("[Live] chunk complete ✓")

            except Exception:
                traceback.print_exc()
                time.sleep(0.5)

    def _loop_safety(self):
        announced = set()
        while not self._stop.is_set():
            try:
                cfg = load_config()
                sch = cfg.get("schedules", {})
                now = dt.datetime.now()
                today = now.date().isoformat()
                for key, rule in sch.items():
                    for t in rule.get("times", []):
                        hh, mm = map(int, t.split(":"))
                        slot = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                        delta = abs((now - slot).total_seconds())
                        tag = (today, key, t)
                        if delta <= 15 * 60 and tag not in announced:
                            msg = run_checks(speak_out=False)
                            if key in msg:
                                speak(f"Reminder: {key} is due around now.")
                                announced.add(tag)
                time.sleep(60)
            except Exception:
                traceback.print_exc()
                time.sleep(60)

    def _loop_forgetting(self):
        last_day = None
        while not self._stop.is_set():
            try:
                now = dt.datetime.now()
                if last_day != now.date() and now.hour == 3:
                    apply_forgetting()
                    speak("Daily cleanup done. Important memories are kept.")
                    last_day = now.date()
                time.sleep(60)
            except Exception:
                traceback.print_exc()
                time.sleep(60)


if __name__ == "__main__":
    # Keep AVFoundation from trying to prompt inside a thread (macOS)
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    try:
        # Ask for camera permission on main thread once
        warmup_camera()

        orch = Orchestrator()
        orch.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        orch.stop()
        log("Stopping…")
 