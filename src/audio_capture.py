# src/audio_capture.py
import os
import time
import threading
import queue
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

from .utils import get_temp_wav_path


RATE = 16000
CHANNELS = 1
BLOCKSIZE = 2048  # audio frames per read


class ContinuousAudioRecorder:
    """
    Producer that continuously records microphone audio into fixed windows and
    (optionally) keeps the webcam session alive (headless) so the LED is on.
    Each completed window is written to a TEMP .wav and queued for processing.

    Usage:
        rec = ContinuousAudioRecorder(window_sec=120, device=None, keep_webcam=True)
        rec.start()
        path = rec.get_chunk(timeout=... )  # temp .wav path
        rec.stop()
    """
    def __init__(self, window_sec: int = 120, device=None, keep_webcam: bool = True):
        self.window_sec = int(window_sec)
        self.device = device
        self.keep_webcam = keep_webcam

        self._q: "queue.Queue[str]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._cap = None
        if self.keep_webcam:
            try:
                self._cap = cv2.VideoCapture(0)
                if not self._cap.isOpened():
                    print("[warn] Webcam not accessible, proceeding with audio only.", flush=True)
                    self._cap = None
                else:
                    try:
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[warn] Webcam init failed: {e}", flush=True)
                self._cap = None

    def start(self):
        self._stop.clear()
        self._thread.start()
        print(f"[rec] Continuous recorder started (window={self.window_sec}s)", flush=True)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        print("[rec] Continuous recorder stopped.", flush=True)

    def get_chunk(self, timeout: float | None = None) -> str | None:
        """Return the next temp wav path, or None on timeout/stop."""
        if self._stop.is_set():
            try:
                return self._q.get_nowait()
            except queue.Empty:
                return None
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        """Recording loop — keeps running and queues a temp wav per window."""
        while not self._stop.is_set():
            tmp_wav = get_temp_wav_path(prefix="livewindow")
            audio_buf = []
            start = time.time()

            try:
                with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="float32",
                                    device=self.device, blocksize=BLOCKSIZE) as stream:
                    while (time.time() - start) < self.window_sec and not self._stop.is_set():
                        block, _ = stream.read(BLOCKSIZE)
                        audio_buf.append(block.copy())

                        if self._cap is not None:
                            # keep camera session active (no GUI, no saving)
                            self._cap.grab()

                        # tiny sleep to avoid busy loop
                        time.sleep(0.001)
            except Exception as e:
                print(f"[rec] audio stream error: {e}", flush=True)

            # Write window to temp wav (silence fallback if needed)
            if audio_buf:
                audio = np.concatenate(audio_buf, axis=0)
            else:
                audio = np.zeros((RATE, CHANNELS), dtype="float32")

            try:
                sf.write(tmp_wav, audio, RATE)
                # Non-blocking queue put; if full, drop the oldest to keep up
                try:
                    self._q.put_nowait(tmp_wav)
                except queue.Full:
                    try:
                        old = self._q.get_nowait()
                        if os.path.exists(old):
                            os.remove(old)
                    except Exception:
                        pass
                    self._q.put_nowait(tmp_wav)
                print(f"[rec] window ready → {tmp_wav}", flush=True)
            except Exception as e:
                print(f"[rec] failed to write temp wav: {e}", flush=True)


def record_push_to_talk(duration=None, rms_thresh=0.006, silence_secs=2.0, max_secs=60) -> str:
    """
    Manual 'push-to-talk' recorder (legacy mode).
    Saves to persistent recordings directory; auto-stops on silence.
    NOTE: Live mode should prefer ContinuousAudioRecorder (no media storage).
    """
    from utils import RECS, now_iso, safe_mkdir

    safe_mkdir(RECS)
    ts = now_iso().replace(":", "-")
    wav = RECS / f"note_{ts}.wav"

    if duration:
        audio = sd.rec(int(duration * RATE), samplerate=RATE,
                       channels=CHANNELS, dtype="float32")
        sd.wait()
    else:
        print("[rec] Press Enter to start; auto-stops on silence.")
        input()
        buf = []
        start = time.time()
        last_voice = time.time()
        with sd.InputStream(samplerate=RATE, channels=CHANNELS,
                            dtype="float32", blocksize=BLOCKSIZE) as stream:
            while True:
                block, _ = stream.read(BLOCKSIZE)
                buf.append(block.copy())
                rms = float(np.sqrt(np.mean(block ** 2)))
                if rms > rms_thresh:
                    last_voice = time.time()
                if (time.time() - last_voice) > silence_secs or (time.time() - start) > max_secs:
                    break
        audio = np.concatenate(buf, axis=0)

    sf.write(str(wav), audio, RATE)
    print(f"[rec] Saved {wav}", flush=True)
    return str(wav)
