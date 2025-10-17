# src/video_ingest_rt.py
import cv2
import sounddevice as sd
import numpy as np
import tempfile, queue, time, wave
from datetime import datetime

AUDIO_RATE = 16000
WINDOW_SEC = 120  # 2 minutes

def capture_live_chunk():
    """Capture 2 minutes from webcam + mic, no file save, return temporary wav path."""
    print("[Live] Capturing live chunk from webcam + mic…")
    cam = cv2.VideoCapture(0)
    q_audio = queue.Queue()
    frames_audio = []

    def audio_cb(indata, frames, time_info, status):
        q_audio.put(indata.copy())

    stream = sd.InputStream(samplerate=AUDIO_RATE, channels=1, callback=audio_cb)
    stream.start()

    start = time.time()
    while time.time() - start < WINDOW_SEC:
        ret, _ = cam.read()
        if not ret:
            continue
        while not q_audio.empty():
            frames_audio.append(q_audio.get_nowait())
        cv2.waitKey(1)

    stream.stop()
    cam.release()
    cv2.destroyAllWindows()

    # combine audio buffers
    audio_data = np.concatenate(frames_audio, axis=0)
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp_wav.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    print(f"[Live] Audio chunk saved: {tmp_wav.name}")
    return tmp_wav.name, datetime.utcnow().isoformat()
