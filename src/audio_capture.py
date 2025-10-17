import sounddevice as sd, soundfile as sf, numpy as np, time
from utils import RECS, now_iso, safe_mkdir
RATE=16000; CHANNELS=1
def record_push_to_talk(duration=None, rms_thresh=0.006, silence_secs=2.0, max_secs=60):
    safe_mkdir(RECS); ts = now_iso().replace(":","-"); wav = RECS/f"note_{ts}.wav"
    if duration:
        audio=sd.rec(int(duration*RATE), samplerate=RATE, channels=CHANNELS, dtype="float32"); sd.wait()
    else:
        print("[rec] Press Enter to start; auto-stops on silence."); input()
        buf=[]; start=time.time(); last_voice=time.time()
        stream=sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="float32", blocksize=2048); stream.start()
        try:
            while True:
                block,_=stream.read(2048); buf.append(block.copy())
                rms=np.sqrt(np.mean(block**2)); 
                if rms>rms_thresh: last_voice=time.time()
                if (time.time()-last_voice)>silence_secs or (time.time()-start)>max_secs: break
        finally: stream.stop(); stream.close()
        audio=np.concatenate(buf,axis=0)
    sf.write(str(wav), audio, RATE); print(f"[rec] Saved {wav}"); return str(wav)
