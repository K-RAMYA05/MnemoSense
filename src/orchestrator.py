import threading, time, datetime as dt, traceback, os
from utils import load_config
from audio_capture import record_push_to_talk
from stt import transcribe, save_transcript
from indexer import index_transcript
from safety_checks import run_checks
from forgetting import apply_forgetting
from tts import speak
from summarize import mmr_summarize
class Orchestrator:
    def __init__(self): self.cfg = load_config(); self._stop = threading.Event()
    def start(self):
        self._stop.clear()
        self.th_log=threading.Thread(target=self._loop_continuous_log, daemon=True)
        self.th_safety=threading.Thread(target=self._loop_safety, daemon=True)
        self.th_forget=threading.Thread(target=self._loop_forgetting, daemon=True)
        self.th_log.start(); self.th_safety.start(); self.th_forget.start()
    def stop(self): self._stop.set()
    def _loop_continuous_log(self):
        while not self._stop.is_set():
            try:
                wav=record_push_to_talk(duration=None, rms_thresh=0.006, silence_secs=2.0, max_secs=60)
                txt,_=transcribe(wav, model_size=self.cfg.get("stt",{}).get("model_size","base"))
                summary=mmr_summarize(txt, max_sentences=self.cfg.get("summary",{}).get("max_sentences",4))
                to_store = summary if not self.cfg.get("summary",{}).get("store_full_text",False) else txt
                save_transcript(wav, to_store, [])
                index_transcript(summary, source=wav, tags=[], max_chunk_len=self.cfg.get("policies",{}).get("max_chunk_len",512))
                if self.cfg.get("summary",{}).get("delete_raw_audio_after_index", True):
                    try: os.remove(wav)
                    except: pass
            except Exception: traceback.print_exc(); time.sleep(2)
    def _loop_safety(self):
        announced=set()
        while not self._stop.is_set():
            try:
                cfg=load_config(); sch=cfg.get("schedules",{})
                now=dt.datetime.now(); today=now.date().isoformat()
                for key, rule in sch.items():
                    for t in rule.get("times", []):
                        hh,mm=map(int,t.split(":")); slot=now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                        delta=abs((now-slot).total_seconds()); tag=(today,key,t)
                        if delta<=15*60 and tag not in announced:
                            msg=run_checks(speak_out=False)
                            if key in msg: speak(f"Reminder: {key} is due around now."); announced.add(tag)
                time.sleep(60)
            except Exception: traceback.print_exc(); time.sleep(60)
    def _loop_forgetting(self):
        last_day=None
        while not self._stop.is_set():
            try:
                now=dt.datetime.now()
                if last_day != now.date() and now.hour == 3:
                    apply_forgetting(); speak("Daily cleanup done. Important memories are kept."); last_day=now.date()
                time.sleep(60)
            except Exception: traceback.print_exc(); time.sleep(60)
