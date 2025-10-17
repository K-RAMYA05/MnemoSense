from utils import load_config
from audio_capture import record_push_to_talk
from stt import transcribe, save_transcript
from tts import speak
from indexer import index_transcript
from qa import answer
from summarize import mmr_summarize
def cmd_log():
    cfg=load_config(); wav=record_push_to_talk(); print("[stt] Transcribing...")
    text,_=transcribe(wav, model_size=cfg.get("stt",{}).get("model_size","base"))
    print(f"[stt] Text: {text}"); summary=mmr_summarize(text, max_sentences=cfg.get("summary",{}).get("max_sentences",4))
    print(f"[summary] {summary}"); save_transcript(wav, summary, [])
    _, n = index_transcript(summary, source=wav, tags=[], max_chunk_len=cfg.get("policies",{}).get("max_chunk_len",512))
    speak("Noted. I saved a concise memory."); print(f"[done] Indexed {n} chunk(s).")
def cmd_ask():
    cfg=load_config(); speak("Ask me now. Press Enter to start and stop."); wav=record_push_to_talk()
    speak("Got it. Let me check your notes."); text,_=transcribe(wav, model_size=cfg.get("stt",{}).get("model_size","base"))
    print(f"[query] {text}"); reply,_=answer(text, top_k=6); speak(reply); print("\nAnswer:\n", reply)
if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("mode", choices=["log","ask"]); args=ap.parse_args()
    if args.mode=="log": cmd_log()
    else: cmd_ask()
