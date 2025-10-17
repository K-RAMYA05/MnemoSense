import argparse, os, requests, cv2, time
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from indexer import index_transcript
from utils import now_iso
def load_model():
    name="Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(name), BlipForConditionalGeneration.from_pretrained(name)
def caption(proc, model, image):
    inputs=proc(images=image, return_tensors="pt"); out=model.generate(**inputs, max_new_tokens=30)
    return proc.decode(out[0], skip_special_tokens=True)
def main(url, every_seconds=10, max_frames=200):
    cache=Path("data/tmp/test_video.mp4"); cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        r=requests.get(url, stream=True); r.raise_for_status()
        with open(cache,"wb") as f:
            for ch in r.iter_content(1<<20): f.write(ch)
    cap=cv2.VideoCapture(str(cache)); fps=cap.get(cv2.CAP_PROP_FPS) or 25; step=int(every_seconds*fps)
    proc, model = load_model(); frame_id=0; picked=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_id % step == 0:
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); pil=Image.fromarray(rgb)
            txt=caption(proc, model, pil); payload=f"{txt}"
            index_transcript(payload, source="video:test", tags=["video"])
            print(f"[idx] {payload}"); picked+=1
            if picked>=max_frames: break
        frame_id+=1
    cap.release(); print("[done] indexed video captions.")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--every-seconds", type=int, default=10); ap.add_argument("--max-frames", type=int, default=200)
    args=ap.parse_args(); main(args.url, args.every_seconds, args.max_frames)
