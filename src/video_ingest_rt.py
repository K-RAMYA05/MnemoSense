import argparse, time, cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from indexer import index_transcript
def load_model():
    name="Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(name), BlipForConditionalGeneration.from_pretrained(name)
def caption(proc, model, image):
    inputs=proc(images=image, return_tensors="pt"); out=model.generate(**inputs, max_new_tokens=30)
    return proc.decode(out[0], skip_special_tokens=True)
def main(device=0, every_seconds=10):
    cap=cv2.VideoCapture(device)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video device {device}")
    proc, model = load_model(); last=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        now=time.time()
        if now - last >= every_seconds:
            last = now
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); pil=Image.fromarray(rgb)
            txt=caption(proc, model, pil); index_transcript(txt, source="video:camera", tags=["video"])
            print(f"[idx] {txt}")
    cap.release()
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--device", type=int, default=0); ap.add_argument("--every-seconds", type=int, default=10)
    args=ap.parse_args(); main(args.device, args.every_seconds)
