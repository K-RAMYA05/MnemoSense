import subprocess
from utils import load_config
def speak(text: str):
    voice = load_config().get("tts",{}).get("voice","Samantha")
    try: subprocess.run(["say","-v",voice, text], check=True)
    except Exception: subprocess.run(["say", text])
