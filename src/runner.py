from orchestrator import Orchestrator
from web_ui import run as run_web
from tts import speak
def main():
    orch=Orchestrator(); orch.start()
    speak("MnemoSense is now running in the background."); run_web(port=5005)
if __name__=="__main__": main()
