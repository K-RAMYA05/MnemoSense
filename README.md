MnemoSense: An Artificial Hippocampus for Dementia Patients
“Helping people remember, stay safe, and live with dignity.”

- Overview
MnemoSense is a cognitive-assistive AI system designed to support individuals with dementia, Alzheimer’s, or memory loss.
Inspired by the hippocampus — the brain’s memory center — MnemoSense acts as an external memory companion that continuously observes, understands, and remembers daily life.
A wearable device captures short segments of video and audio, analyzes the surroundings, and transcribes only the meaningful content — not the raw footage.
It then creates rich contextual summaries that include what happened, who was involved, and what was discussed.
When the user speaks to it, MnemoSense can:
Recall what happened, who they interacted with, and what they talked about
Provide spoken reminders for medication, meals, and safety
Offer situational awareness (where they are, what’s around them)
Respond verbally, acting like a kind, always-present companion
By merging LLMs, speech processing, and situational AI, MnemoSense functions as an artificial hippocampus — helping memory-impaired users remain oriented, autonomous, and safe.

- Core Idea
“Instead of recording your life, it remembers the meaning of it.”
Unlike surveillance-based systems that store raw footage, MnemoSense captures 2-minute multimodal (audio + video) windows, transcribes the dialogue, detects context and participants, and stores a semantic summary instead of the full data.
Each memory entry contains:
Who was present (faces or voices recognized)
Where the user was (room, indoor/outdoor context)
What was discussed (topic-level conversational summary)
What actions occurred (activities, reminders, or events)
This turns the device into a privacy-preserving personal historian — capable of telling users what they did, who they met, and what they talked about, anytime they ask.

- Technical Architecture
System Flow
Continuous Multimodal Capture
Captures short synchronized video + audio segments every 120 seconds via webcam or wearable sensors.
Performs lightweight situational awareness (scene type, people nearby, ambient conditions).
Transcription + Conversation Understanding
Processes speech using OpenAI Whisper (STT).
Extracts key topics and conversational intent, summarizing what was said and by whom.
Merges conversation and scene information into a single context-rich summary.
Semantic Embedding + Vector Storage
Converts summaries into embeddings using Sentence-Transformers.
Stores these in a FAISS vector database, forming a searchable “memory space.”
Raw video/audio is deleted — only meaning remains.
Query → Recall → Response Loop
The user asks, “Who did I talk to today?” or “What did I discuss with my doctor?”
The query is embedded and compared against the vector database to retrieve the most relevant “memories.”
The top results are passed to GPT-4o-mini, which composes a natural, coherent answer.
The answer is spoken back using TTS, enabling full voice-in → voice-out recall.

- Tech Stack
Frontend / UI -	Flask + Vanilla JS (Voice recording & playback)
Video / Audio Capture -	OpenCV · SoundDevice · ffmpeg-python
Speech Recognition (STT) -	OpenAI Whisper
Conversation Summarization -	MMR-based text selection + LLM-assisted dialogue abstraction
Situational Awareness -	OpenCV (scene detection / face cues / motion context)
Embeddings & Retrieval -	Sentence-Transformers · FAISS Vector DB
LLM Reasoning -	OpenAI GPT-4o-mini
Voice Output (TTS) -	macOS say / pyttsx3
Backend Orchestration -	Python (continuous threaded ingestion + Flask UI)
Data Handling -	YAML configs · JSONL transcripts · NumPy vector storage

- Example Interactions
- Memory Recall
User: “Who did I talk to today?”
MnemoSense: “You spoke with your friend Arjun in the afternoon about your doctor’s visit and evening plans.”
- Situational Awareness
User: “Where am I right now?”
MnemoSense: “You’re in the living room near the window. The TV is on, and someone is talking to you from the kitchen.”
- Smart Reminder
MnemoSense: “It’s 8 PM — time for your evening medicine.”
- Privacy by Design
No raw media stored — only text summaries and encrypted embeddings.
All processing runs locally on the device (edge-first).
User-controlled deletion and retention policies.

- How to Run
# Clone repository
git clone https://github.com/K-RAMYA05/MnemoSense.git
cd MnemoSense-main

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install faiss-cpu sentence-transformers opencv-python ffmpeg-python

# Configure OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini

# Start continuous memory ingestion
python -m src.continuous_ingest

# Launch interactive web interface
python -m src.web_ui
Then open http://localhost:5005 — click 🎙️, speak your query, and MnemoSense will listen, recall, and reply.
- Future Directions
- Multi-person recognition + conversation tracking.
- Context-adaptive reminders based on activity recognition.
- Real-time hazard detection / wandering alerts.
- Edge deployment on Raspberry Pi / Jetson Nano.
- Integration with biosensors for emotion and stress inference.
- Personalized long-term memory graph for caregivers and clinicians.

- Acknowledgments:
Developed as part of Good Vibes Only! — An AI/ML Buildathon and Networking Event (#LATechWeek), driven by the vision to build AI for accessibility.
MnemoSense is inspired by the mission to restore awareness, autonomy, and dignity to those living with memory loss.
Built with ❤️ using open-source AI, signal processing, and human-centered design.

