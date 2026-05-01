"""
Voice Chat Client
-----------------
Speak into your mic → Whisper transcribes → AI answers → optional TTS reads it back.

Usage:
    python voice_chat.py              # text responses only
    python voice_chat.py --tts        # AI reads answer aloud
    python voice_chat.py --model finetuned  # use fine-tuned model
"""

import argparse
import sys
import time

import numpy as np
import requests
import sounddevice as sd
import whisper

API_BASE    = "http://localhost:8000"
SAMPLE_RATE = 16000   # Whisper expects 16 kHz

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tts",   action="store_true", help="Read AI answer aloud")
parser.add_argument("--model", choices=["base", "finetuned"], default=None,
                    help="Override active model")
parser.add_argument("--seconds", type=int, default=5,
                    help="Recording duration in seconds (default 5)")
args = parser.parse_args()

# ── Load Whisper tiny ─────────────────────────────────────────────────────────
print("Loading Whisper tiny model...")
whisper_model = whisper.load_model("tiny")
print("Whisper ready.\n")

# ── Optional TTS engine ───────────────────────────────────────────────────────
tts_engine = None
if args.tts:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 175)
    print("TTS enabled — AI will speak responses.\n")

# ── Switch model if requested ─────────────────────────────────────────────────
if args.model:
    r = requests.post(f"{API_BASE}/model?name={args.model}")
    print(f"Switched to model: {r.json()['active_model']}\n")

def get_active_model() -> str:
    return requests.get(f"{API_BASE}/model").json()["active_model"]

def record(seconds: int) -> np.ndarray:
    """Record from default microphone and return float32 numpy array at 16 kHz."""
    print(f"  Recording {seconds}s — speak now...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("  Done recording.")
    return audio.flatten()

def transcribe(audio: np.ndarray) -> str:
    """Run Whisper tiny on a float32 numpy array — no ffmpeg needed."""
    result = whisper_model.transcribe(audio, fp16=False, language="en")
    return result["text"].strip()

def ask_ai(question: str) -> dict:
    """Send question to FastAPI and return full response dict."""
    r = requests.get(f"{API_BASE}/chat", params={"prompt": question}, timeout=120)
    return r.json()

def speak(text: str):
    """Read text aloud using pyttsx3 (only if --tts flag is set)."""
    if tts_engine:
        tts_engine.say(text)
        tts_engine.runAndWait()

# ── Main loop ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Voice Chat — Local AI")
print(f"  Active model : {get_active_model()}")
print(f"  Recording    : {args.seconds}s per question")
print(f"  TTS          : {'ON' if args.tts else 'OFF'}")
print("=" * 55)
print("Press Enter to record | type 'quit' to exit\n")

turn = 0
while True:
    try:
        cmd = input("[ Press Enter to speak ] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if cmd == "quit":
        print("Goodbye!")
        break

    turn += 1
    print(f"\n--- Turn {turn} ---")

    # 1. Record
    audio = record(args.seconds)

    # 2. Transcribe
    print("  Transcribing...")
    t0 = time.time()
    question = transcribe(audio)
    transcribe_time = round(time.time() - t0, 2)

    if not question:
        print("  (nothing heard — try again)\n")
        continue

    print(f"  You said   : \"{question}\"  [{transcribe_time}s]")

    # 3. Ask AI
    print("  Thinking...")
    response = ask_ai(question)
    answer   = response.get("response", "No response")
    latency  = response.get("latency", "?")
    model    = response.get("model", "?")

    print(f"  AI [{model}] ({latency}s) :\n")
    print(f"  {answer}\n")

    # 4. Speak (optional)
    speak(answer)
