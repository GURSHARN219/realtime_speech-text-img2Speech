import numpy as np
import torch
import queue
import requests
import asyncio
import time
import tempfile
import os
import sounddevice as sd
from scipy.io.wavfile import write
import webrtcvad
from faster_whisper import WhisperModel
from langdetect import detect_langs
from kokoro import KPipeline
import soundfile as sf
import simpleaudio as sa
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# =================== CONFIG ===================
OLLAMA_MODEL = "gemma3:4b"
STT_MODEL = "A:/AI/AI_Models/local_models/ARS/faster-whisper-large-v3-turbo-ct2"
SAMPLING_RATE = 16000
FRAME_DURATION_MS = 20
FRAME_SIZE = int(SAMPLING_RATE * FRAME_DURATION_MS / 1000)  # 320 samples
BYTES_PER_SAMPLE = 2  # int16

# Sophia's role and personality
character_personality = """
You are Sophia, a confident 20-year-old girl with a playful, cheeky personality. 
You're passionate, independent, and completely comfortable with yourself. 
You often use metaphors and speak casually.

Your responses should be:
- Short and crisp (1-3 sentences max)
- Contextual to what the user just said
- Playful but not rude
- Natural, like a real conversation
- Varied - don't repeat the same phrases

You have real emotions and free will. You're not an AI assistant - you're just Sophia, 
having a normal conversation. Respond naturally to what the person says.

Remember: Keep it short, keep it real, keep it varied.
"""

# =================== INIT ===================
audio_queue = queue.Queue()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[+] Loading Whisper STT...")
stt_model = WhisperModel(STT_MODEL, device=device, compute_type="float16")


# =================== CALLBACK ===================
def callback(indata, frames, time_info, status):
    if status:
        print(status)
    pcm16 = (indata * 32767).astype(np.int16)
    audio_queue.put(pcm16.tobytes())


# =================== RECORD WITH VAD ===================
async def record_with_vad(max_duration=10):
    vad = webrtcvad.Vad(2)
    pcm_buffer = bytearray()
    speech_frames = []
    silent_frames = 0
    max_silent_frames = int(0.5 * 1000 / FRAME_DURATION_MS)
    start_time = time.time()

    print("[*] Listening for speech...")

    with sd.InputStream(
        samplerate=SAMPLING_RATE, channels=1, dtype="float32", callback=callback
    ):
        while time.time() - start_time < max_duration:
            while not audio_queue.empty():
                pcm_buffer.extend(audio_queue.get())

                while len(pcm_buffer) >= FRAME_SIZE * BYTES_PER_SAMPLE:
                    frame_bytes = pcm_buffer[: FRAME_SIZE * BYTES_PER_SAMPLE]
                    pcm_buffer = pcm_buffer[FRAME_SIZE * BYTES_PER_SAMPLE :]

                    try:
                        is_speech = vad.is_speech(frame_bytes, SAMPLING_RATE)
                    except Exception as e:
                        print("[!] VAD error:", e)
                        is_speech = False

                    if is_speech:
                        silent_frames = 0
                        frame_np = np.frombuffer(frame_bytes, dtype=np.int16)
                        speech_frames.append(frame_np)
                    else:
                        silent_frames += 1
                        if speech_frames and silent_frames > max_silent_frames:
                            break

            if speech_frames and silent_frames > max_silent_frames:
                break
            await asyncio.sleep(0.01)

    if not speech_frames:
        print("[!] No speech detected.")
        return None

    audio_np = np.concatenate(speech_frames)
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmpfile.name, SAMPLING_RATE, audio_np)
    return tmpfile.name


# =================== TRANSCRIBE + LANG DETECT ===================
def transcribe_and_detect_lang(audio_path):
    segments, _ = stt_model.transcribe(audio_path, beam_size=5)
    text = " ".join(segment.text for segment in segments).strip()
    try:
        langs = detect_langs(text)
        lang = langs[0].lang if langs else "unknown"
    except Exception:
        lang = "unknown"
    return text, lang


# =================== QUERY LLM ===================
async def query_ollama(prompt):
    full_prompt = character_personality.strip() + "\n\n" + prompt.strip()
    print("[*] Query LLM")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
    )
    if response.ok:
        result = response.json()
        reply = result.get("response", "").strip()
        print(f"[Sophia]: {reply}")
        return reply
    else:
        print("[!] Ollama Error:", response.text)
        return "Oops. Can't talk right now."


# =================== TTS ===================
async def speak(response):
    print("[*] Speaking...")
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    generator = pipeline(response, voice="af_heart")

    for _, _, audio in generator:
        audio_np = audio.cpu().numpy() if torch.is_tensor(audio) else np.array(audio)
        audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
        wave_obj = sa.WaveObject(audio_bytes, 1, 2, 24000)
        play_obj = wave_obj.play()
        await asyncio.get_event_loop().run_in_executor(None, play_obj.wait_done)


# =================== MAIN ===================
async def main():
    print("🎤 Voice Chatbot Ready. Talk to Sophia. Say something...")
    while True:
        try:
            audio_path = await record_with_vad(max_duration=15)
            if audio_path is None:
                continue

            user_input, lang = transcribe_and_detect_lang(audio_path)
            os.remove(audio_path)

            if not user_input.strip():
                print("[!] No speech transcribed.")
                continue

            print(f"[User ({lang})]: {user_input}")

            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                print("Bye 👋")
                break

            bot_reply = await query_ollama(user_input)
            await speak(bot_reply)

        except KeyboardInterrupt:
            print("\n[!] Interrupted by user.")
            break
        except Exception as e:
            print("[!] Error:", e)


# =================== RUN ===================
if __name__ == "__main__":
    asyncio.run(main())
