Here's your content converted into a proper `README.md` format, ready to be used in your GitHub repository:

---

````markdown
# 🎙️ Voice Chatbot "Sophia" - College Project

Sophia is a real-time **voice chatbot** designed for interactive conversations. It uses optimized local models for STT, TTS, and LLM response generation — all running efficiently on CPU/GPU.

---

## 🔧 1. Overview

Sophia can:
- 🎤 Listen to human speech using **Voice Activity Detection (VAD)**
- 📝 Transcribe speech to text using **Faster-Whisper**
- 💬 Generate responses via **Gemma3:4b** (local LLM)
- 🗣️ Convert text replies into natural-sounding voice using **Kokoro TTS**
- 💻 Runs **fully offline** on your local system (GPU/CPU)

---

## 🧠 2. System Architecture

```mermaid
graph TD
A[🎙️ Microphone] --> B{🔎 VAD Filter}
B -- Speech Detected --> C[📝 Whisper STT]
B -- Silence --> E[🔁 Re-Listen]
C --> D[🌍 Language Detection]
D --> F[🧠 Gemma LLM]
F --> G[🗣️ Kokoro TTS]
G --> H[🔊 Speaker]
````

---

## ⚙️ 3. Key Components

### 3.1 🔇 Voice Activity Detection (VAD)

* **Purpose**: Filters background noise; captures only actual speech.
* **Code Sample**:

```python
vad = webrtcvad.Vad(2)  # Aggressiveness level (0–3)
is_speech = vad.is_speech(audio_frame, SAMPLING_RATE)
```

### 3.2 📝 Speech-to-Text (Whisper)

* **Model**: `faster-whisper-large-v3-turbo-ct2`
* **Code Sample**:

```python
segments, _ = stt_model.transcribe(audio.wav, beam_size=5)
text = " ".join(segment.text for segment in segments)
```

### 3.3 🌍 Language Detection

* **Library**: `langdetect`
* **Code Sample**:

```python
langs = detect_langs(text)  # e.g., [en:0.998, fr:0.002]
primary_lang = langs[0].lang
```

### 3.4 🧠 Response Generation (Gemma LLM)

* **Prompt Injection**:

```python
full_prompt = f"{character_personality}\n\n{user_input}"
```

* **API Call**:

```python
requests.post("http://localhost:11434/api/generate", 
              json={"model": "gemma3:4b", "prompt": full_prompt})
```

### 3.5 🗣️ Text-to-Speech (Kokoro)

* **Pipeline**:

```python
pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
generator = pipeline(response, voice="af_heart")
```

* **Playback**: Uses `simpleaudio` for low-latency audio output.

---

## 🔁 4. Workflow

1. 🎧 Microphone → VAD → Collect valid audio frames
2. 📝 Save → Whisper STT → Detect language
3. 💬 Inject personality → Query LLM (Gemma)
4. 🗣️ Generate TTS → Play output

---

## 🧩 5. Setup Guide

### 📦 Dependencies

```bash
pip install numpy torch sounddevice webrtcvad faster-whisper langdetect simpleaudio
git clone https://github.com/hexgrad/Kokoro-82M  # For Kokoro TTS
```

### 🖥️ Hardware

* GPU Recommended (for Whisper and Kokoro)
* Microphone + Speakers

### ▶️ Run the Chatbot

1. Start Ollama LLM server:

```bash
ollama serve  # Gemma runs at http://localhost:11434
```

2. Run the chatbot:

```bash
python sophia_chatbot.py
```

---

## ⚡ 6. Performance Metrics

| Component   | Latency (GPU) | Latency (CPU) |
| ----------- | ------------- | ------------- |
| VAD         | < 10ms        | < 10ms        |
| Whisper STT | \~1.2s        | \~4.5s        |
| Gemma LLM   | \~0.8s        | \~3.1s        |
| Kokoro TTS  | \~0.5s        | \~1.8s        |

---

## 🛠️ 7. Customization

* **Change Personality**: Edit the `character_personality` string
* **Swap Models**:

  * Use smaller `faster-whisper` variants (e.g., `tiny.en`)
  * Change `OLLAMA_MODEL` to `llama3`, `mistral`, etc.
* **Voice Options**: Use `af_heart`, `jp_005`, or any supported by Kokoro

---

## 🧪 8. Troubleshooting

| Issue              | Solution                                                  |
| ------------------ | --------------------------------------------------------- |
| No speech detected | Use `Vad(1)` instead of `Vad(2)`                          |
| Ollama error       | Ensure `gemma3:4b` is downloaded: `ollama pull gemma3:4b` |
| CUDA OOM           | Use smaller Whisper model or set `compute_type="int8"`    |

---

## 🚀 9. Future Enhancements

* Wake-word detection ("Hey Sophia")
* Add memory (contextual conversation)
* Multilingual support (output speech in input language)

---

### 👨‍🎓 Project by: **Gursharn Singh**

**Course**: BTech CSE (AIML)
**GitHub**: [GURSHARN219/realtime\_speech-text-img2Speech](https://github.com/GURSHARN219/realtime_speech-text-img2Speech.git)

```

---

Let me know if you want it saved as a `.md` file or want to embed project screenshots, badges, or demo GIFs for a more polished GitHub presentation.
```
