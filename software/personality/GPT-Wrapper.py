#Time and sound libraries
import time
import pyaudio
import playsound3
import speech_recognition as sr
import os
import re
import threading
import queue
# httpx removed; using gTTS exclusively for TTS
#AI Libraries
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

# === 🔐 Load API Keys ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === ⚙️ Voice options ===
# Set to True to post-process TTS output with a robotic Baymax-like effect.
ROBOTIC_VOICE = True
# Set to True to prefer a local male-sounding voice via pyttsx3.
# If pyttsx3 isn't installed, the code will fall back to gTTS.
MAKE_VOICE_MALE = True

def sanitize_text_for_tts(text: str) -> str:
    """Strip common markdown/formatting so TTS doesn't read characters like '*' or backticks.

    This keeps the spoken output clean by removing code blocks, inline code
    markers, markdown links, and emphasis characters.
    """
    if not text:
        return text
    s = str(text)
    # Remove fenced code blocks
    s = re.sub(r'```.*?```', ' ', s, flags=re.DOTALL)
    # Turn inline code `code` into code
    s = re.sub(r'`([^`]*)`', r"\1", s)
    # Convert markdown links [text](url) -> text
    s = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r"\1", s)
    # Remove common emphasis and strike markers (*, **, _, ~)
    s = s.replace('*', '').replace('_', '').replace('~', '')
    # Remove leftover multiple spaces/newlines
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def apply_robotic_effects(input_path: str, output_path: str) -> bool:
    """Try to apply a simple robotic effect to an audio file.

    This function uses pydub and numpy when available. It applies a
    low-frequency ring modulation (tremolo), a light bitcrusher, and
    re-normalization. If the required libraries aren't present, it
    returns False and the caller can fall back to unmodified audio.
    """
    try:
        import shutil
        # Prefer the imageio-ffmpeg bundled binary if available; it provides
        # a usable ffmpeg executable without requiring the user to modify PATH.
        ffmpeg_exe = None
        try:
            import imageio_ffmpeg as iioff
            ffmpeg_exe = iioff.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None

        from pydub import AudioSegment
        import numpy as np

        seg = AudioSegment.from_file(input_path)
        samples = np.array(seg.get_array_of_samples())

        # If stereo (interleaved), reshape to (n_samples, n_channels)
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels))
            # mix to mono for effect and then replicate later
            mono = samples.mean(axis=1).astype(np.float32)
        else:
            mono = samples.astype(np.float32)

        # Normalize to -1.0 .. 1.0
        max_val = np.max(np.abs(mono)) if mono.size else 1.0
        if max_val == 0:
            max_val = 1.0
        mono = mono / max_val

        sr = seg.frame_rate
        t = np.arange(len(mono)) / float(sr)

        # Ring modulation / tremolo: 30-60 Hz gives a robotic timbre
        lfo_freq = 45.0
        lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * lfo_freq * t))
        modulated = mono * lfo

        # Light bitcrush: reduce bit depth by quantizing samples
        bit_depth = 6  # 6-bit-ish effect
        levels = float(2 ** bit_depth - 1)
        crushed = np.round(modulated * levels) / levels

        # Slight speed change (pitch) by resampling: speed up slightly
        # We'll skip resampling to avoid extra deps; the ring modulation + crush suffice

        # Convert back to int16
        out = (crushed * 32767.0).astype(np.int16)

        # If original was stereo, duplicate mono to two channels
        if seg.channels > 1:
            out_stereo = np.repeat(out[:, None], seg.channels, axis=1).reshape(-1)
            out_bytes = out_stereo.tobytes()
            new_seg = AudioSegment(
                out_bytes,
                frame_rate=sr,
                sample_width=2,
                channels=seg.channels,
            )
        else:
            out_bytes = out.tobytes()
            new_seg = AudioSegment(out_bytes, frame_rate=sr, sample_width=2, channels=1)

        # Export to requested output path (use mp3 if output_path ends with .mp3)
        fmt = "mp3" if output_path.lower().endswith(".mp3") else "wav"
        new_seg.export(output_path, format=fmt)
        return True
    except Exception as e:
        print(f"Robotic effect failed (missing libs or error): {e}")
        return False

if not OPENAI_API_KEY:
    raise ValueError(" OPENAI_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise ValueError(" GROQ_API_KEY not found in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# === 🎙️ Microphone Setup ===
MIC_INDEX = 1  # Set this to your actual mic index

def get_audio():
    # Backwards-compatible get_audio with short timeouts so main loop can
    # also service console inputs. Returns recognized text or empty string.
    r = sr.Recognizer()
    with sr.Microphone(device_index=MIC_INDEX) as source:
        try:
            # wait up to 3s for phrase to start, limit phrase to 5s
            audio = r.listen(source, timeout=3, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            print(f"Listen error: {e}")
            return ""
    try:
        with open("test_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())
        print("Saved test_audio.wav")
    except:
        pass
    try:
        said = r.recognize_google(audio)
        print("Heard:", said)
        return said
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Google STT error: {e}")
        return ""

def speak(text):
    # sanitize text so TTS doesn't read formatting characters like asterisks
    try:
        text = sanitize_text_for_tts(text)
    except Exception:
        pass
    # Optionally use pyttsx3 for a local male-sounding voice (no files).
    if MAKE_VOICE_MALE:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices') or []
            male_voice = None
            for v in voices:
                # Check name/id for 'male' hint; fallback to first available voice
                v_name = getattr(v, 'name', '') or ''
                v_id = getattr(v, 'id', '') or ''
                if 'male' in v_name.lower() or 'male' in v_id.lower():
                    male_voice = v
                    break
            if not male_voice and voices:
                male_voice = voices[0]
            if male_voice:
                try:
                    engine.setProperty('voice', male_voice.id)
                except Exception:
                    pass
            # Slow down to make voice sound a bit deeper
            try:
                rate = engine.getProperty('rate')
                engine.setProperty('rate', max(80, int(rate * 0.9)))
            except Exception:
                pass
            engine.say(text)
            engine.runAndWait()
            return
        except Exception as e:
            print(f"pyttsx3 male voice failed (falling back to gTTS): {e}")

    # Use gTTS as the default TTS method. Optionally apply robotic effects.
    try:
        print("🔊 Speaking with gTTS...")
        from gtts import gTTS

        tts = gTTS(text)
        raw_path = "speech.mp3"
        tts.save(raw_path)

        play_path = raw_path
        if ROBOTIC_VOICE:
            robotic_path = "speech_robotic.mp3"
            processed = apply_robotic_effects(raw_path, robotic_path)
            if processed:
                play_path = robotic_path

        try:
            playsound3.playsound(play_path)
        except Exception as play_err:
            print(f"playsound failed: {play_err}")
            try:
                os.system(f'start "" "{play_path}"')
            except Exception as e:
                print(f"OS play fallback failed: {e}")
    except Exception as e:
        print(f" Failed to speak (gTTS): {e}")
    finally:
        # Cleanup generated files if present
        for fp in ("speech.mp3", "speech_robotic.mp3"):
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass

def ask_openai_conversation(user_input, system_prompt):
    try:
        print("Using GPT-4o...")
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        if "insufficient_quota" in str(e).lower():
            print("OpenAI quota exceeded.")
            return None
        print(f"OpenAI error: {e}")
        return None

def ask_groq(user_input):
    try:
        print("Falling back to Groq (llama-3.3-70b-versatile)...")
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Ascleon, a kind, friendly healthcare companion."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return "I'm having trouble accessing my backup model right now."

# === 🔁 Main Loop ===
print("Ascleon is online. Say anything to talk to Ascleon.\nSay 'shut down' to exit.\n")

first = True

# Start a background console reader so users can type instead of speaking.
console_queue = queue.Queue()

def _console_reader(q: queue.Queue):
    try:
        while True:
            # Blocking read; when user types, push into queue
            s = input()
            q.put(s)
    except Exception:
        return

threading.Thread(target=_console_reader, args=(console_queue,), daemon=True).start()

while True:
    # Prefer console input if available; otherwise listen on the mic.
    try:
        user_input = console_queue.get_nowait()
        # echo typed input so logs show it
        print(f"Typed: {user_input}")
    except queue.Empty:
        user_input = get_audio()

    if not user_input:
        continue

    if "shut down" in user_input.lower():
        print("Shutting down.")
        break

    if first or "hello" in user_input.lower():
        system_prompt = "You are Ascleon, a kind, gentle healthcare companion. Greet the user and offer help."
        first = False
    else:
        system_prompt = "You are Ascleon, continuing the conversation in a warm and helpful tone."

    reply = ask_openai_conversation(user_input, system_prompt)
    if not reply:
        reply = ask_groq(user_input)
    speak(reply)