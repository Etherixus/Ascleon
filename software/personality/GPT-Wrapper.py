#Time and sound libraries
import time
import pyaudio
import playsound3
import pyttsx3
import speech_recognition as sr
import os
import httpx
#AI Libraries
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

# === 🔐 Load API Keys ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(" OPENAI_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise ValueError(" GROQ_API_KEY not found in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# === 🎙️ Microphone Setup ===
MIC_INDEX = 1  # Set this to your actual mic index

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone(device_index=MIC_INDEX) as source:
        try:
            audio = r.listen(source)
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
    try:
        print("🔊 Speaking with OpenAI TTS...")
        response = httpx.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "tts-1",
                "voice": "onyx",  # Try "echo" or "fable" too
                "input": text
            }
        )
        if response.status_code != 200:
            print("TTS error (OpenAI), fallback to gTTS")
            raise Exception("OpenAI TTS failed")
        with open("speech.mp3", "wb") as f:
            f.write(response.content)
        playsound3.playsound("speech.mp3")
        os.remove("speech.mp3")
    except Exception as e:
        print(f"⚠️ TTS fallback: {e}")
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices.id)  # Set to male voice (index 0)
        engine.say("Hello, this is a male voice.")
        engine.runAndWait()

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

while True:
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