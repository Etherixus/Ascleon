"""
camera_to_gpt.py

Small bridge that captures frames from the existing CameraStream (or directly
from Picamera2), performs lightweight analysis (face count, motion estimate,
color summary) and sends a text description to the GPT wrapper functions.

Design goals:
- Never send raw images to remote APIs. Only send short textual descriptions.
- Respect whether remote APIs are configured; fallback to local TTS if not.
- Sanitize descriptions before speaking/sending.
"""
from __future__ import annotations
import time
import importlib.util
import pathlib
import sys
from typing import Optional

import cv2
import numpy as np

# Try to import local CameraStream class
try:
    from .camera_stream import CameraStream
except Exception:
    CameraStream = None


def _load_gpt_wrapper_module():
    """Dynamically load the GPT-Wrapper.py module so this file doesn't rely on
    it being a proper package import. Returns the loaded module object.
    """
    base = pathlib.Path(__file__).resolve().parents[1]  # .../software
    wrapper_path = base / "personality" / "GPT-Wrapper.py"
    if not wrapper_path.exists():
        raise FileNotFoundError(f"GPT-Wrapper.py not found at {wrapper_path}")
    spec = importlib.util.spec_from_file_location("gpt_wrapper", str(wrapper_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def analyze_frame(frame: np.ndarray, face_cascade: Optional[cv2.CascadeClassifier] = None, prev_gray: Optional[np.ndarray] = None) -> tuple[str, Optional[np.ndarray]]:
    """Return a short text description for a single BGR frame and an updated prev_gray.

    Description includes: face count (if face_cascade provided), motion estimate
    (percent of pixels changed since prev_gray), and dominant color (RGB approx).
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    parts = []

    # Face detection
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            parts.append("no faces detected")
        elif len(faces) == 1:
            parts.append("one face detected")
        else:
            parts.append(f"{len(faces)} faces detected")
    # Motion estimate
    motion_pct = None
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = cv2.absdiff(prev_gray, gray)
        _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_pct = (th > 0).sum() / float(th.size)
        parts.append(f"motion about {int(motion_pct * 100)}% of pixels")

    # Dominant color: simple average, map to basic color words
    mean_bgr = cv2.mean(frame)[:3]
    mean_rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
    # Map to simple color names
    r, g, b = mean_rgb
    if r > g and r > b:
        color_name = "reddish"
    elif g > r and g > b:
        color_name = "greenish"
    elif b > r and b > g:
        color_name = "bluish"
    else:
        color_name = "neutral-colored"
    parts.append(f"scene looks {color_name} (avg RGB {mean_rgb})")

    desc = ", ".join(parts)
    return desc, gray


class CameraToGpt:
    def __init__(self, face_mode: bool = False, interval: float = 3.0, send_remote: bool = True):
        """face_mode: whether to run face detection
        interval: seconds between analyses
        send_remote: if False, never call remote LLMs even if keys are present
        """
        self.face_mode = face_mode
        self.interval = interval
        self.send_remote = send_remote
        self._camera = CameraStream(face_mode=face_mode) if CameraStream is not None else None
        self._face_cascade = None
        if face_mode:
            self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # lazy load the GPT wrapper when needed
        self._gpt = None

    def _ensure_gpt(self):
        if self._gpt is None:
            self._gpt = _load_gpt_wrapper_module()

    def _send_description(self, description: str) -> None:
        """Send description to GPT wrapper: if remote enabled and keys exist, call ask_openai_conversation, else call speak locally."""
        # ensure description is short
        description = description[:800]
        try:
            self._ensure_gpt()
        except Exception as e:
            print(f"Could not load GPT wrapper: {e}. Will only speak description locally.")
            # local speak if possible
            try:
                if self._gpt is None:
                    # try fallback simple say via pyttsx3
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.say(description)
                    engine.runAndWait()
                    return
            except Exception:
                print(description)
                return

        g = self._gpt
        # sanitize the description before sending/speaking
        try:
            description = g.sanitize_text_for_tts(description)
        except Exception:
            pass

        # prefer remote APIs only if send_remote flag true and keys present
        if self.send_remote and getattr(g, 'OPENAI_API_KEY', None):
            try:
                system_prompt = "You are Ascleon, describe a camera scene in 1-2 short sentences, no PII."
                reply = g.ask_openai_conversation(description, system_prompt)
                if reply:
                    g.speak(reply)
                    return
            except Exception as e:
                print(f"OpenAI call failed: {e}")
        if self.send_remote and getattr(g, 'GROQ_API_KEY', None):
            try:
                reply = g.ask_groq(description)
                if reply:
                    g.speak(reply)
                    return
            except Exception as e:
                print(f"Groq call failed: {e}")

        # fallback: speak the raw description locally
        try:
            g.speak(description)
        except Exception:
            print(description)

    def run_loop(self, max_iterations: Optional[int] = None):
        """Start the camera (if available) and run periodic analysis until stopped or
        max_iterations reached (None = infinite). This function runs in the
        foreground and prints what it does.
        """
        prev_gray = None
        iteration = 0

        if self._camera is None:
            # fallback: use cv2.VideoCapture(0)
            print("CameraStream class not available; trying cv2.VideoCapture(0)")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("No camera available")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(self.interval)
                        continue
                    desc, prev_gray = analyze_frame(frame, self._face_cascade, prev_gray)
                    print(f"Frame analysis: {desc}")
                    self._send_description(desc)
                    iteration += 1
                    if max_iterations and iteration >= max_iterations:
                        break
                    time.sleep(self.interval)
            finally:
                cap.release()
            return

        # use CameraStream
        print("Starting Picamera2 CameraStream for analysis")
        self._camera.start()
        try:
            while True:
                frame = self._camera.picam2.capture_array()
                desc, prev_gray = analyze_frame(frame, self._face_cascade, prev_gray)
                print(f"Frame analysis: {desc}")
                self._send_description(desc)
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    break
                time.sleep(self.interval)
        finally:
            self._camera.stop()


if __name__ == '__main__':
    # Simple CLI to run camera->gpt once or a few times
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--face', action='store_true', help='enable face detection')
    parser.add_argument('--interval', type=float, default=3.0, help='seconds between analyses')
    parser.add_argument('--no-remote', action='store_true', help='do not call remote APIs')
    parser.add_argument('--count', type=int, default=0, help='how many iterations (0 = infinite)')
    args = parser.parse_args()

    bridge = CameraToGpt(face_mode=args.face, interval=args.interval, send_remote=not args.no_remote)
    bridge.run_loop(max_iterations=(args.count or None))
