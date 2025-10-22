# camera_stream.py
import cv2
from picamera2 import Picamera2
import threading

class CameraStream:
    def __init__(self, face_mode=False):
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (640, 480)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.configure("preview")
        self.running = False
        self.face_mode = face_mode

        if self.face_mode:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def start(self):
        self.running = True
        self.picam2.start()
        threading.Thread(target=self._stream, daemon=True).start()

    def stop(self):
        self.running = False
        self.picam2.stop()

    def _stream(self):
        while self.running:
            frame = self.picam2.capture_array()
            
            if self.face_mode:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Ascleon Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

        cv2.destroyAllWindows()
