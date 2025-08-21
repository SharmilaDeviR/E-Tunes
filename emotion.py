"""
emotion.py — Webcam & emotion detection utilities
Assumes a TF Keras model saved at models/emotion_model.h5 that takes a 48x48 grayscale face.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Literal, List
import numpy as np
import cv2
import tensorflow as tf

MODEL_PATH = Path("models/emotion_model.h5")

# Ordered label list must match your model training
EMOTION_LABELS: List[str] = ['angry','disgust','fear','happy','neutral','sad','surprise']

class EmotionDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path.as_posix())
        # Use OpenCV’s Haar cascade for quick face detection (bundled with cv2)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        gray = gray.astype("float32") / 255.0
        return np.expand_dims(gray[..., np.newaxis], axis=0)  # (1,48,48,1)

    def predict_emotion_from_frame(self, frame_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Returns (label, confidence) for the most prominent face.
        If no face, runs on the center crop of frame.
        """
        h, w = frame_bgr.shape[:2]
        faces = self.face_cascade.detectMultiScale(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            cx, cy = w//2, h//2
            size = min(h, w)//2
            x1, y1 = max(cx-size,0), max(cy-size,0)
            crop = frame_bgr[y1:y1+2*size, x1:x1+2*size]
        else:
            x, y, fw, fh = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
            crop = frame_bgr[y:y+fh, x:x+fw]

        x = self._preprocess(crop)
        preds = self.model.predict(x, verbose=0)[0]  # (num_classes,)
        idx = int(np.argmax(preds))
        return EMOTION_LABELS[idx], float(preds[idx])

    def capture_from_camera(self, camera_index: int = 0, *, warmup_frames: int = 3) -> Optional[np.ndarray]:
        """Grab a single frame from webcam."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None
        for _ in range(warmup_frames):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
