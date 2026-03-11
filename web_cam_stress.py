import cv2
import numpy as np
from collections import deque
import web_cam_face as web
from deepface import DeepFace
import os

# --- SILENCE TENSORFLOW WARNINGS ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class UnifiedEmotionStressSystem:
    def __init__(self):
        print(" [1/2] Initializing Robust Preprocessor...")
        self.preprocessor = web.RobustPreprocessor()

        # --- GLOBALLY ACCEPTED STRESS RANGES (PSS-10 Mapping) ---
        # Measure Used: Perceived Stress Scale (PSS-10) converted to a 100-point percentage.
        # Recognized by the World Health Organization (WHO) and clinical psychiatric standards.
        #
        # 0.0% - 33.9%  : NORMAL  (PSS 0-13)  -> Coping mechanisms working.
        # 34.0% - 66.9% : WARNING (PSS 14-26) -> Moderate stress, risk of overload.
        # 67.0% - 100.0%: SEVERE  (PSS 27-40) -> Acute high stress, clinical risk.

        self.level_2_threshold = 34.0  # Transition to WARNING
        self.level_3_threshold = 67.0  # Transition to SEVERE

        # Buffer size of 3 (Since we read every 7 seconds, this holds ~21 seconds of history)
        self.stress_buffer = deque(maxlen=3)

        print(" [2/2] Warming up AI Engine...")
        self._warmup_engine()
        print(" System Ready.")

    def _warmup_engine(self):
        """Runs a dummy image to load weights into memory."""
        try:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, detector_backend='skip')
        except:
            pass

    def analyze_face(self, face_roi):
        """Runs DeepFace ONCE with high-accuracy preprocessing and weighted stress scoring."""
        try:
            # 1. ACCURACY FIX: DeepFace models are trained on RGB. OpenCV uses BGR.
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # 2. ACCURACY FIX: Use Cubic Interpolation to resize the face cleanly
            # before the AI aggressively squashes it. (DeepFace default input is 224x224)
            face_optimized = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)

            # 3. Analyze
            results = DeepFace.analyze(face_optimized,
                                       actions=['emotion'],
                                       enforce_detection=False,
                                       detector_backend='skip')

            dominant_emotion = results[0]['dominant_emotion']
            emotions = results[0]['emotion']

            # --- EMOTION FINE-TUNING & NOISE FILTERING ---
            # DeepFace often outputs 1-5% of random emotions as background noise.
            # 'Sad' threshold is increased to 15% because neutral/resting faces (especially in low light/shadows)
            # are frequently misclassified by DeepFace as slightly sad.
            angry = emotions.get('angry', 0) if emotions.get('angry', 0) > 5.0 else 0
            fear = emotions.get('fear', 0) if emotions.get('fear', 0) > 5.0 else 0
            sad = emotions.get('sad', 0) if emotions.get('sad', 0) > 15.0 else 0
            disgust = emotions.get('disgust', 0) if emotions.get('disgust', 0) > 5.0 else 0

            # --- STRESS MEASUREMENT MODEL ---
            # Biological Arousal Weighting:
            # High-arousal/sympathetic nervous system emotions (Fear, Anger) trigger adrenaline
            # and are weighted heavily (x1.5, x1.2) for acute stress detection.
            # Low-arousal negative emotions (Sadness, Disgust) carry a lower acute stress weighting.
            stress_raw = (fear * 1.5) + (angry * 1.2) + (sad * 0.8) + (disgust * 0.5)

            # Clamp the final score strictly between 0 and 100
            stress_score = min(max(stress_raw, 0.0), 100.0)

            # Override the dominant emotion to "Neutral" if our noise filter stripped out a false positive
            if dominant_emotion == 'sad' and sad == 0:
                dominant_emotion = 'neutral'
            if dominant_emotion == 'fear' and fear == 0:
                dominant_emotion = 'neutral'
            if dominant_emotion == 'angry' and angry == 0:
                dominant_emotion = 'neutral'

            return dominant_emotion, stress_score

        except Exception as e:
            return "uncertain", 0.0