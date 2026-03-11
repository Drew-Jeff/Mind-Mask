import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class RobustPreprocessor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'face_detector.tflite')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        self.face_detector = vision.FaceDetector.create_from_options(options)

        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.quality_threshold = 0.5

    def apply_illumination_correction(self, gray_image):
        return self.clahe.apply(gray_image)

    def check_region_occlusion(self, roi_gray, detector, scale=1.1, neighbors=5):
        if roi_gray.size == 0: return True
        features = detector.detectMultiScale(roi_gray, scaleFactor=scale, minNeighbors=neighbors)
        return len(features) == 0

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.face_detector.detect(mp_image)

        processed_faces = []

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = max(0, bbox.origin_x), max(0, bbox.origin_y), bbox.width, bbox.height
                h_img, w_img, _ = frame.shape
                w, h = min(w, w_img - x), min(h, h_img - y)

                roi_color = frame[y:y + h, x:x + w]
                if roi_color.size == 0: continue

                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                enhanced_gray_roi = self.apply_illumination_correction(roi_gray)
                h_roi, w_roi = enhanced_gray_roi.shape

                # Regions
                roi_mouth = enhanced_gray_roi[int(h_roi * 0.6):, :]
                roi_upper_left = enhanced_gray_roi[0:int(h_roi / 2), 0:int(w_roi / 2)]
                roi_upper_right = enhanced_gray_roi[0:int(h_roi / 2), int(w_roi / 2):]

                # Check Occlusions
                mouth_occ = self.check_region_occlusion(roi_mouth, self.mouth_cascade, 1.7, 11)
                left_side_occ = self.check_region_occlusion(roi_upper_left, self.eye_cascade, 1.1, 5)
                right_side_occ = self.check_region_occlusion(roi_upper_right, self.eye_cascade, 1.1, 5)

                current_score = 0.4
                if not mouth_occ: current_score += 0.2
                if not left_side_occ: current_score += 0.2
                if not right_side_occ: current_score += 0.2

                emotion_ready = current_score >= self.quality_threshold
                box_color = (0, 255, 0) if emotion_ready else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                # Draw Occlusion warnings
                if mouth_occ: cv2.putText(frame, "Mouth Occ", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                          (0, 0, 255), 1)
                if left_side_occ: cv2.putText(frame, "L-Side Occ", (x - 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                              (0, 0, 255), 1)
                if right_side_occ: cv2.putText(frame, "R-Side Occ", (x + w, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                               (0, 0, 255), 1)

                status_text = "READY" if emotion_ready else "LOW QUAL"
                cv2.putText(frame, f"{status_text} ({current_score:.1f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            box_color, 2)

                face_data = {
                    "image": roi_color,  # <-- FIX: DeepFace gets the raw COLOR image now for high accuracy!
                    "coords": (x, y, w, h),
                    "emotion_ready": emotion_ready
                }
                processed_faces.append(face_data)

        return frame, processed_faces