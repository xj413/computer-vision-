#!/usr/bin/env python3
"""
Real-Time Customer Demographic & Sentiment Analyser
====================================================
analyses customer demographics (age, gender) and emotion in real-time using webcam feed.

uses OpenCV for video capture and DeepFace for facial analysis.
Implements frame skipping for performance optimization.

side project
"""

import cv2
import csv
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from deepface import DeepFace

# ============================================================================
# CONFIGURATION
# ============================================================================
WEBCAM_INDEX = 0  # Default webcam
ANALYSIS_INTERVAL = 30  # Run DeepFace every N frames
CSV_FILENAME = "customer_log.csv"  # Output log file
CSV_COLUMNS = ["Timestamp", "Gender", "Age", "Emotion"]

# DeepFace configuration
DEEPFACE_ACTIONS = ["age", "gender", "emotion"]
DEEPFACE_DETECTOR = "opencv"  # Fast detector for real-time use

# Display settings
BOX_COLOR = (0, 255, 0)  # Green in BGR
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (0, 255, 0)  # Green
FONT_THICKNESS = 2
TEXT_OFFSET_Y = 10  # Pixels above the bounding box


# ============================================================================
# DATA LOGGING
# ============================================================================
class CSVLogger:
    """Handles CSV logging with append functionality."""

    def __init__(self, filename: str, columns: List[str]):
        self.filename = filename
        self.columns = columns
        self._initialize_file()

    def _initialize_file(self) -> None:
        """Create file with headers if it doesn't exist."""
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
            print(f"[INFO] Created new log file: {self.filename}")
        else:
            print(f"[INFO] Appending to existing log file: {self.filename}")

    def log(self, timestamp: str, gender: str, age: int, emotion: str) -> None:
        """Append a detection record to the CSV file."""
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, gender, age, emotion])


# ============================================================================
# FACE ANALYSIS
# ============================================================================
class FaceAnalyzer:
    """
    Handles face detection and analysis using DeepFace.
    Implements caching for frame skipping optimization.
    """

    def __init__(self, analysis_interval: int = 30):
        self.analysis_interval = analysis_interval
        self.frame_count = 0
        self.cached_results: List[Dict[str, Any]] = []

    def analyze(self, frame) -> List[Dict[str, Any]]:
        """
        Analyze faces in the frame with frame skipping optimization.

        Frame Skipping Logic:
        - Only runs expensive DeepFace analysis every N frames
        - Returns cached results on intermediate frames
        - Maintains smooth display while reducing CPU load

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of face analysis results
        """
        self.frame_count += 1

        # Only run analysis every N frames
        if self.frame_count >= self.analysis_interval:
            self.frame_count = 0
            self.cached_results = self._run_analysis(frame)

        return self.cached_results

    def _run_analysis(self, frame) -> List[Dict[str, Any]]:
        """
        Run DeepFace analysis on the frame.

        Returns:
            List of dictionaries containing face data
        """
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=DEEPFACE_ACTIONS,
                detector_backend=DEEPFACE_DETECTOR,
                enforce_detection=False,  # Prevents crash when no face visible
                silent=True  # Suppress verbose output
            )

            # DeepFace returns a single dict if one face, list if multiple
            if isinstance(results, dict):
                results = [results]

            # Process and normalize results
            processed = []
            for face_data in results:
                # Skip if no valid face region detected
                region = face_data.get("region", {})
                if region.get("w", 0) == 0 or region.get("h", 0) == 0:
                    continue

                processed.append({
                    "region": region,
                    "age": int(face_data.get("age", 0)),
                    "gender": self._normalize_gender(face_data.get("dominant_gender", "Unknown")),
                    "emotion": face_data.get("dominant_emotion", "Unknown").capitalize()
                })

            return processed

        except Exception as e:
            # Log error but don't crash - return empty results
            print(f"[WARNING] Analysis error: {str(e)[:50]}")
            return []

    @staticmethod
    def _normalize_gender(gender: str) -> str:
        """Convert gender to display format."""
        gender_lower = gender.lower()
        if gender_lower == "man" or gender_lower == "male":
            return "Man"
        elif gender_lower == "woman" or gender_lower == "female":
            return "Woman"
        return gender.capitalize()


# ============================================================================
# VIDEO PROCESSOR
# ============================================================================
class VideoProcessor:
    """Handles video capture, display, and orchestrates analysis."""

    def __init__(self, webcam_index: int = 0):
        self.cap = cv2.VideoCapture(webcam_index)
        self.analyzer = FaceAnalyzer(analysis_interval=ANALYSIS_INTERVAL)
        self.logger = CSVLogger(CSV_FILENAME, CSV_COLUMNS)
        self.running = False

        # Verify camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {webcam_index}")

        # Get camera properties for display
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] Camera initialized: {self.width}x{self.height} @ {self.fps:.1f} FPS")

    def run(self) -> None:
        """Main processing loop."""
        self.running = True
        print("[INFO] Starting video capture. Press 'q' to quit.")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam")
                break

            # Analyze faces (with frame skipping)
            faces = self.analyzer.analyze(frame)

            # Process each detected face
            for face_data in faces:
                self._draw_face_overlay(frame, face_data)
                self._log_detection(face_data)

            # Display frame
            cv2.imshow("Customer Demographic & Sentiment Analyzer", frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit command received")
                break

        self.cleanup()

    def _draw_face_overlay(self, frame, face_data: Dict[str, Any]) -> None:
        """Draw bounding box and text overlay on the frame."""
        region = face_data["region"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        # Draw green bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            BOX_COLOR,
            BOX_THICKNESS
        )

        # Prepare label text: "Gender, Age, Emotion"
        label = f"{face_data['gender']}, {face_data['age']}, {face_data['emotion']}"

        # Calculate text position (above the bounding box)
        text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = x
        text_y = max(y - TEXT_OFFSET_Y, text_size[1] + 5)  # Ensure text stays in frame

        # Draw background rectangle for better text visibility
        bg_x1, bg_y1 = text_x - 2, text_y - text_size[1] - 5
        bg_x2, bg_y2 = text_x + text_size[0] + 2, text_y + 5
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

        # Draw label text
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS
        )

    def _log_detection(self, face_data: Dict[str, Any]) -> None:
        """Log face detection to CSV."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.log(
            timestamp=timestamp,
            gender=face_data["gender"],
            age=face_data["age"],
            emotion=face_data["emotion"]
        )

    def cleanup(self) -> None:
        """Release resources."""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Exiting.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Application entry point."""
    print("=" * 60)
    print("Real-Time Customer Demographic & Sentiment Analyzer")
    print("=" * 60)
    print(f"Analysis Interval: Every {ANALYSIS_INTERVAL} frames")
    print(f"Log File: {CSV_FILENAME}")
    print("-" * 60)

    try:
        processor = VideoProcessor(webcam_index=WEBCAM_INDEX)
        processor.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
