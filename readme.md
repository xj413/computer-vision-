# Real-Time Customer Demographic & Sentiment Analyzer

A python application that analyzes customer demographics (age, gender) and sentiment (emotion) in real-time using webcam feed.

## Features

- **Real-time face detection** using OpenCV
- **Demographic analysis**: Age and gender estimation
- **Sentiment analysis**: Emotion recognition (happy, sad, angry, neutral, etc.)
- **Visual overlay**: Green bounding boxes with demographic labels
- **CSV logging**: Automatic logging of all detections with timestamps
- **Performance optimized**: Frame skipping for smooth 30 FPS display

---

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: First run may take longer as DeepFace downloads required models (~500MB).

---

## Usage

```bash
python main.py
```

Press `q` to quit the application.

---

## Output

### Live Display
- Green bounding box around each detected face
- Text overlay: `Gender, Age, Emotion` (e.g., "Man, 25, Happy")

### CSV Log (`customer_log.csv`)
```csv
Timestamp,Gender,Age,Emotion
2026-01-20 15:30:45,Man,28,Happy
2026-01-20 15:30:46,Woman,32,Neutral
```

---

## Frame Skipping Logic Explained

### The Problem
DeepFace's face analysis is computationally expensive (~100-300ms per frame). Running it on every frame would result in:
- Only 3-10 FPS display (choppy video)
- High CPU/GPU usage
- Poor user experience

### The Solution: Frame Skipping

```
Frame 1  → Run DeepFace analysis → Cache results → Display
Frame 2  → Use cached results                    → Display
Frame 3  → Use cached results                    → Display
...
Frame 30 → Run DeepFace analysis → Update cache  → Display
Frame 31 → Use cached results                    → Display
...
```

### How It Works

1. **Analysis Interval**: Set to 30 frames by default (`ANALYSIS_INTERVAL = 30`)

2. **Frame Counter**: Tracks frames since last analysis
   ```python
   self.frame_count += 1
   if self.frame_count >= self.analysis_interval:
       self.frame_count = 0
       self.cached_results = self._run_analysis(frame)
   ```

3. **Result Caching**: The last analysis results are stored and reused
   ```python
   return self.cached_results  # Return cached data for intermediate frames
   ```

4. **Smooth Display**: While DeepFace runs only ~1 time per second (at 30 FPS), the webcam feed displays at full speed with the last known demographic data overlaid.

### Performance Impact

| Metric | Without Skipping | With Skipping (30 frames) |
|--------|------------------|---------------------------|
| Display FPS | 3-10 | 30 |
| CPU Usage | 100% | ~15-25% |
| Analysis Latency | Per frame | ~1 second |

### Trade-offs

- **Pro**: Smooth video display, lower CPU usage
- **Con**: Demographic data updates only once per second
- **Mitigation**: For faster-moving scenarios, reduce `ANALYSIS_INTERVAL` to 15

---

## Configuration

Edit these constants in `main.py`:

```python
WEBCAM_INDEX = 0           # Camera index (0 = default)
ANALYSIS_INTERVAL = 30     # Frames between DeepFace calls
CSV_FILENAME = "customer_log.csv"
DEEPFACE_DETECTOR = "opencv"  # Options: opencv, mtcnn, retinaface, ssd
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
├─────────────────────────────────────────────────────────────┤
│  VideoProcessor                                              │
│  ├── Captures webcam frames                                  │
│  ├── Orchestrates analysis                                   │
│  └── Displays output                                         │
├─────────────────────────────────────────────────────────────┤
│  FaceAnalyzer                                                │
│  ├── Implements frame skipping                               │
│  ├── Runs DeepFace analysis                                  │
│  └── Caches results                                          │
├─────────────────────────────────────────────────────────────┤
│  CSVLogger                                                   │
│  ├── Creates log file with headers                           │
│  └── Appends detection records                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Failed to open webcam" | Check camera permissions, try different `WEBCAM_INDEX` |
| Slow first startup | Normal - DeepFace is downloading models |
| No faces detected | Ensure good lighting, face the camera directly |
| High CPU usage | Increase `ANALYSIS_INTERVAL` or use GPU backend |

