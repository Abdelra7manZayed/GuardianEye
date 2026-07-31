# PPE Violation Detection

A Streamlit app that scans uploaded video for missing personal protective
equipment (helmets, vests) using a fine-tuned YOLO11 model. Detected
violations are blurred for privacy, boxed for visibility, and tracked
across frames so the same person isn't double-counted.

## How it works

1. **Upload** a video (`mp4`, `avi`, `mov`, `mkv`) through the web UI.
2. **Sampling** — instead of running detection on every frame, the app
   only analyzes a short burst of frames every few seconds (both
   configurable in the sidebar). Skipped frames are written through
   unchanged. This cuts the number of model calls dramatically on longer
   videos.
3. **Detection & tracking** — on tested frames, YOLO11 finds PPE
   violations and ByteTrack assigns a persistent ID to each person, so
   the same worker is tracked across frames instead of being treated as
   a new detection each time.
4. **Blur & flag** — any frame with a violation gets a full-frame blur,
   with only the violator's bounding box restored to sharp and labeled
   (`ID | class | confidence`).
5. **Output** — the processed video is shown in-browser and available to
   download once processing finishes.

## Model

| | |
|---|---|
| Architecture | YOLO11s |
| Parameters | 9.4M |
| GFLOPs | 21.3 |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 x 640 |
| Optimizer | AdamW |
| Initial LR | 0.01 |

**Test set results** (48 held-out images):

| Metric | Score |
|---|---|
| Precision | 0.805 |
| Recall | 0.810 |
| mAP@0.50 | 0.891 |
| mAP@0.50-0.95 | 0.650 |

## Dataset

[Roboflow — helmet-vest-detection](https://universe.roboflow.com/vellisyas-workspace/helmet-vest-detection-hfaea)
(version 10), 4 classes:

- `complete_vest_helmet`
- `no_helmet`
- `no_vest`
- `no_vest_no_helmet`

| Split | Images |
|---|---|
| Train | 352 |
| Validation | 100 |
| Test | 48 |

## Demo





<!-- Replace the line below with your own video link or GIF once you have one. -->
[Watch the demo video](https://github.com/user-attachments/assets/ee1c1b39-2e50-4461-b4f5-554e4edc8fc0)


## Running the app

```bash
pip install streamlit opencv-python ultralytics
streamlit run app.py
```

The app looks for a trained weights file (`best.pt`) automatically under
`runs/`. If it can't find one, set the path manually in the sidebar.

## Sidebar controls

| Control | What it does |
|---|---|
| Model path | Path to the trained `best.pt` weights file |
| Confidence threshold | Minimum detection confidence required for a box to count |
| Blur strength | How strong the Gaussian blur is over a violation frame |
| Test interval (seconds) | How often the model runs a fresh detection pass |
| Frames tested per burst | How many consecutive frames are analyzed each time the model runs |

## Team

- [Mohamed Mahmoud Salem](https://www.linkedin.com/in/mohamed-mahmoud-mohamed-salem)
- [Ahmed Ayman](https://www.linkedin.com/in/ahmed-ayman-b1648232b)
- [Antwan Gamil](https://www.linkedin.com/in/antwan-gamil)
- [Abdelrahman Gamal Zayed](https://www.linkedin.com/in/abdelrahman-gamal-zayed)
- [Omar Saber](https://www.linkedin.com/in/omar-saber1)

## Project structure

```
app.py     # Streamlit application
nti-1.ipynb  # Training notebook: dataset download, YOLO11 fine-tuning, evaluation
```

