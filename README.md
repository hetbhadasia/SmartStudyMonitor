# Smart Study Monitor

Smart Study Monitor is a local webcam-based study companion. It watches for signs of sleep, a missing or covered face, and a visible cell phone, then displays a warning and plays an alert. A small browser dashboard lets you view the annotated camera feed and start or stop monitoring.

## Features

- **Sleepiness warning:** Uses facial landmarks to estimate whether the eyes are closed across consecutive frames.
- **Face visibility warning:** Alerts when a face is not detected for a run of consecutive frames. This can happen when a face is covered or out of view.
- **Phone warning:** Uses the bundled YOLOv8n model to detect cell phones in the camera image.
- **Browser dashboard:** Shows the live, annotated camera feed and provides a control to stop or restart monitoring.
- **Local processing:** The camera feed is processed by the Python app and served to a browser on your own computer; the app does not upload it to a remote service.

## Requirements

- Python 3.12 is the version used by the project's current environment.
- Git LFS, to retrieve the YOLO model weights.
- A webcam and audio output device.
- A desktop environment that can open a browser window.

The application uses OpenCV, cvzone, Ultralytics, and Pygame. The YOLO weights and alert audio files are included in the repository.

## Setup

Clone the repository and change into the project directory:

```bash
git lfs install
git clone <repository-url>
cd "Smart Study Monitor"
```

Install Git LFS before cloning so `yolov8n.pt` is fetched as a model file rather than a pointer file.

Create and activate a virtual environment, then install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python cvzone ultralytics pygame
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

From the project root, start the application:

```bash
python app.py
```

The app starts its local web server at [http://localhost:8000](http://localhost:8000) and attempts to open that page in your browser. Monitoring starts automatically. Use **Stop Camera** to stop the camera and silence alerts; use **Open Camera** to resume.

Port `8000` is currently fixed in `app.py`. If it is already in use, stop the other service using it before starting this application.

## How detection works

- Sleepiness is inferred from eye landmarks over 15 consecutive frames.
- A missing face triggers a visibility warning after 20 consecutive frames. The app cannot tell whether the face is covered or simply outside the camera's view.
- A phone is flagged when the YOLO model identifies a `cell phone` with confidence greater than 0.5.
- Alerts are prioritized in this order: missing face, sleepiness, then phone.

Detection is heuristic and depends on camera position, lighting, and model accuracy. It is intended as a study aid, not a safety or health monitoring system.

## Project structure

```text
.
├── app.py                 # Webcam processing, detection, alarms, and local HTTP server
├── web/
│   ├── index.html         # Browser dashboard
│   └── style.css          # Dashboard styles
├── yolov8n.pt             # YOLOv8n weights used for phone detection
├── alarm.mp3              # Sleepiness alert
├── faudio.mp3             # Face visibility alert
└── paudio.mp3             # Phone detection alert
```

## Troubleshooting

- **Camera does not start:** Grant camera access to the terminal or Python process in your operating system's privacy settings, and close other applications using the webcam.
- **No sound:** Check the system's audio output and volume. Pygame initializes the audio mixer when the app starts.
- **Dashboard does not load:** Confirm the app is still running and that port `8000` is available.
- **Model or audio file not found:** Run `python app.py` from the project root so the relative asset paths resolve correctly.
