import { FaceLandmarker, ObjectDetector, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const cam = document.getElementById('cam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const btn = document.getElementById('toggle');
const status = document.getElementById('status');
const statusText = document.getElementById('statusText');
const placeholder = document.getElementById('placeholder');
const warn = document.getElementById('warn');

const sounds = {
  facehide: new Audio('faudio.mp3'),
  sleep: new Audio('alarm.mp3'),
  phone: new Audio('paudio.mp3'),
};
const MESSAGES = {
  facehide: "DONT COVER YOUR FACE!",
  sleep: "WAKE UP & STUDY!",
  phone: "PUT THE PHONE AWAY!",
};
const SLEEP_FRAMES = 15, COVER_FRAMES = 20, EAR_LIMIT = 11.0;

let faceLandmarker, objectDetector, stream, running = false;
let closed = 0, covered = 0, current = null, lastVideoTime = -1;

function setStatus(cls, text) { status.className = 'status ' + cls; statusText.textContent = text; }

async function loadModels() {
  if (faceLandmarker) return;
  setStatus('off', 'Loading models…');
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "face_landmarker.task" },
    runningMode: "VIDEO", numFaces: 1,
  });
  objectDetector = await ObjectDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite" },
    runningMode: "VIDEO", scoreThreshold: 0.5, categoryAllowlist: ["cell phone"],
  });
}

function play(name) {
  const a = sounds[name];
  if (current && current !== name) stopAudio();
  if (a.paused) { a.currentTime = 0; a.play().catch(() => {}); }
  current = name;
}
function stopAudio() {
  Object.values(sounds).forEach(a => { a.pause(); a.currentTime = 0; });
  current = null;
}
const busy = () => Object.values(sounds).some(a => !a.paused && !a.ended);

function loop() {
  if (!running) return;
  if (cam.readyState >= 2 && cam.currentTime !== lastVideoTime) {
    lastVideoTime = cam.currentTime;
    const now = performance.now();
    const w = cam.videoWidth, h = cam.videoHeight;
    if (canvas.width !== w) { canvas.width = w; canvas.height = h; }
    ctx.clearRect(0, 0, w, h);

    const face = faceLandmarker.detectForVideo(cam, now).faceLandmarks[0];
    let sleepy = false, hidden = false;
    if (face) {
      covered = 0;
      const d = (a, b) => Math.hypot((face[a].x - face[b].x) * w, (face[a].y - face[b].y) * h);
      const ratio = d(159, 145) / d(130, 243) * 100;
      closed = ratio < EAR_LIMIT ? closed + 1 : 0;
      sleepy = closed >= SLEEP_FRAMES;
    } else {
      closed = 0; covered++;
      hidden = covered >= COVER_FRAMES;
    }

    let phone = false;
    for (const det of objectDetector.detectForVideo(cam, now).detections) {
      phone = true;
      const b = det.boundingBox;
      ctx.strokeStyle = '#f0f'; ctx.lineWidth = 3;
      ctx.strokeRect(b.originX, b.originY, b.width, b.height);
    }

    const active = hidden ? 'facehide' : sleepy ? 'sleep' : phone ? 'phone' : null;
    if (active) play(active);
    else if (!busy()) current = null;
    const shown = active || (busy() ? current : null);
    warn.hidden = !shown;
    if (shown) warn.textContent = MESSAGES[shown];
  }
  requestAnimationFrame(loop);
}

async function start() {
  try {
    await loadModels();
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  } catch (e) {
    setStatus('off', 'Error: ' + (e.name === 'NotAllowedError' ? 'camera permission denied' : e.message));
    return;
  }
  cam.srcObject = stream;
  await cam.play();
  cam.hidden = false; placeholder.classList.add('hidden');
  running = true; closed = covered = 0;
  setStatus('live', 'Monitoring live');
  btn.textContent = 'Stop Camera'; btn.classList.add('stop');
  // Unlock audio playback inside the user gesture
  Object.values(sounds).forEach(a => { a.muted = true; a.play().then(() => { a.pause(); a.currentTime = 0; a.muted = false; }).catch(() => { a.muted = false; }); });
  loop();
}

function stop() {
  running = false;
  stream?.getTracks().forEach(t => t.stop());
  cam.srcObject = null; cam.hidden = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  placeholder.classList.remove('hidden');
  warn.hidden = true; stopAudio();
  setStatus('off', 'Camera off');
  btn.textContent = 'Open Camera'; btn.classList.remove('stop');
}

btn.textContent = 'Open Camera'; btn.classList.remove('stop');
btn.onclick = () => running ? stop() : start();
