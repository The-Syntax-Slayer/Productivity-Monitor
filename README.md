# Productivity Monitor

AI-powered employee productivity monitoring that runs entirely in your browser. No server, no database, no login — and no video ever leaves the device.

Point a webcam at a workstation and the app detects whether the person is present, tracks working vs. away time, and reports productivity for the session — live, on one screen.

## Features

- **Live presence detection** — MoveNet pose estimation + COCO-SSD person detection, fused with a strict person gate and hysteresis so a single noisy frame can never flip the state.
- **Working / Away tracking** — real-time timers, productivity percentage, timeline strip, and a timestamped event log with a CSV download.
- **3-second warm-up** — on camera start the app analyses the scene first and shows `GETTING READY`. It never guesses a status before the AI has actually seen the frame.
- **Session history** — every ended session is saved locally with totals and productivity, with a 7-day overview and a full report CSV.
- **Away alerts** — configurable alert when someone is away continuously for too long (banner + sound, once per stretch).
- **Shift timing** — set working hours and days; the session ends automatically at shift end, and a banner appears at shift start. The camera never starts by itself.
- **Private by design** — all detection runs on-device with TensorFlow.js. Video frames are never uploaded, stored, or sent anywhere.

## How detection works

1. Every 500 ms a frame is analysed by MoveNet (pose keypoints) and COCO-SSD (person boxes).
2. A person counts as present only when the pose score is ≥ 0.3, at least 5 keypoints score ≥ 0.3, and an anchor keypoint (nose, shoulder, or hip) scores ≥ 0.3 — or when a COCO-SSD person box scores ≥ 0.5.
3. State changes use hysteresis: AWAY only after ~3 s with no person in frame (6 consecutive samples), back to WORKING after ~1.5 s of a person present (3 samples).
4. Productivity = Working ÷ (Working + Away) × 100.

If the AI models fail to load, the app says so (`AI FAILED`) and offers a retry — it never reports fake working time.

## Tech stack

- **Next.js 15 + React 19** — single route, static build
- **TensorFlow.js** — MoveNet (SinglePose Lightning) + COCO-SSD, WebGL with CPU fallback
- **Tailwind CSS** — light, minimal UI
- **localStorage** — session log, history, and settings (nothing leaves the browser)
- **Vitest** — unit tests for the detection rules

## Getting started

```bash
npm install
npm run dev     # http://localhost:3000
npm test        # unit tests
npm run build   # production build
```

Deploy anywhere static hosting works — no environment variables or backend needed. On Vercel, connect the repository and deploy the `main` branch.

## Project structure

```
app/            Next.js app — layout, styles, and the single dashboard page
src/lib/
  detection.ts  Presence rules, hysteresis, and the productivity formula
  features.ts   History, away alerts, and shift timing (pure functions)
```

## Acknowledgements

- Detection rules adapted from the MIT-licensed [AI-powered employee productivity monitoring system](https://github.com/PathumRathnayaka/AI-powered-employee-productivity-monitoring-system) by Pathum Rathnayaka.
- MoveNet and COCO-SSD via TensorFlow.js (Google); COCO dataset by Microsoft.
- Built with Next.js (Vercel) and React (Meta).

## License

MIT — see [LICENSE](LICENSE).
