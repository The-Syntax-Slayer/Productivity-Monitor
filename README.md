# Productivity Monitor

Browser-only employee productivity monitor — **no login, no backend, no database**.

Upgraded from the open-source project
[AI-powered-employee-productivity-monitoring-system](https://github.com/PathumRathnayaka/AI-powered-employee-productivity-monitoring-system)
by **Pathum Rathnayaka (MIT License, see `LICENSE`)**. That repo's Flask +
MongoDB backend cannot run on Vercel, so detection was rebuilt to run entirely
in the browser with TensorFlow.js (MoveNet pose + COCO-SSD), using only the
clone's own detection rules.

## Detection rules

Only two states (user decision, 9 Nov 2026 — phone and sleeping detection
were removed entirely):

| State | Rule |
| --- | --- |
| WORKING | A real person is visible: MoveNet pose score ≥ 0.3, ≥ 5 keypoints ≥ 0.3, at least one anchor keypoint (nose / shoulder / hip) ≥ 0.3 — or a COCO-SSD person box ≥ 0.5 |
| AWAY | No person visible for ~3 s (6 consecutive no-person samples at 500 ms/sample) |

**AWAY flicker fix (user-tested on a real laptop, 9 Nov 2026):** walking away
used to loop AWAY 1 s → WORKING → AWAY. Two causes: the old code flipped to
AWAY on *any single* no-person frame, and it counted *any* MoveNet pose —
even a low-score background blob — as a person. Fixed with the strict
person gate above plus consecutive-counter hysteresis both ways: 6 absent
samples to go AWAY, 3 present samples (~1.5 s) to come back to WORKING.
A single frame can never flip the state. The old 5-sample/60%-majority
confirmation buffer was replaced by this design (a majority buffer lets
stray hits flip the state).

Productivity % = Working ÷ (Working + Away).

## Webcam — fixed bug

A previous build showed no camera footage because the `<video>` element was
rendered only after the AI models loaded, so the stream had nowhere to attach.
Here the video element is **always mounted**, `getUserMedia` runs first, the
stream is attached and `play()`ed immediately, `setRunning(true)` is called
right after that, and the TensorFlow models load in the background
("Camera on — loading AI…"). A `useEffect` re-attaches the stream on every
render. If camera permission is denied, a clear on-screen error appears.

Live only — no simulation mode, no demo controls, no forced states.
Every state change comes from the camera.

## Features

- One route `/`, live webcam + state badge, live timers per state
- Productivity %, timeline strip, event log (timestamp, from → to, duration,
  engine, confidence), End Session summary + real CSV download
- localStorage persistence of events + timers

## Run / test / build

```bash
npm install
npm test    # vitest unit tests for the detection rules
npm run build
npm run dev
```
