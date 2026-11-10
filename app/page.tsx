'use client';

/**
 * Productivity Monitor — one premium dashboard, no login, no backend.
 * Browser-only employee productivity monitor. Detection runs in the
 * browser with TensorFlow.js (MoveNet pose + COCO-SSD), using ONLY the
 * detection rules in src/lib/detection.ts:
 *   - WORKING: a real person is present (strict pose quality gate)
 *   - AWAY: no person visible for ~3 s (6 consecutive no-person samples)
 *
 * AWAY FLICKER FIX (user-tested on a real laptop, 9 Nov 2026): walking away
 * looped AWAY 1s -> WORKING -> AWAY. Two causes: classifyFrame() flipped to
 * AWAY on ANY single no-person frame, and ANY MoveNet pose (even a low-score
 * background blob) counted as a person. Fixed with a strict person gate
 * (pose score >= 0.3, >= 5 high-conf keypoints, one anchor keypoint) plus
 * consecutive-counter hysteresis both ways: 6 absent samples to go AWAY,
 * 3 present samples to come back. A single frame can never flip state.
 * PHONE and SLEEPING detection were removed entirely at the user's request.
 *
 * WARM-UP ("GETTING READY", 10 Nov 2026): Start Camera used to claim
 * WORKING instantly (presence started confirmed) and to flip to AWAY while
 * the models were still loading (model-less frames counted as "no person").
 * Now presence starts UNCONFIRMED, frames before any model loads are not
 * counted at all, and the first ~3 s of analysed frames (6 samples) only
 * move the state silently — no timers, no logged event. The Status card
 * shows GETTING READY until then, then the real WORKING/AWAY.
 * AI-FAILURE RULE (10 Nov 2026, after a laptop test showed fake WORKING
 * 100% with "AI failed to load"): the 15 s safety net is GONE — it used
 * to end the warm-up into fake WORKING time. Timers now count ONLY when
 * the AI is ready and the warm-up is done; on a load failure (30 s
 * timeout included) the Status shows AI FAILED, the donut shows "—",
 * and a Retry AI button reloads the models without restarting the
 * camera or the session. A "No video signal" chip appears when the
 * camera is on but the <video> gets no frames for 3 s.
 *
 * Live only — no simulation mode, no demo controls, no forced states.
 * Every state change comes from the camera.
 *
 * WEBCAM FIX (this is why the previous build showed no footage):
 *   1. The <video> element below is ALWAYS mounted — it is plain JSX, never
 *      inside a conditional render. It exists before the camera starts.
 *   2. startCamera() calls getUserMedia FIRST, attaches the stream to the
 *      video element immediately, calls play(), and calls setRunning(true)
 *      IMMEDIATELY after that — BEFORE any TensorFlow model is imported.
 *   3. Models load in the background afterwards ("Camera on — loading AI…").
 *   4. A useEffect with no dependency array re-attaches streamRef to the
 *      video element on every render, so a re-render can never lose the stream.
 *
 * UI (redesigned 9 Nov 2026, approved A+B combined mockup, LIGHT theme):
 * top bar (48px bare aperture logo + two-tone wordmark, LIVE pill,
 * session clock, productivity pill, gradient hairline) + camera-first
 * split on TOP (big 16:9 live camera, full-bleed, no header/status
 * text; Start Camera centred when off, small controls bottom-right
 * when on) + right rail (timeline, event log + CSV) + KPI strip
 * BELOW (Working Time + Away Time under the camera, full-size Status
 * card with 104px donut % under the right rail; strip columns match
 * the split above). Single screen, no scroll.
 * Logo: bare custom aperture mark in the emerald→brand gradient, no
 * tile, no border, no shadow. Avg Confidence KPI removed at the user's
 * request (the confidence still records in the event log / CSV).
 * FONT SYSTEM (approved Font 1 — Editorial Serif): Fraunces headings +
 * giant WORKING/AWAY word + title; Inter body/labels; JetBrains Mono
 * (tabular) for every timer/clock number. Buttons stay Inter.
 * MINIMAL TEXT (user rule): only labels, buttons, status words, numbers
 * and the title remain — no subtitles, no explanatory paragraphs, no
 * footer attribution, no version line.
 * No sidebar, no avatar, no fake nav/icons — nothing that does nothing.
 *
 * FEATURES 3/4/5 (10 Nov 2026, pure functions in src/lib/features.ts):
 *   3. History — each ended session saves a summary to localStorage
 *      (clone-productivity-monitor-history-v1, newest 100); top-bar
 *      History button overlays a 7-day view + report CSV. The dashboard
 *      stays mounted underneath, so a running session is never interrupted.
 *   4. Away alert — continuous AWAY >= settings.awayAlertMinutes fires
 *      once per stretch (red pulse + banner + beep, optional browser
 *      notification, off by default); resets on return to WORKING.
 *   5. Shift — configured start/end + working days; shift end auto-ends
 *      the session (history saved), shift start shows a banner (camera
 *      never auto-starts), running outside shift shows an "Outside shift"
 *      note. Same-day shifts only. All in the top-bar clock popover.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  classifyFrame,
  createPresenceState,
  eventsToCsv,
  formatDuration,
  isCocoPerson,
  isPosePerson,
  productivityPercent,
  timelineSegments,
  updatePresence,
  type MonitorState,
} from '../src/lib/detection';
import {
  appendHistory,
  DEFAULT_SETTINGS,
  groupByDay,
  HISTORY_KEY,
  historyToCsv,
  isOutsideShift,
  isShiftEnded,
  isShiftStarted,
  isShiftValid,
  parseHistory,
  parseSettings,
  SETTINGS_KEY,
  shouldAlertAway,
  summarizeSession,
  type HistoryEntry,
  type Settings,
} from '../src/lib/features';

type AiStatus = 'off' | 'loading' | 'ready' | 'error';

type LogEvent = {
  at: string;
  from: MonitorState;
  to: MonitorState;
  durationInPreviousMs: number;
  engine: string;
  confidence: number;
};

type Timers = Record<MonitorState, number>;

const EMPTY_TIMERS: Timers = { WORKING: 0, AWAY: 0 };

const STATE_STYLE: Record<
  MonitorState,
  { word: string; text: string; glowText: string; glowCard: string }
> = {
  WORKING: {
    word: 'WORKING',
    text: 'text-emerald-500',
    glowText: 'text-glow-working',
    glowCard: 'card-glow-working',
  },
  AWAY: {
    word: 'AWAY',
    text: 'text-red-500',
    glowText: 'text-glow-away',
    glowCard: 'card-glow-away',
  },
};

const STORAGE_KEY = 'clone-productivity-monitor-v1';

/* Donut ring geometry (r = 52). */
const DONUT_R = 52;
const DONUT_C = 2 * Math.PI * DONUT_R;

export default function Home() {
  // The video element is rendered unconditionally below — videoRef is never null after mount.
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseRef = useRef<any>(null); // MoveNet detector, loaded in the background
  const cocoRef = useRef<any>(null); // COCO-SSD detector, loaded in the background
  const detectTimerRef = useRef<number | null>(null);
  const tickTimerRef = useRef<number | null>(null);

  const [running, setRunning] = useState(false);
  const [aiStatus, setAiStatus] = useState<AiStatus>('off');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [current, setCurrent] = useState<MonitorState>('WORKING');
  const [timers, setTimers] = useState<Timers>(EMPTY_TIMERS);
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [sessionStart, setSessionStart] = useState<number | null>(null);
  const [showSummary, setShowSummary] = useState(false);

  // Features 3/4/5 — settings, history, away alert, shift.
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [awayAlert, setAwayAlert] = useState(false);
  const [shiftBanner, setShiftBanner] = useState(false);

  // Warm-up ("GETTING READY") — after Start Camera the app analyses real
  // frames for ~3 s (ANALYZE_SAMPLES × 500 ms) before it claims WORKING
  // or AWAY. Until then the Status card shows GETTING READY, no timers
  // accumulate and no event is logged.
  const ANALYZE_SAMPLES = 6;
  const [analyzing, setAnalyzing] = useState(false);

  // Refs mirror state for the detection/tick intervals (avoids stale closures).
  const currentRef = useRef<MonitorState>('WORKING');
  const stateSinceRef = useRef<number>(Date.now());
  // Hysteresis state for the AWAY flicker fix — consecutive counters only.
  // Starts unconfirmed: nobody is assumed present until real frames say so.
  const presenceRef = useRef(createPresenceState(false));
  const runningRef = useRef(false);

  // Feature refs mirror state for the tick interval (avoids stale closures).
  const settingsRef = useRef<Settings>(DEFAULT_SETTINGS);
  const timersRef = useRef<Timers>(EMPTY_TIMERS);
  const eventsRef = useRef<LogEvent[]>([]);
  const sessionStartRef = useRef<number | null>(null);
  const alertFiredRef = useRef(false);
  const shiftBannerDayRef = useRef<string | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const endSessionRef = useRef<() => void>(() => {});
  const analyzingRef = useRef(false);
  const analyzedSamplesRef = useRef(0);
  // AI load state mirror (for the tick interval) + load-attempt guard so a
  // slow earlier load can never overwrite a newer retry's result.
  const aiStatusRef = useRef<AiStatus>('off');
  const loadAttemptRef = useRef(0);
  // No-video-signal diagnostic: camera on but the <video> gets no frames.
  const [noVideoSignal, setNoVideoSignal] = useState(false);
  const noVideoRef = useRef(0);

  // WEB CAM FIX #4 — re-attach the stream on every render if it ever got lost.
  // Also retries play() while paused: a stream that attached but never
  // started playing renders as a black frame, which looks like a dead camera.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !streamRef.current) return;
    if (v.srcObject !== streamRef.current) {
      v.srcObject = streamRef.current;
    }
    if (v.paused) {
      v.play().catch(() => {});
    }
  });

  // Restore a previous session's log from localStorage.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw) as { events: LogEvent[]; timers: Timers };
        // Drop old saved events involving the removed PHONE/SLEEPING states.
        if (saved.events?.length) {
          setEvents(
            saved.events.filter((e: LogEvent) =>
              (e.from === 'WORKING' || e.from === 'AWAY') && (e.to === 'WORKING' || e.to === 'AWAY'),
            ),
          );
        }
        // Normalise old saved timers: PHONE/SLEEPING no longer exist.
        if (saved.timers) {
          setTimers({ WORKING: saved.timers.WORKING ?? 0, AWAY: saved.timers.AWAY ?? 0 });
        }
      }
    } catch {
      // Corrupt saved data is ignored — the session just starts fresh.
    }
  }, []);

  // Persist the event log + timers.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ events, timers }));
    } catch {
      // Storage full / unavailable — the live session still works.
    }
  }, [events, timers]);

  // Features 3/5 — load settings + history once on mount.
  useEffect(() => {
    try {
      const rawSettings = localStorage.getItem(SETTINGS_KEY);
      if (rawSettings) setSettings(parseSettings(JSON.parse(rawSettings)));
    } catch {
      // Corrupt settings are ignored — defaults apply.
    }
    try {
      const rawHistory = localStorage.getItem(HISTORY_KEY);
      if (rawHistory) setHistory(parseHistory(JSON.parse(rawHistory)));
    } catch {
      // Corrupt history is ignored — it starts empty.
    }
  }, []);

  // Persist settings on change.
  useEffect(() => {
    settingsRef.current = settings;
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      // Storage unavailable — settings still work for this visit.
    }
  }, [settings]);

  // Escape closes the settings popover (and the History view). The popover
  // is rendered fixed at viewport level, so keyboard dismissal must live
  // here too — the backdrop alone is not enough.
  useEffect(() => {
    if (!showSettings && !historyOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowSettings(false);
        setHistoryOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showSettings, historyOpen]);

  // Mirror timers / events / session start for the tick interval.
  useEffect(() => {
    timersRef.current = timers;
  }, [timers]);
  useEffect(() => {
    eventsRef.current = events;
  }, [events]);
  useEffect(() => {
    sessionStartRef.current = sessionStart;
  }, [sessionStart]);
  useEffect(() => {
    aiStatusRef.current = aiStatus;
  }, [aiStatus]);

  const recordTransition = useCallback((to: MonitorState, engine: string, confidence: number) => {
    const from = currentRef.current;
    if (from === to) return;
    const now = Date.now();
    const durationInPreviousMs = now - stateSinceRef.current;
    setEvents((prev) => [
      ...prev,
      { at: new Date(now).toISOString(), from, to, durationInPreviousMs, engine, confidence },
    ]);
    currentRef.current = to;
    stateSinceRef.current = now;
    setCurrent(to);
    // Feature 4 — back to WORKING resets the away alert for the next stretch.
    if (to === 'WORKING') {
      alertFiredRef.current = false;
      setAwayAlert(false);
    }
  }, []);

  const runDetectionFrame = useCallback(async () => {
    const v = videoRef.current;
    if (!v || v.readyState < 2 || v.videoWidth === 0) return;
    // Models still loading — this frame proves nothing. Do NOT count it:
    // counting model-less frames as "no person" is what used to flip a
    // person sitting in front of the camera to AWAY right after start.
    if (!poseRef.current && !cocoRef.current) return;

    // Person-present gate: a frame only counts as "person" if MoveNet
    // returns a quality pose OR COCO-SSD returns a high-score person box.
    // A stray low-score pose on the background must NOT count — that was
    // half of the AWAY flicker (it flipped AWAY straight back to WORKING).
    let personPresent = false;
    let confidence = 0;
    let engine = 'MoveNet + COCO-SSD';

    try {
      if (cocoRef.current) {
        const preds = (await cocoRef.current.detect(v)) as { class: string; bbox: [number, number, number, number]; score: number }[];
        for (const p of preds) {
          if (p.class === 'person' && isCocoPerson(p.score)) {
            personPresent = true;
            confidence = Math.max(confidence, p.score);
          }
        }
        engine = 'COCO-SSD';
      }
      if (poseRef.current) {
        const poses = (await poseRef.current.estimatePoses(v)) as any[];
        const pose = poses?.[0];
        if (pose && isPosePerson(pose)) {
          personPresent = true;
          confidence = Math.max(confidence, pose.score ?? 0);
        }
        engine = cocoRef.current ? 'MoveNet + COCO-SSD' : 'MoveNet';
      }
    } catch {
      // One bad frame must never kill the loop — the next frame retries.
      return;
    }

    // Hysteresis: 6 consecutive absent samples (~3 s) to go AWAY, 3
    // consecutive present samples (~1.5 s) to come back. A single frame
    // can never flip the state — this is the AWAY flicker fix.
    presenceRef.current = updatePresence(presenceRef.current, personPresent);
    const next = classifyFrame({ confirmedPresent: presenceRef.current.confirmedPresent });

    // Warm-up: count only frames a loaded model actually analysed. While
    // analysing, the state still tracks the hysteresis silently — but no
    // event is logged, so no fake WORKING→AWAY transition is ever recorded.
    // Once enough real frames are in, durations restart clean from here.
    if (analyzingRef.current) {
      analyzedSamplesRef.current += 1;
      currentRef.current = next;
      setCurrent(next);
      if (analyzedSamplesRef.current >= ANALYZE_SAMPLES) {
        analyzingRef.current = false;
        setAnalyzing(false);
        stateSinceRef.current = Date.now();
      }
      return;
    }
    recordTransition(next, engine, Number(confidence.toFixed(2)));
  }, [recordTransition]);

  const loadModelsInBackground = useCallback(async () => {
    // A slow earlier load must never overwrite a newer retry: only the
    // latest attempt may apply its result. The load also gets a 30 s
    // timeout — a hung model download used to leave the UI on "loading AI…"
    // forever while the timers silently counted fake WORKING time.
    const attempt = ++loadAttemptRef.current;
    const timeout = new Promise<never>((_, reject) =>
      window.setTimeout(() => reject(new Error('AI load timed out')), 30000),
    );
    try {
      const load = (async () => {
        const tf = await import('@tensorflow/tfjs');
        await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
        await tf.ready();
        const poseDetection = await import('@tensorflow-models/pose-detection');
        const cocoSsd = await import('@tensorflow-models/coco-ssd');
        const [pose, coco] = await Promise.all([
          poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          }),
          cocoSsd.load(),
        ]);
        return { pose, coco };
      })();
      const { pose, coco } = await Promise.race([load, timeout]);
      if (attempt !== loadAttemptRef.current) return; // superseded by a retry
      poseRef.current = pose;
      cocoRef.current = coco;
      if (runningRef.current) setAiStatus('ready');
    } catch (err) {
      if (attempt !== loadAttemptRef.current) return; // superseded by a retry
      console.error('AI failed to load', err);
      poseRef.current = null;
      cocoRef.current = null;
      if (runningRef.current) {
        // Honest failure state: warm-up ends, timers stay paused, Status
        // shows AI FAILED — never falls back to fake WORKING time.
        setAiStatus('error');
        analyzingRef.current = false;
        setAnalyzing(false);
      }
    }
  }, []);

  // Retry the AI load without restarting the camera or the session: the
  // warm-up restarts from zero so detection is judged on fresh frames.
  const retryAi = useCallback(() => {
    if (!runningRef.current) return;
    setAiStatus('loading');
    setNoVideoSignal(false);
    noVideoRef.current = 0;
    presenceRef.current = createPresenceState(false);
    analyzedSamplesRef.current = 0;
    analyzingRef.current = true;
    setAnalyzing(true);
    stateSinceRef.current = Date.now();
    void loadModelsInBackground();
  }, [loadModelsInBackground]);

  // Feature 4 — one short beep via Web Audio (no audio file). Only ever
  // called while a session is running, so no autoplay issue.
  const beep = useCallback(() => {
    try {
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) return;
      audioCtxRef.current ??= new Ctx();
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') void ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.4);
    } catch {
      // Audio unavailable — the visual alert still shows.
    }
  }, []);

  // Feature 3 — write one history entry for an ended session. Uses refs
  // so the tick interval (shift auto-end) can call it without stale state.
  const saveHistoryEntry = useCallback(() => {
    const startedAtMs =
      sessionStartRef.current ??
      (eventsRef.current.length > 0 ? Date.parse(eventsRef.current[0].at) : Date.now());
    const endedAtMs = Date.now();
    const t = timersRef.current;
    if (t.WORKING + t.AWAY <= 0) return; // Nothing tracked — nothing to save.
    const entry = summarizeSession({
      startedAtMs,
      endedAtMs,
      workingMs: t.WORKING,
      awayMs: t.AWAY,
      eventCount: eventsRef.current.length,
    });
    setHistory((prev) => {
      const next = appendHistory(prev, entry);
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      } catch {
        // Storage unavailable — history still shows for this visit.
      }
      return next;
    });
  }, []);

  const startTimers = useCallback(() => {
    if (tickTimerRef.current) window.clearInterval(tickTimerRef.current);
    tickTimerRef.current = window.setInterval(() => {
      // Timers count ONLY once the AI is actually deciding (models ready
      // and warm-up done). While loading, failed, or still analysing, no
      // second is ever billed as WORKING or AWAY — that is what used to
      // produce fake 100% productivity with no detection at all.
      if (!analyzingRef.current && aiStatusRef.current === 'ready') {
        setTimers((prev) => ({ ...prev, [currentRef.current]: prev[currentRef.current] + 1000 }));
      }

      // No-video-signal check: camera on, but the <video> element has no
      // frames for 3 s straight (shutter closed, wrong device, stuck
      // track). Surfaces as a chip instead of a silent black screen.
      if (runningRef.current) {
        const v = videoRef.current;
        if (v && (v.videoWidth === 0 || v.videoHeight === 0)) {
          noVideoRef.current += 1;
          if (noVideoRef.current >= 3) setNoVideoSignal(true);
        } else {
          noVideoRef.current = 0;
          setNoVideoSignal(false);
        }
      }

      // Feature 4 — away alert, checked in the existing 1 s tick.
      const now = Date.now();
      if (!analyzingRef.current && aiStatusRef.current === 'ready' && currentRef.current === 'AWAY') {
        const awayStretchMs = now - stateSinceRef.current;
        if (shouldAlertAway(awayStretchMs, settingsRef.current.awayAlertMinutes, alertFiredRef.current)) {
          alertFiredRef.current = true;
          setAwayAlert(true);
          if (settingsRef.current.sound) beep();
          if (
            settingsRef.current.notify &&
            'Notification' in window &&
            Notification.permission === 'granted'
          ) {
            try {
              new Notification('Productivity Monitor', {
                body: `Away for ${settingsRef.current.awayAlertMinutes} min`,
              });
            } catch {
              // Notification failed — the visual alert still shows.
            }
          }
        }
      }

      // Feature 5 — shift end: auto-end the session (history is saved
      // inside endSession). Shift start: show the banner once per day.
      if (runningRef.current && isShiftEnded(now, settingsRef.current)) {
        endSessionRef.current();
      } else if (!runningRef.current && isShiftStarted(now, settingsRef.current)) {
        const today = new Date(now).toDateString();
        if (shiftBannerDayRef.current !== today) {
          shiftBannerDayRef.current = today;
          setShiftBanner(true);
        }
      }
    }, 1000);
    if (detectTimerRef.current) window.clearInterval(detectTimerRef.current);
    detectTimerRef.current = window.setInterval(() => {
      void runDetectionFrame();
    }, 500);
  }, [runDetectionFrame, beep]);

  const stopTimers = useCallback(() => {
    for (const ref of [tickTimerRef, detectTimerRef]) {
      if (ref.current) window.clearInterval(ref.current);
      ref.current = null;
    }
  }, []);

  // WEBCAM FIX #2 — camera first, AI second. setRunning(true) happens right
  // after the stream is attached and playing, BEFORE any model import.
  const startCamera = useCallback(async () => {
    setCameraError(null);
    setShowSummary(false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (v) {
        v.srcObject = stream;
        await v.play().catch(() => {});
      }
      // Stream is attached and playing — the camera is live NOW.
      setRunning(true);
      runningRef.current = true;
      setShiftBanner(false); // Feature 5 — the shift banner did its job.
      setAiStatus('loading'); // "Camera on — loading AI…" while models load behind
      const startedAt = Date.now();
      setSessionStart(startedAt);
      sessionStartRef.current = startedAt;
      stateSinceRef.current = startedAt;
      // No assumed person: presence starts unconfirmed, and the warm-up
      // analyses real frames before any state is claimed.
      presenceRef.current = createPresenceState(false);
      analyzedSamplesRef.current = 0;
      analyzingRef.current = true;
      setAnalyzing(true);
      startTimers();
      void loadModelsInBackground(); // background — never blocks the video
    } catch (err) {
      const msg =
        err instanceof DOMException && err.name === 'NotAllowedError'
          ? 'Camera permission denied. Allow camera access in the browser address bar, then press Start Camera again.'
          : 'Could not open the camera. Check that no other app is using it, then try again.';
      setCameraError(msg);
      setAiStatus('error');
    }
  }, [loadModelsInBackground, startTimers]);

  const stopCamera = useCallback(() => {
    stopTimers();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const v = videoRef.current;
    if (v) v.srcObject = null;
    setRunning(false);
    runningRef.current = false;
    setAiStatus('off');
    setNoVideoSignal(false);
    noVideoRef.current = 0;
    analyzingRef.current = false;
    analyzedSamplesRef.current = 0;
    setAnalyzing(false);
  }, [stopTimers]);

  const endSession = useCallback(() => {
    // Feature 3 — the session's summary goes to history before the
    // camera stops (also covers the shift auto-end path).
    saveHistoryEntry();
    stopCamera();
    setShowSummary(true);
  }, [stopCamera, saveHistoryEntry]);

  // Keep the ref fresh so the tick interval can auto-end the session.
  useEffect(() => {
    endSessionRef.current = endSession;
  });

  const downloadCsv = useCallback(() => {
    const csv = eventsToCsv(events);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `productivity-session-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [events]);

  // Feature 3 — download the full history as one report CSV.
  const downloadReportCsv = useCallback(() => {
    const csv = historyToCsv(history);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `productivity-report-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [history]);

  // Feature 4 — notification toggle: permission is asked once; if denied
  // the toggle stays off and never nags again.
  const toggleNotify = useCallback(async (on: boolean) => {
    if (!on) {
      setSettings((prev) => ({ ...prev, notify: false }));
      return;
    }
    if (!('Notification' in window)) return;
    if (Notification.permission === 'granted') {
      setSettings((prev) => ({ ...prev, notify: true }));
    } else if (Notification.permission === 'default') {
      const p = await Notification.requestPermission();
      setSettings((prev) => ({ ...prev, notify: p === 'granted' }));
    }
    // 'denied' -> stays off, no prompt.
  }, []);

  const updateSettings = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  }, []);

  const resetSession = useCallback(() => {
    stopCamera();
    setTimers(EMPTY_TIMERS);
    setEvents([]);
    setSessionStart(null);
    setShowSummary(false);
    setAwayAlert(false);
    alertFiredRef.current = false;
    currentRef.current = 'WORKING';
    setCurrent('WORKING');
    presenceRef.current = createPresenceState(false);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore.
    }
  }, [stopCamera]);

  const sessionMs = sessionStart ? Date.now() - sessionStart : 0;
  const productivity = productivityPercent({
    WORKING: timers.WORKING,
    AWAY: timers.AWAY,
  });
  // DEFECT 2: timeline is derived from the SAME event stream as the log
  // (timelineSegments in src/lib/detection.ts) — never from the separate
  // per-second timer accumulator. A segment's duration therefore always
  // equals the matching log row's "Duration before".
  const timeline = timelineSegments(events, current, Date.now(), sessionStart);
  const longest = events.length
    ? events.reduce((a, b) => (a.durationInPreviousMs > b.durationInPreviousMs ? a : b))
    : null;

  /* --- Presentation-only derived values (no logic change) --- */
  const totalTrackedMs = timers.WORKING + timers.AWAY;
  const workingShare = totalTrackedMs > 0 ? Math.round((timers.WORKING / totalTrackedMs) * 100) : 0;
  const awayShare = totalTrackedMs > 0 ? 100 - workingShare : 0;
  const donutOffset = DONUT_C * (1 - productivity / 100);
  const style = STATE_STYLE[current];
  // Display-only: READY when the camera is off; AI FAILED (red, honest)
  // when the models could not load; GETTING READY (neutral, slightly
  // smaller so the longer word fits) while the warm-up analyses real
  // frames; the real WORKING/AWAY style only after the AI has decided.
  // The detection state itself is untouched by this.
  const displayStyle = !running
    ? { word: 'READY', text: 'text-slate-400', glowText: '', glowCard: '', size: 'text-3xl' }
    : aiStatus === 'error'
      ? { word: 'AI FAILED', text: 'text-red-500', glowText: 'text-glow-away', glowCard: '', size: 'text-2xl' }
      : analyzing
        ? { word: 'GETTING READY', text: 'text-slate-400', glowText: '', glowCard: '', size: 'text-2xl' }
        : { ...style, size: 'text-3xl' };
  // The donut is "live" only when the AI is really deciding; otherwise it
  // is an empty grey ring with "—" — no fake productivity % is ever shown.
  const donutLive = running && !analyzing && aiStatus === 'ready';
  const firstEventAt = events.length > 0 ? new Date(events[0].at).toLocaleTimeString() : null;

  /* --- Features 3/4/5 derived values (presentation only) --- */
  const shiftSet = isShiftValid(settings);
  const outsideShift = running && isOutsideShift(Date.now(), settings);
  const last7 = groupByDay(history, Date.now(), 7);
  const weekWorkingMs = last7.reduce((s, d) => s + d.workingMs, 0);
  const weekAwayMs = last7.reduce((s, d) => s + d.awayMs, 0);
  const weekProductivity = productivityPercent({ WORKING: weekWorkingMs, AWAY: weekAwayMs });
  const weekSessions = last7.reduce((s, d) => s + d.sessions, 0);

  return (
    <div className="flex h-dvh w-full flex-col overflow-hidden bg-mist font-sans text-navy max-lg:h-auto max-lg:min-h-dvh max-lg:overflow-visible">
      {/* ======================= TOP BAR ======================= */}
      <header className="animate-fade-up flex shrink-0 items-center justify-between gap-4 px-4 pt-3 sm:px-5 lg:px-6">
        <div className="flex min-w-0 items-center gap-3">
          {/* Logo mark — bare icon only, same emerald→brand gradient,
              no tile, no border, no shadow (more premium). */}
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" aria-hidden="true" className="shrink-0">
            <defs>
              <linearGradient id="logoGrad" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
                <stop stopColor="#10b981" />
                <stop offset="1" stopColor="#2563eb" />
              </linearGradient>
            </defs>
            <circle cx="12" cy="12" r="8.5" stroke="url(#logoGrad)" strokeWidth="2" />
            <path d="M12 3.5v4M18 18l-3.2-3.2M6 18l3.2-3.2" stroke="url(#logoGrad)" strokeWidth="2" strokeLinecap="round" />
            <circle cx="12" cy="12" r="2.2" fill="url(#logoGrad)" />
          </svg>
          <div className="min-w-0">
            <h1 className="font-display truncate text-2xl font-extrabold tracking-tight sm:text-3xl">
              <span className="text-ink">Productivity</span>{' '}
              <span className="bg-gradient-to-r from-emerald-500 to-brand bg-clip-text text-transparent">Monitor</span>
            </h1>
          </div>
        </div>

        <div className="relative flex shrink-0 items-center gap-2 sm:gap-3">
          {/* LIVE pill — honest: LIVE only while the camera is actually running. */}
          <span
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-bold transition-colors duration-300 ${
              running
                ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/40'
                : 'bg-slate-200 text-slate-600'
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                running ? 'animate-pulse-ring bg-white' : 'bg-slate-400'
              }`}
            />
            {running ? 'LIVE' : 'READY'}
          </span>
          <span className="hidden items-center gap-2 rounded-full bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 ring-1 ring-slate-200 sm:inline-flex">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
              <path d="M12 7v5l3 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <span className="font-mono tabular-nums">Session {formatDuration(sessionMs)}</span>
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500 px-3 py-1.5 text-xs font-bold text-white shadow-lg shadow-emerald-500/30 transition-all duration-300">
            Productivity {productivity}%
          </span>
          {/* Feature 5 — shift chip, only when a shift is configured. */}
          {shiftSet && (
            <span className="hidden rounded-full bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 ring-1 ring-slate-200 md:inline-flex">
              Shift {settings.shiftStart}–{settings.shiftEnd}
            </span>
          )}
          {/* Feature 3 — History button. Feature 5 — settings (clock icon). */}
          <button
            onClick={() => setHistoryOpen(true)}
            className="rounded-full bg-white px-3 py-1.5 text-xs font-bold text-slate-700 ring-1 ring-slate-200 transition-all duration-300 hover:-translate-y-0.5 hover:bg-slate-50"
          >
            History
          </button>
          <button
            onClick={() => setShowSettings((v) => !v)}
            aria-label="Settings"
            className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-slate-600 ring-1 ring-slate-200 transition-all duration-300 hover:-translate-y-0.5 hover:bg-slate-50"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
              <path d="M12 7v5l3 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
          {/* Outside-shift note (Feature 5) — only while running outside shift hours. */}
          {outsideShift && (
            <span className="hidden text-xs font-semibold text-amber-600 sm:inline">Outside shift</span>
          )}

        </div>
      </header>
      {/* Gradient hairline — brand accent under the header. */}
      <div className="mx-4 mt-3 h-px shrink-0 bg-gradient-to-r from-emerald-500 via-brand to-transparent sm:mx-5 lg:mx-6" aria-hidden="true" />

      {/* Feature 4/5 — one banner slot: away alert wins over shift start. */}
      {(awayAlert || shiftBanner) && (
        <div
          className={`mx-4 mt-3 flex shrink-0 items-center justify-between gap-3 rounded-2xl px-4 py-2 text-sm font-semibold sm:mx-5 lg:mx-6 ${
            awayAlert ? 'bg-red-50 text-red-700 ring-1 ring-red-200' : 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200'
          }`}
        >
          <span>
            {awayAlert ? `Away for ${settings.awayAlertMinutes} min` : 'Shift started — Start Camera'}
          </span>
          <button
            onClick={() => {
              setAwayAlert(false);
              setShiftBanner(false);
            }}
            className="shrink-0 rounded-full px-2 py-1 text-xs font-bold opacity-70 hover:opacity-100"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* ======================= MAIN COLUMN ======================= */}
      <main className="relative flex min-h-0 flex-1 flex-col gap-3 px-4 pb-3 pt-3 sm:px-5 lg:px-6 lg:pb-4">
        {/* ---------- Main split: camera left, rail right ---------- */}
        <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[minmax(0,1fr)_350px] xl:grid-cols-[minmax(0,1fr)_380px]">
          {/* ======== Camera card ======== */}
          <section
            className="animate-fade-up flex min-h-0 flex-col rounded-3xl bg-white p-2 ring-1 ring-slate-200/80 transition-shadow duration-300"
            style={{ animationDelay: '240ms' }}
          >
            {/* Frame fills the whole card — no "Live camera" header, no
                status text on the video. Start Camera sits centred under
                "Camera is off"; once running, small controls float at the
                bottom-right. The <video> below is ALWAYS in the DOM
                (never conditional). */}
            <div className="relative mx-auto my-auto aspect-video w-full max-w-[min(100%,calc((100dvh-17rem)*16/9))] overflow-hidden rounded-2xl bg-slate-900">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 h-full w-full object-cover"
              />
              {!running && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-900/70 px-4 text-center text-white">
                  <p className="font-display text-lg font-bold">Camera is off</p>
                  <button
                    onClick={startCamera}
                    className="btn-glow inline-flex items-center gap-2 rounded-2xl bg-brand px-5 py-3 font-sans text-sm font-bold text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-blue-600 active:translate-y-0"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                      <path d="M8 5.5v13l11-6.5-11-6.5Z" />
                    </svg>
                    Start Camera
                  </button>
                </div>
              )}
              {running && aiStatus === 'loading' && (
                <div className="absolute left-3 top-3 rounded-full bg-black/60 px-3 py-1.5 text-xs font-semibold text-white">
                  Camera on — loading AI…
                </div>
              )}
              {running && aiStatus === 'error' && (
                <div className="absolute left-3 top-3 flex items-center gap-2">
                  <div className="rounded-full bg-red-600 px-3 py-1.5 text-xs font-semibold text-white">
                    AI failed to load
                  </div>
                  <button
                    onClick={retryAi}
                    className="rounded-full bg-white px-3 py-1.5 text-xs font-bold text-slate-800 shadow-lg transition-all hover:-translate-y-0.5"
                  >
                    Retry AI
                  </button>
                </div>
              )}
              {running && noVideoSignal && (
                <div className="absolute right-3 top-3 rounded-full bg-amber-500 px-3 py-1.5 text-xs font-semibold text-white">
                  No video signal
                </div>
              )}
              {cameraError && (
                <div className="absolute inset-x-3 top-3 rounded-xl border border-red-200/80 bg-red-50/95 p-3 text-sm text-red-800 shadow-lg">
                  {cameraError}
                </div>
              )}
              {running && (
                <div className="absolute bottom-3 right-3 flex flex-wrap items-center gap-2">
                  <button
                    onClick={endSession}
                    className="rounded-xl bg-navy/90 px-3.5 py-2 text-xs font-bold text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-slate-800 active:translate-y-0"
                  >
                    End Session
                  </button>
                  <button
                    onClick={stopCamera}
                    className="rounded-xl bg-black/55 px-3 py-2 text-xs font-semibold text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-black/70 active:translate-y-0"
                  >
                    Stop Camera
                  </button>
                  <button
                    onClick={resetSession}
                    className="rounded-xl bg-black/55 px-3 py-2 text-xs font-semibold text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-black/70 active:translate-y-0"
                  >
                    Reset Session
                  </button>
                </div>
              )}
            </div>
          </section>

          {/* ======== Right rail ======== */}
          <aside className="flex min-h-0 flex-col gap-3 max-lg:overflow-visible">
            {/* Timeline */}
            <div
              className="animate-fade-up shrink-0 rounded-3xl bg-white p-4 ring-1 ring-slate-200/80"
              style={{ animationDelay: '360ms' }}
            >
              <h2 className="font-display text-base font-bold">Timeline</h2>
              {events.length === 0 ? (
                <p className="mt-2 text-sm text-slate-500">No events yet</p>
              ) : (
                <div className="mt-3 flex h-3.5 w-full overflow-hidden rounded-full bg-slate-100">
                  {timeline.map((seg, i) => {
                    const total = timeline.reduce((sum, x) => sum + x.durationMs, 0) || 1;
                    const width = Math.max((seg.durationMs / total) * 100, 2);
                    const color = seg.state === 'WORKING' ? 'bg-emerald-500' : 'bg-red-500';
                    return (
                      <div
                        key={i}
                        title={`${seg.state} — ${(seg.durationMs / 1000).toFixed(1)}s`}
                        className={`${color} h-full transition-all duration-500`}
                        style={{ width: `${width}%` }}
                      />
                    );
                  })}
                </div>
              )}
              <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                <span>{firstEventAt ? `${firstEventAt} · Start` : 'Start'}</span>
                <span>Now {new Date().toLocaleTimeString()}</span>
              </div>
              <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-600">
                <span>
                  <span className="mr-1 inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" />
                  Working
                </span>
                <span>
                  <span className="mr-1 inline-block h-2.5 w-2.5 rounded-full bg-red-500" />
                  Away
                </span>
              </div>
            </div>

            {/* Event log */}
            <div
              className="animate-fade-up flex min-h-0 flex-1 flex-col rounded-3xl bg-white p-4 ring-1 ring-slate-200/80"
              style={{ animationDelay: '420ms' }}
            >
              <div className="mb-2 flex shrink-0 flex-wrap items-center justify-between gap-2">
                <h2 className="font-display text-base font-bold">Event log</h2>
                <button
                  onClick={downloadCsv}
                  disabled={events.length === 0}
                  className="rounded-xl bg-brand px-3.5 py-2 text-xs font-bold text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-blue-600 disabled:translate-y-0 disabled:opacity-40"
                >
                  Download CSV
                </button>
              </div>
              {events.length === 0 ? (
                <p className="text-sm text-slate-500">No events yet</p>
              ) : (
                <div className="thin-scroll min-h-0 flex-1 overflow-y-auto pr-1">
                  <div className="space-y-2">
                    {[...events]
                      .slice(-12)
                      .reverse()
                      .map((e, i) => (
                        <div
                          key={i}
                          className="flex items-center justify-between gap-2 rounded-xl bg-slate-50 px-3 py-2 text-xs ring-1 ring-slate-100"
                        >
                          <span className="flex min-w-0 items-center gap-2">
                            <span className={`h-2 w-2 shrink-0 rounded-full ${e.to === 'WORKING' ? 'bg-emerald-500' : 'bg-red-500'}`} />
                            <span className="font-mono tabular-nums text-slate-500">
                              {new Date(e.at).toLocaleTimeString()}
                            </span>
                            <span className="truncate font-semibold">
                              {e.from} → {e.to}
                            </span>
                          </span>
                          <span className="shrink-0 font-mono tabular-nums text-slate-500">
                            {(e.durationInPreviousMs / 1000).toFixed(1)}s · {e.confidence.toFixed(2)}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </aside>
        </div>

        {/* ---------- KPI strip — columns match the split above,
            so Working/Away Time align with the camera card and the
            Status card aligns with the right rail ---------- */}
        <section className="grid shrink-0 grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_350px] xl:grid-cols-[minmax(0,1fr)_380px]">
          {/* Working + Away time (under the camera card) */}
          <div className="grid min-w-0 grid-cols-2 gap-3">
          <div
            className="animate-fade-up relative overflow-hidden rounded-2xl bg-white p-3 ring-1 ring-slate-200/80 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-emerald-500/10 sm:p-3.5"
            style={{ animationDelay: '60ms' }}
          >
            <span className="absolute inset-y-0 left-0 w-1.5 bg-emerald-500" />
            <div className="flex h-full flex-col justify-center pl-2">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-600">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                Working Time
              </div>
              <div className="mt-1 font-mono text-xl font-bold tabular-nums sm:text-2xl">
                {formatDuration(timers.WORKING)}
              </div>
              <div className="text-xs text-slate-400">{workingShare}% of tracked session</div>
            </div>
          </div>

          <div
            className="animate-fade-up relative overflow-hidden rounded-2xl bg-white p-3 ring-1 ring-slate-200/80 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-red-500/10 sm:p-3.5"
            style={{ animationDelay: '120ms' }}
          >
            <span className="absolute inset-y-0 left-0 w-1.5 bg-red-500" />
            <div className="flex h-full flex-col justify-center pl-2">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-600">
                <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
                Away Time
              </div>
              <div className="mt-1 font-mono text-xl font-bold tabular-nums sm:text-2xl">
                {formatDuration(timers.AWAY)}
              </div>
              <div className="text-xs text-slate-400">{awayShare}% of tracked session</div>
            </div>
          </div>

          </div>

          {/* Status card, full size (under the right rail) */}
          <div
            className={`animate-fade-up relative overflow-hidden rounded-3xl bg-white p-4 ring-1 ring-slate-200/80 transition-all duration-500 ${displayStyle.glowCard} ${
              awayAlert ? 'animate-pulse-ring-red' : ''
            }`}
            style={{ animationDelay: '180ms' }}
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-xs font-semibold uppercase tracking-widest text-slate-400">Status</div>
                <div className={`mt-1 font-display ${displayStyle.size} font-extrabold tracking-tight transition-colors duration-300 ${displayStyle.text} ${displayStyle.glowText} ${analyzing ? 'animate-pulse' : ''}`}>
                  {displayStyle.word}
                </div>
              </div>
              {/* Donut, original size — % only inside (no inner label, so
                  nothing can clip); "Productivity" lives in the top bar.
                  The donut is live only while the AI is really deciding;
                  otherwise it is an empty grey ring with "—". */}
              <div className="relative h-[104px] w-[104px] shrink-0">
                <svg width="104" height="104" viewBox="0 0 120 120" className={donutLive ? (current === 'WORKING' ? 'donut-glow' : 'donut-glow-away') : ''}>
                  <circle cx="60" cy="60" r={DONUT_R} fill="none" stroke="#e2e8f0" strokeWidth="11" />
                  <circle
                    cx="60"
                    cy="60"
                    r={DONUT_R}
                    fill="none"
                    stroke={!donutLive ? '#cbd5e1' : current === 'WORKING' ? '#10b981' : '#ef4444'}
                    strokeWidth="11"
                    strokeLinecap="round"
                    strokeDasharray={DONUT_C}
                    strokeDashoffset={donutOffset}
                    transform="rotate(-90 60 60)"
                    style={{ transition: 'stroke-dashoffset 0.6s ease, stroke 0.3s ease' }}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="font-display text-2xl font-extrabold tabular-nums">{donutLive ? `${productivity}%` : '—'}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ======================= Feature 3 — History view =======================
            Overlay, not a page swap: the dashboard (and its always-mounted
            <video>) stays in the DOM underneath, so a running session is
            never interrupted by opening History. */}
        {historyOpen && (
          <div className="absolute inset-0 z-20 flex flex-col gap-3 overflow-hidden bg-mist px-4 pb-3 pt-3 sm:px-5 lg:px-6 lg:pb-4">
            <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
              <h2 className="font-display text-xl font-extrabold">History</h2>
              <div className="flex gap-2">
                <button
                  onClick={downloadReportCsv}
                  disabled={history.length === 0}
                  className="rounded-xl bg-brand px-3.5 py-2 text-xs font-bold text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-blue-600 disabled:opacity-40"
                >
                  Download report CSV
                </button>
                <button
                  onClick={() => setHistoryOpen(false)}
                  className="rounded-xl bg-white px-3.5 py-2 text-xs font-bold text-slate-700 ring-1 ring-slate-200 transition-all duration-300 hover:-translate-y-0.5 hover:bg-slate-50"
                >
                  Back
                </button>
              </div>
            </div>
            {history.length === 0 ? (
              <p className="text-sm text-slate-500">No past sessions yet</p>
            ) : (
              <>
                {/* 7-day totals */}
                <div className="grid shrink-0 grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-2xl bg-white p-3 ring-1 ring-slate-200/80">
                    <div className="text-xs font-semibold text-slate-500">Last 7 days</div>
                    <div className="mt-1 font-display text-xl font-extrabold">{weekSessions} sessions</div>
                  </div>
                  <div className="rounded-2xl bg-white p-3 ring-1 ring-slate-200/80">
                    <div className="text-xs font-semibold text-slate-500">Working</div>
                    <div className="mt-1 font-mono text-xl font-bold tabular-nums">{formatDuration(weekWorkingMs)}</div>
                  </div>
                  <div className="rounded-2xl bg-white p-3 ring-1 ring-slate-200/80">
                    <div className="text-xs font-semibold text-slate-500">Away</div>
                    <div className="mt-1 font-mono text-xl font-bold tabular-nums">{formatDuration(weekAwayMs)}</div>
                  </div>
                  <div className="rounded-2xl bg-white p-3 ring-1 ring-slate-200/80">
                    <div className="text-xs font-semibold text-slate-500">Productivity</div>
                    <div className="mt-1 font-display text-xl font-extrabold">{weekProductivity}%</div>
                  </div>
                </div>
                {/* Day bars */}
                <div className="flex min-h-0 flex-1 flex-col justify-center gap-3 overflow-hidden rounded-3xl bg-white p-4 ring-1 ring-slate-200/80">
                  {last7.map((day) => {
                    const total = day.workingMs + day.awayMs;
                    const workingPct = total > 0 ? (day.workingMs / total) * 100 : 0;
                    return (
                      <div key={day.date} className="flex items-center gap-3">
                        <span className="w-24 shrink-0 text-sm font-semibold">{day.label}</span>
                        <div className="flex h-3.5 min-w-0 flex-1 overflow-hidden rounded-full bg-slate-100">
                          {total > 0 ? (
                            <>
                              <div className="h-full bg-emerald-500 transition-all duration-500" style={{ width: `${workingPct}%` }} />
                              <div className="h-full flex-1 bg-red-500 transition-all duration-500" />
                            </>
                          ) : (
                            <div className="h-full w-full bg-slate-200" />
                          )}
                        </div>
                        <span className="w-20 shrink-0 text-right font-mono text-sm tabular-nums">
                          {formatDuration(day.workingMs)}
                        </span>
                        <span className="w-12 shrink-0 text-right font-display text-sm font-bold">
                          {day.productivity}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        )}
      </main>

      {/* Features 4/5 — settings popover: exactly these fields, all real. */}
      {showSettings && (
        <>
          <div className="fixed inset-0 z-50 bg-navy/20" onClick={() => setShowSettings(false)} aria-hidden="true" />
          <div className="fixed right-4 top-[4.25rem] z-[60] w-72 rounded-2xl bg-white p-4 shadow-2xl ring-1 ring-slate-200 sm:right-5 lg:right-6" role="dialog" aria-label="Settings">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-base font-bold">Settings</h2>
              <button
                onClick={() => setShowSettings(false)}
                aria-label="Close settings"
                className="flex h-7 w-7 items-center justify-center rounded-full text-slate-400 hover:bg-slate-100 hover:text-slate-600"
              >
                ✕
              </button>
            </div>
            <div className="mt-3 space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <label className="block">
                  <span className="text-xs font-semibold text-slate-500">Shift start</span>
                  <input
                    type="time"
                    value={settings.shiftStart ?? ''}
                    onChange={(e) => updateSettings({ shiftStart: e.target.value || null })}
                    className="mt-1 w-full rounded-xl bg-slate-50 px-2.5 py-2 font-mono text-sm ring-1 ring-slate-200"
                  />
                </label>
                <label className="block">
                  <span className="text-xs font-semibold text-slate-500">Shift end</span>
                  <input
                    type="time"
                    value={settings.shiftEnd ?? ''}
                    onChange={(e) => updateSettings({ shiftEnd: e.target.value || null })}
                    className="mt-1 w-full rounded-xl bg-slate-50 px-2.5 py-2 font-mono text-sm ring-1 ring-slate-200"
                  />
                </label>
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-500">Working days</span>
                <div className="mt-1 flex gap-1">
                  {[
                    { d: 1, l: 'M' },
                    { d: 2, l: 'T' },
                    { d: 3, l: 'W' },
                    { d: 4, l: 'T' },
                    { d: 5, l: 'F' },
                    { d: 6, l: 'S' },
                    { d: 0, l: 'S' },
                  ].map(({ d, l }) => {
                    const on = settings.workingDays.includes(d);
                    return (
                      <button
                        key={d}
                        onClick={() => {
                          const next = on
                            ? settings.workingDays.filter((x) => x !== d)
                            : [...settings.workingDays, d];
                          if (next.length > 0) updateSettings({ workingDays: next });
                        }}
                        aria-pressed={on}
                        className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold transition-colors ${
                          on ? 'bg-emerald-500 text-white' : 'bg-slate-100 text-slate-500'
                        }`}
                      >
                        {l}
                      </button>
                    );
                  })}
                </div>
              </div>
              <label className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-500">Away alert (min)</span>
                <input
                  type="number"
                  min={1}
                  max={60}
                  value={settings.awayAlertMinutes}
                  onChange={(e) =>
                    updateSettings({
                      awayAlertMinutes: Math.min(60, Math.max(1, Number(e.target.value) || 5)),
                    })
                  }
                  className="w-16 rounded-xl bg-slate-50 px-2.5 py-2 text-center font-mono text-sm ring-1 ring-slate-200"
                />
              </label>
              <label className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-500">Sound</span>
                <input
                  type="checkbox"
                  checked={settings.sound}
                  onChange={(e) => updateSettings({ sound: e.target.checked })}
                  className="h-4 w-4 accent-emerald-500"
                />
              </label>
              <label className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-500">Notification</span>
                <input
                  type="checkbox"
                  checked={settings.notify}
                  onChange={(e) => void toggleNotify(e.target.checked)}
                  className="h-4 w-4 accent-emerald-500"
                />
              </label>
            </div>
          </div>
        </>
      )}

      {/* ======================= End-session summary (overlay) ======================= */}
      {showSummary && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-navy/40 p-4 backdrop-blur-sm">
          <section className="animate-fade-up w-full max-w-lg rounded-3xl bg-white p-6 shadow-2xl ring-1 ring-slate-200">
            <h2 className="font-display text-xl font-extrabold">Session summary</h2>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-2xl bg-slate-50 p-3 ring-1 ring-slate-100">
                <div className="text-slate-500">Productivity</div>
                <div className="font-display text-2xl font-extrabold">{productivity}%</div>
              </div>
              <div className="rounded-2xl bg-slate-50 p-3 ring-1 ring-slate-100">
                <div className="text-slate-500">Session length</div>
                <div className="font-display text-2xl font-extrabold tabular-nums">{formatDuration(sessionMs)}</div>
              </div>
              <div className="rounded-2xl bg-slate-50 p-3 ring-1 ring-slate-100">
                <div className="text-slate-500">State changes</div>
                <div className="font-display text-2xl font-extrabold">{events.length}</div>
              </div>
              <div className="rounded-2xl bg-slate-50 p-3 ring-1 ring-slate-100">
                <div className="text-slate-500">Longest stretch</div>
                <div className="font-display text-2xl font-extrabold">
                  {longest ? `${longest.from} — ${(longest.durationInPreviousMs / 1000).toFixed(0)}s` : '—'}
                </div>
              </div>
            </div>
            <div className="mt-5 flex flex-wrap gap-2">
              <button
                onClick={downloadCsv}
                disabled={events.length === 0}
                className="rounded-xl bg-brand px-4 py-2.5 text-sm font-bold text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-blue-600 disabled:opacity-40"
              >
                Download session CSV
              </button>
              <button
                onClick={resetSession}
                className="rounded-xl bg-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-700 transition-all duration-300 hover:-translate-y-0.5 hover:bg-slate-300"
              >
                Start a new session
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
