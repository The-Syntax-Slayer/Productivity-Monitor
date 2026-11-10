/**
 * Detection rules — ported from the cloned open-source repo
 * PathumRathnayaka/AI-powered-employee-productivity-monitoring-system (MIT).
 *
 * Only TWO states remain (user decision, 9 Nov 2026): WORKING and AWAY.
 * PHONE and SLEEPING detection were removed entirely — the clone's
 * phone_detector.py and sleep_pose_detector.py rules are gone.
 *
 * Sources still used in the clone:
 * - ai_engine/logic/away_detector.py   -> the away-confirmation idea
 * - backend/services/summary_generator.py -> productivityPercent()
 *
 * Pure functions only — no DOM, no TensorFlow, no WorkScope code.
 *
 * AWAY FLICKER FIX (user-reported, tested on a real laptop 9 Nov 2026):
 * walking away looped AWAY 1s -> WORKING -> AWAY -> WORKING. Two causes:
 *   1. classifyFrame() returned AWAY on ANY single frame with no person
 *      (`awayConfirmed || !personPresent`), so one missed frame flipped the
 *      state instantly — the 3 s threshold was never actually enforced.
 *   2. MoveNet returns a pose object even with nobody in frame, and the old
 *      code counted ANY pose as a person (no score check at all), so a
 *      stray low-score pose on the background counted as "person present"
 *      and flipped AWAY straight back to WORKING.
 * Fix: (a) a strict person-present quality gate (below), and (b)
 * consecutive-counter hysteresis in both directions — a single frame can
 * NEVER flip state. The old 5-sample/60%-majority buffer was REPLACED by
 * this design: a majority buffer lets 2 stray hits in 5 samples flip the
 * state, which is exactly the flicker the user saw.
 */

export type MonitorState = 'WORKING' | 'AWAY';

/**
 * Person-present quality gate.
 *
 * MoveNet SINGLEPOSE_LIGHTNING always returns a pose, even on an empty
 * chair / background, with a very low overall score. A pose only counts
 * as a person when ALL of these hold:
 *   - overall pose score >= 0.3
 *   - at least 5 keypoints with score >= 0.3 (a real body, not noise)
 *   - at least one anchor keypoint (nose / shoulder / hip) >= 0.3, so a
 *     partial body at the edge of frame still counts, but a random
 *     background blob does not.
 */
export const POSE_MIN_SCORE = 0.3;
export const MIN_HIGH_CONF_KEYPOINTS = 5;
export const KEYPOINTS_FOR_PRESENCE = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'] as const;

export type KeypointLike = { name: string; score: number };

export function isPosePerson(pose: { score?: number | null; keypoints: KeypointLike[] }): boolean {
  const poseScore = pose.score ?? 0;
  if (poseScore < POSE_MIN_SCORE) return false;
  const highConf = pose.keypoints.filter((k) => (k.score ?? 0) >= POSE_MIN_SCORE);
  if (highConf.length < MIN_HIGH_CONF_KEYPOINTS) return false;
  const anchorSeen = KEYPOINTS_FOR_PRESENCE.some((name) =>
    pose.keypoints.some((k) => k.name === name && (k.score ?? 0) >= POSE_MIN_SCORE),
  );
  return anchorSeen;
}

/** COCO-SSD person box only counts as a person at score >= 0.5. */
export const COCO_PERSON_MIN_SCORE = 0.5;

export function isCocoPerson(score: number): boolean {
  return score >= COCO_PERSON_MIN_SCORE;
}

/**
 * Hysteresis — consecutive counters, both directions.
 * Detection runs every 500 ms (SAMPLE_INTERVAL_MS), so:
 *   - AWAY only after 6 consecutive no-person samples (~3 s of nobody).
 *   - Back to WORKING only after 3 consecutive person samples (~1.5 s).
 * A single present/absent frame only moves a streak by 1 and resets the
 * opposite streak — it can never flip the confirmed state on its own.
 * Alternating present/absent frames therefore never flip state at all.
 */
export const SAMPLE_INTERVAL_MS = 500;
export const AWAY_AFTER_SAMPLES = 6;
export const RETURN_AFTER_SAMPLES = 3;
/** ~3 s of nobody before AWAY (6 samples x 500 ms). Kept for reference. */
export const AWAY_AFTER_MS = AWAY_AFTER_SAMPLES * SAMPLE_INTERVAL_MS;

export type PresenceState = {
  confirmedPresent: boolean;
  presentStreak: number;
  absentStreak: number;
};

export function createPresenceState(initiallyPresent = true): PresenceState {
  return { confirmedPresent: initiallyPresent, presentStreak: 0, absentStreak: 0 };
}

export function updatePresence(prev: PresenceState, present: boolean): PresenceState {
  if (present) {
    const presentStreak = prev.presentStreak + 1;
    return {
      confirmedPresent: prev.confirmedPresent || presentStreak >= RETURN_AFTER_SAMPLES,
      presentStreak,
      absentStreak: 0,
    };
  }
  const absentStreak = prev.absentStreak + 1;
  return {
    confirmedPresent: prev.confirmedPresent && absentStreak < AWAY_AFTER_SAMPLES,
    presentStreak: 0,
    absentStreak,
  };
}

/** State for one frame: only the hysteresis-confirmed presence matters. */
export function classifyFrame(input: { confirmedPresent: boolean }): MonitorState {
  return input.confirmedPresent ? 'WORKING' : 'AWAY';
}

/** Productivity % = Working ÷ (Working + Away) x 100. */
export function productivityPercent(timersMs: Record<MonitorState, number>): number {
  const working = timersMs.WORKING;
  const total = timersMs.WORKING + timersMs.AWAY;
  if (total <= 0) return 0;
  return Math.round((working / total) * 100);
}

/**
 * Timeline and event log share ONE source of truth.
 *
 * Each event records `durationInPreviousMs`: the time spent in `from`
 * before moving to `to`. So the timeline segment for that stretch is
 * the FROM state with that duration — NOT the TO state.
 *
 * This builds the timeline from the event stream only (never from the
 * separate per-second timer accumulator):
 *   - one segment per event: { state: e.from, durationMs: e.durationInPreviousMs }
 *   - plus one trailing segment for the state still in progress:
 *     current state, duration = now - start of the current stretch
 *     (last event's `at`, or the session start when there are no events yet).
 * A segment's duration therefore always equals the matching log row's
 * "Duration before" for the same from → to transition.
 */
export type TimelineSegment = { state: MonitorState; durationMs: number };

export type TimelineEvent = {
  at: string;
  from: MonitorState;
  to: MonitorState;
  durationInPreviousMs: number;
};

export function timelineSegments(
  events: TimelineEvent[],
  current: MonitorState,
  nowMs: number,
  sessionStartMs: number | null,
): TimelineSegment[] {
  const segments: TimelineSegment[] = events.map((e) => ({
    state: e.from,
    durationMs: e.durationInPreviousMs,
  }));
  const currentStretchStartMs =
    events.length > 0 ? Date.parse(events[events.length - 1].at) : sessionStartMs;
  if (currentStretchStartMs !== null && !Number.isNaN(currentStretchStartMs)) {
    segments.push({ state: current, durationMs: Math.max(0, nowMs - currentStretchStartMs) });
  }
  return segments;
}

export function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

export function eventsToCsv(
  events: { at: string; from: string; to: string; durationInPreviousMs: number; engine: string; confidence: number }[],
): string {
  const header = 'timestamp,from_state,to_state,duration_in_previous_state_seconds,engine,confidence';
  const rows = events.map((e) =>
    [e.at, e.from, e.to, (e.durationInPreviousMs / 1000).toFixed(1), e.engine, e.confidence.toFixed(2)].join(','),
  );
  return [header, ...rows].join('\n');
}
