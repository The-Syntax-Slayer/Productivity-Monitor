/**
 * Features 3, 4, 5 — pure functions only (no DOM, no localStorage here).
 * The page component owns all storage / timers / audio side effects.
 *
 * Feature 3 — past sessions history (day-wise reports).
 * Feature 4 — away alert (long AWAY stretch -> one alert per stretch).
 * Feature 5 — shift timing (auto end at shift end, start banner, outside note).
 *
 * Detection rules are NOT here — they live in src/lib/detection.ts and
 * are untouched by these features.
 */

import { productivityPercent } from './detection';

/* ============================== Settings ============================== */

export type Settings = {
  /** Shift start "HH:MM" local time, or null = no shift configured. */
  shiftStart: string | null;
  /** Shift end "HH:MM" local time, or null = no shift configured. */
  shiftEnd: string | null;
  /** JS getDay() values (0 = Sun … 6 = Sat) that are working days. */
  workingDays: number[];
  /** Continuous AWAY minutes before the alert fires. */
  awayAlertMinutes: number;
  /** Play the beep when the away alert fires. */
  sound: boolean;
  /** Also fire a browser notification when the away alert fires. */
  notify: boolean;
};

export const DEFAULT_SETTINGS: Settings = {
  shiftStart: null,
  shiftEnd: null,
  workingDays: [1, 2, 3, 4, 5, 6], // Mon–Sat
  awayAlertMinutes: 5,
  sound: true,
  notify: false,
};

export const SETTINGS_KEY = 'clone-productivity-monitor-settings-v1';

/** Merge a saved (possibly partial / corrupt) settings object over defaults. */
export function parseSettings(raw: unknown): Settings {
  const s = (raw && typeof raw === 'object' ? raw : {}) as Partial<Settings>;
  const minutes =
    typeof s.awayAlertMinutes === 'number' && Number.isFinite(s.awayAlertMinutes)
      ? Math.min(60, Math.max(1, Math.round(s.awayAlertMinutes)))
      : DEFAULT_SETTINGS.awayAlertMinutes;
  const days = Array.isArray(s.workingDays)
    ? s.workingDays.filter((d) => Number.isInteger(d) && d >= 0 && d <= 6)
    : DEFAULT_SETTINGS.workingDays;
  return {
    shiftStart:
      typeof s.shiftStart === 'string' && timeToMinutes(s.shiftStart) !== null ? s.shiftStart : null,
    shiftEnd: typeof s.shiftEnd === 'string' && timeToMinutes(s.shiftEnd) !== null ? s.shiftEnd : null,
    workingDays: days.length > 0 ? days : DEFAULT_SETTINGS.workingDays,
    awayAlertMinutes: minutes,
    sound: typeof s.sound === 'boolean' ? s.sound : true,
    notify: typeof s.notify === 'boolean' ? s.notify : false,
  };
}

/* ============================== History ============================== */

export type HistoryEntry = {
  /** Local date "YYYY-MM-DD" of the session start. */
  date: string;
  startedAt: string; // ISO
  endedAt: string; // ISO
  workingMs: number;
  awayMs: number;
  productivity: number; // 0–100
  eventCount: number;
};

export const HISTORY_KEY = 'clone-productivity-monitor-history-v1';
export const HISTORY_MAX_ENTRIES = 100;

/** Local "YYYY-MM-DD" for a timestamp (local timezone, not UTC). */
export function dateKey(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** Build one history entry from an ended session's totals. */
export function summarizeSession(input: {
  startedAtMs: number;
  endedAtMs: number;
  workingMs: number;
  awayMs: number;
  eventCount: number;
}): HistoryEntry {
  return {
    date: dateKey(input.startedAtMs),
    startedAt: new Date(input.startedAtMs).toISOString(),
    endedAt: new Date(input.endedAtMs).toISOString(),
    workingMs: Math.max(0, Math.round(input.workingMs)),
    awayMs: Math.max(0, Math.round(input.awayMs)),
    productivity: productivityPercent({ WORKING: input.workingMs, AWAY: input.awayMs }),
    eventCount: Math.max(0, input.eventCount),
  };
}

/** Append one entry, keeping only the newest HISTORY_MAX_ENTRIES. */
export function appendHistory(list: HistoryEntry[], entry: HistoryEntry): HistoryEntry[] {
  return [...list, entry].slice(-HISTORY_MAX_ENTRIES);
}

/** Parse a saved history list; corrupt / wrong-shaped entries are dropped. */
export function parseHistory(raw: unknown): HistoryEntry[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (e): e is HistoryEntry =>
      !!e &&
      typeof e === 'object' &&
      typeof (e as HistoryEntry).date === 'string' &&
      typeof (e as HistoryEntry).startedAt === 'string' &&
      typeof (e as HistoryEntry).endedAt === 'string' &&
      typeof (e as HistoryEntry).workingMs === 'number' &&
      typeof (e as HistoryEntry).awayMs === 'number' &&
      typeof (e as HistoryEntry).productivity === 'number' &&
      typeof (e as HistoryEntry).eventCount === 'number',
  );
}

export type DayAggregate = {
  date: string;
  label: string; // e.g. "Mon 10 Nov"
  workingMs: number;
  awayMs: number;
  productivity: number; // 0–100, 0 when nothing tracked that day
  sessions: number;
};

/**
 * Aggregate entries into the last `days` calendar days ending today
 * (local time), oldest first. Days with no sessions get zero totals.
 */
export function groupByDay(entries: HistoryEntry[], nowMs: number, days = 7): DayAggregate[] {
  const out: DayAggregate[] = [];
  const today = new Date(nowMs);
  today.setHours(0, 0, 0, 0);
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const key = dateKey(d.getTime());
    const dayEntries = entries.filter((e) => e.date === key);
    const workingMs = dayEntries.reduce((s, e) => s + e.workingMs, 0);
    const awayMs = dayEntries.reduce((s, e) => s + e.awayMs, 0);
    out.push({
      date: key,
      label: d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }),
      workingMs,
      awayMs,
      productivity: productivityPercent({ WORKING: workingMs, AWAY: awayMs }),
      sessions: dayEntries.length,
    });
  }
  return out;
}

/** Full history as CSV (one row per ended session). */
export function historyToCsv(entries: HistoryEntry[]): string {
  const header =
    'date,started_at,ended_at,working_seconds,away_seconds,productivity_percent,events';
  const rows = entries.map((e) =>
    [
      e.date,
      e.startedAt,
      e.endedAt,
      (e.workingMs / 1000).toFixed(1),
      (e.awayMs / 1000).toFixed(1),
      e.productivity,
      e.eventCount,
    ].join(','),
  );
  return [header, ...rows].join('\n');
}

/* ============================== Shift ============================== */

/** "HH:MM" -> minutes since midnight, or null when unset/invalid. */
export function timeToMinutes(t: string | null | undefined): number | null {
  if (typeof t !== 'string') return null;
  const m = /^(\d{2}):(\d{2})$/.exec(t);
  if (!m) return null;
  const h = Number(m[1]);
  const min = Number(m[2]);
  if (h > 23 || min > 59) return null;
  return h * 60 + min;
}

/**
 * A shift is usable only when both start and end are set and end is
 * strictly after start — same-day shifts only, no midnight crossing.
 */
export function isShiftValid(settings: Settings): boolean {
  const start = timeToMinutes(settings.shiftStart);
  const end = timeToMinutes(settings.shiftEnd);
  return start !== null && end !== null && end > start;
}

function shiftWindowMs(nowMs: number, settings: Settings): { startMs: number; endMs: number } | null {
  if (!isShiftValid(settings)) return null;
  const d = new Date(nowMs);
  if (!settings.workingDays.includes(d.getDay())) return null;
  const start = timeToMinutes(settings.shiftStart)!;
  const end = timeToMinutes(settings.shiftEnd)!;
  const base = new Date(d);
  base.setHours(0, 0, 0, 0);
  const baseMs = base.getTime();
  return { startMs: baseMs + start * 60000, endMs: baseMs + end * 60000 };
}

/** True when a valid shift is set, today is a working day, and now is inside it. */
export function isWithinShift(nowMs: number, settings: Settings): boolean {
  const w = shiftWindowMs(nowMs, settings);
  return !!w && nowMs >= w.startMs && nowMs < w.endMs;
}

/** True when a shift is configured but now is outside it (or not a working day). */
export function isOutsideShift(nowMs: number, settings: Settings): boolean {
  return isShiftValid(settings) && !isWithinShift(nowMs, settings);
}

/** True when the shift has ended today (auto-end trigger). */
export function isShiftEnded(nowMs: number, settings: Settings): boolean {
  const w = shiftWindowMs(nowMs, settings);
  return !!w && nowMs >= w.endMs;
}

/** True when the shift has started and not yet ended today (start banner). */
export function isShiftStarted(nowMs: number, settings: Settings): boolean {
  return isWithinShift(nowMs, settings);
}

/* ============================== Away alert ============================== */

/**
 * Fire the alert when a continuous AWAY stretch reaches the configured
 * minutes, and only once per stretch (fired = already alerted this stretch).
 */
export function shouldAlertAway(awayStretchMs: number, alertMinutes: number, fired: boolean): boolean {
  if (fired) return false;
  return awayStretchMs >= alertMinutes * 60000;
}
