import { describe, expect, it } from 'vitest';
import {
  appendHistory,
  dateKey,
  DEFAULT_SETTINGS,
  groupByDay,
  historyToCsv,
  HISTORY_MAX_ENTRIES,
  isOutsideShift,
  isShiftEnded,
  isShiftStarted,
  isShiftValid,
  isWithinShift,
  parseHistory,
  parseSettings,
  shouldAlertAway,
  summarizeSession,
  timeToMinutes,
  type HistoryEntry,
  type Settings,
} from './features';

const entry = (over: Partial<HistoryEntry> = {}): HistoryEntry => ({
  date: '2026-11-10',
  startedAt: '2026-11-10T04:30:00.000Z',
  endedAt: '2026-11-10T13:30:00.000Z',
  workingMs: 6 * 3600_000,
  awayMs: 2 * 3600_000,
  productivity: 75,
  eventCount: 12,
  ...over,
});

describe('parseSettings', () => {
  it('returns defaults for null/corrupt input', () => {
    expect(parseSettings(null)).toEqual(DEFAULT_SETTINGS);
    expect(parseSettings('nope')).toEqual(DEFAULT_SETTINGS);
  });

  it('keeps valid fields and clamps alert minutes to 1–60', () => {
    const s = parseSettings({ awayAlertMinutes: 99, sound: false, shiftStart: '10:00', shiftEnd: '19:00' });
    expect(s.awayAlertMinutes).toBe(60);
    expect(s.sound).toBe(false);
    expect(s.shiftStart).toBe('10:00');
    expect(s.shiftEnd).toBe('19:00');
  });

  it('drops invalid shift times and empty working days', () => {
    const s = parseSettings({ shiftStart: '25:00', shiftEnd: 'x', workingDays: [] });
    expect(s.shiftStart).toBeNull();
    expect(s.shiftEnd).toBeNull();
    expect(s.workingDays).toEqual(DEFAULT_SETTINGS.workingDays);
  });
});

describe('summarizeSession / appendHistory / parseHistory', () => {
  it('builds an entry with the right productivity and local date', () => {
    const e = summarizeSession({
      startedAtMs: Date.UTC(2026, 10, 10, 4, 30),
      endedAtMs: Date.UTC(2026, 10, 10, 13, 30),
      workingMs: 3 * 3600_000,
      awayMs: 1 * 3600_000,
      eventCount: 4,
    });
    expect(e.productivity).toBe(75);
    expect(e.eventCount).toBe(4);
    expect(e.date).toBe(dateKey(Date.UTC(2026, 10, 10, 4, 30)));
  });

  it('returns 0 productivity for an empty session', () => {
    const e = summarizeSession({ startedAtMs: 1, endedAtMs: 2, workingMs: 0, awayMs: 0, eventCount: 0 });
    expect(e.productivity).toBe(0);
  });

  it('caps history at the newest 100 entries', () => {
    let list: HistoryEntry[] = [];
    for (let i = 0; i < HISTORY_MAX_ENTRIES + 5; i++) list = appendHistory(list, entry({ date: `d${i}` }));
    expect(list).toHaveLength(HISTORY_MAX_ENTRIES);
    expect(list[0].date).toBe('d5');
    expect(list[list.length - 1].date).toBe(`d${HISTORY_MAX_ENTRIES + 4}`);
  });

  it('parseHistory drops wrong-shaped entries', () => {
    expect(parseHistory([entry(), { date: 'x' }, null, 'junk'])).toHaveLength(1);
    expect(parseHistory(null)).toEqual([]);
  });
});

describe('groupByDay', () => {
  it('aggregates the last 7 days oldest-first, zero for empty days', () => {
    const now = new Date(2026, 10, 10, 12, 0, 0).getTime(); // Tue 10 Nov 2026 local
    const days = groupByDay(
      [entry({ date: dateKey(now), workingMs: 3600_000, awayMs: 3600_000, productivity: 50 })],
      now,
      7,
    );
    expect(days).toHaveLength(7);
    expect(days[6].date).toBe(dateKey(now));
    expect(days[6].workingMs).toBe(3600_000);
    expect(days[6].productivity).toBe(50);
    expect(days[6].sessions).toBe(1);
    expect(days[0].workingMs).toBe(0);
    expect(days[0].productivity).toBe(0);
    expect(days[0].sessions).toBe(0);
  });
});

describe('historyToCsv', () => {
  it('writes header + one row per entry', () => {
    const csv = historyToCsv([entry()]);
    const lines = csv.split('\n');
    expect(lines[0]).toContain('productivity_percent');
    expect(lines).toHaveLength(2);
    expect(lines[1]).toContain('2026-11-10');
    expect(lines[1]).toContain('21600.0'); // 6h working seconds
  });
});

describe('shift math', () => {
  const shift: Settings = { ...DEFAULT_SETTINGS, shiftStart: '10:00', shiftEnd: '19:00' };

  it('timeToMinutes parses HH:MM and rejects bad input', () => {
    expect(timeToMinutes('10:00')).toBe(600);
    expect(timeToMinutes('19:30')).toBe(1170);
    expect(timeToMinutes(null)).toBeNull();
    expect(timeToMinutes('25:00')).toBeNull();
    expect(timeToMinutes('9:00')).toBeNull();
  });

  it('isShiftValid needs both times with end after start (no midnight crossing)', () => {
    expect(isShiftValid(shift)).toBe(true);
    expect(isShiftValid({ ...shift, shiftEnd: null })).toBe(false);
    expect(isShiftValid({ ...shift, shiftStart: '19:00', shiftEnd: '10:00' })).toBe(false);
    expect(isShiftValid({ ...shift, shiftStart: '10:00', shiftEnd: '10:00' })).toBe(false);
  });

  it('within / outside / ended / started follow the local clock', () => {
    const monday = new Date(2026, 10, 9, 12, 0, 0).getTime(); // Mon 9 Nov 2026
    expect(isWithinShift(monday, shift)).toBe(true);
    expect(isOutsideShift(monday, shift)).toBe(false);
    expect(isShiftStarted(monday, shift)).toBe(true);
    expect(isShiftEnded(monday, shift)).toBe(false);

    const before = new Date(2026, 10, 9, 9, 0, 0).getTime();
    expect(isWithinShift(before, shift)).toBe(false);
    expect(isOutsideShift(before, shift)).toBe(true);
    expect(isShiftEnded(before, shift)).toBe(false);

    const after = new Date(2026, 10, 9, 19, 0, 0).getTime();
    expect(isWithinShift(after, shift)).toBe(false);
    expect(isShiftEnded(after, shift)).toBe(true);

    const sunday = new Date(2026, 10, 8, 12, 0, 0).getTime(); // Sun — not a working day
    expect(isWithinShift(sunday, shift)).toBe(false);
    expect(isOutsideShift(sunday, shift)).toBe(true);
    expect(isShiftEnded(sunday, shift)).toBe(false);
  });

  it('no shift configured means never outside and never ended', () => {
    const t = new Date(2026, 10, 9, 12, 0, 0).getTime();
    expect(isWithinShift(t, DEFAULT_SETTINGS)).toBe(false);
    expect(isOutsideShift(t, DEFAULT_SETTINGS)).toBe(false);
    expect(isShiftEnded(t, DEFAULT_SETTINGS)).toBe(false);
    expect(isShiftStarted(t, DEFAULT_SETTINGS)).toBe(false);
  });
});

describe('shouldAlertAway', () => {
  it('fires at the threshold, once per stretch', () => {
    expect(shouldAlertAway(5 * 60000, 5, false)).toBe(true);
    expect(shouldAlertAway(4 * 60000 + 59999, 5, false)).toBe(false);
    expect(shouldAlertAway(10 * 60000, 5, true)).toBe(false);
    expect(shouldAlertAway(3 * 60000, 3, false)).toBe(true);
  });
});
