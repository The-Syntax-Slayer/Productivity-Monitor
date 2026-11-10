import { describe, expect, it } from 'vitest';
import {
  AWAY_AFTER_MS,
  AWAY_AFTER_SAMPLES,
  RETURN_AFTER_SAMPLES,
  classifyFrame,
  createPresenceState,
  eventsToCsv,
  formatDuration,
  isCocoPerson,
  isPosePerson,
  productivityPercent,
  timelineSegments,
  updatePresence,
} from './detection';

/** Drive the hysteresis with a sample list; returns the final PresenceState. */
function runSamples(samples: boolean[], start = createPresenceState(true)) {
  return samples.reduce((s, present) => updatePresence(s, present), start);
}

describe('person-present quality gate — a stray low-score pose must not count', () => {
  const strongKp = (name: string, score = 0.9) => ({ name, score });
  const goodPose = {
    score: 0.8,
    keypoints: [
      strongKp('nose'), strongKp('left_eye'), strongKp('right_eye'),
      strongKp('left_shoulder'), strongKp('right_shoulder'),
      strongKp('left_hip'), strongKp('right_hip'),
    ],
  };

  it('a real person (high pose score, many high-conf keypoints) counts', () => {
    expect(isPosePerson(goodPose)).toBe(true);
  });
  it('a low-score pose on the background does not count', () => {
    expect(isPosePerson({ ...goodPose, score: 0.12 })).toBe(false);
    expect(isPosePerson({ ...goodPose, score: 0.29 })).toBe(false);
  });
  it('too few high-confidence keypoints does not count', () => {
    const sparse = {
      score: 0.8,
      keypoints: [strongKp('nose'), strongKp('left_shoulder'), { name: 'left_hip', score: 0.1 }, { name: 'right_hip', score: 0.05 }],
    };
    expect(isPosePerson(sparse)).toBe(false);
  });
  it('no anchor keypoint (nose/shoulders/hips) visible does not count', () => {
    const noAnchor = {
      score: 0.8,
      keypoints: [
        strongKp('left_eye'), strongKp('right_eye'), strongKp('left_ear'),
        strongKp('right_ear'), strongKp('left_wrist'), strongKp('right_wrist'),
      ],
    };
    expect(isPosePerson(noAnchor)).toBe(false);
  });
  it('a partial body at the frame edge (one anchor + enough keypoints) still counts', () => {
    const partial = {
      score: 0.5,
      keypoints: [
        strongKp('nose'), strongKp('left_shoulder'), strongKp('right_shoulder'),
        strongKp('left_elbow'), strongKp('right_elbow'), strongKp('left_wrist'),
      ],
    };
    expect(isPosePerson(partial)).toBe(true);
  });
  it('COCO-SSD person box needs score >= 0.5', () => {
    expect(isCocoPerson(0.9)).toBe(true);
    expect(isCocoPerson(0.5)).toBe(true);
    expect(isCocoPerson(0.49)).toBe(false);
    expect(isCocoPerson(0.2)).toBe(false);
  });
});

describe('AWAY hysteresis — consecutive counters, both directions (flicker fix)', () => {
  it('(i) alternating present/absent frames never flip state', () => {
    const end = runSamples([false, true, false, true, false, true, false, true, false, true]);
    expect(end.confirmedPresent).toBe(true); // still WORKING
    const awayEnd = runSamples(
      [false, true, false, true, false, true, false, true],
      { confirmedPresent: false, presentStreak: 0, absentStreak: 0 },
    );
    expect(awayEnd.confirmedPresent).toBe(false); // still AWAY
  });
  it('(ii) 6 consecutive absent samples flip WORKING → AWAY (3 s at 500 ms/sample)', () => {
    expect(AWAY_AFTER_SAMPLES).toBe(6);
    expect(AWAY_AFTER_MS).toBe(3000);
    const after5 = runSamples([false, false, false, false, false]);
    expect(after5.confirmedPresent).toBe(true); // 5 absent = not yet away
    const after6 = runSamples([false, false, false, false, false, false]);
    expect(after6.confirmedPresent).toBe(false); // 6th absent flips to AWAY
    expect(classifyFrame({ confirmedPresent: after6.confirmedPresent })).toBe('AWAY');
  });
  it('(iii) a single present frame while AWAY does NOT flip back to WORKING', () => {
    const away = { confirmedPresent: false, presentStreak: 0, absentStreak: 6 };
    const one = updatePresence(away, true);
    expect(one.confirmedPresent).toBe(false);
    expect(one.presentStreak).toBe(1);
    const two = updatePresence(one, true);
    expect(two.confirmedPresent).toBe(false); // 2 present still not enough
  });
  it('(iv) 3 consecutive present frames flip AWAY → WORKING', () => {
    expect(RETURN_AFTER_SAMPLES).toBe(3);
    const away = { confirmedPresent: false, presentStreak: 0, absentStreak: 6 };
    const end = runSamples([true, true, true], away);
    expect(end.confirmedPresent).toBe(true);
    expect(classifyFrame({ confirmedPresent: end.confirmedPresent })).toBe('WORKING');
  });
  it('the user walk-out scenario: absent streak interrupted by one stray hit stays AWAY-bound', () => {
    // Walk out: 5 absent, 1 stray background pose hit, then absent again.
    // The stray hit resets the absent streak (a real reappearance should),
    // but it does NOT flip anything back — 6 more absent still flip to AWAY.
    let s = createPresenceState(true);
    for (const p of [false, false, false, false, false, true, false, false, false, false, false, false]) {
      s = updatePresence(s, p);
    }
    expect(s.confirmedPresent).toBe(false);
  });
  it('classifyFrame is just the confirmed presence', () => {
    expect(classifyFrame({ confirmedPresent: true })).toBe('WORKING');
    expect(classifyFrame({ confirmedPresent: false })).toBe('AWAY');
  });
});

describe('productivity math — Working ÷ (Working + Away)', () => {
  it('all working = 100%', () => {
    expect(productivityPercent({ WORKING: 60000, AWAY: 0 })).toBe(100);
  });
  it('half working, half away = 50%', () => {
    expect(productivityPercent({ WORKING: 30000, AWAY: 30000 })).toBe(50);
  });
  it('no time at all = 0, not NaN', () => {
    expect(productivityPercent({ WORKING: 0, AWAY: 0 })).toBe(0);
  });
});

describe('helpers', () => {
  it('formats durations as HH:MM:SS', () => {
    expect(formatDuration(3661000)).toBe('01:01:01');
    expect(formatDuration(5000)).toBe('00:00:05');
  });
  it('CSV has the right header and one row per event', () => {
    const csv = eventsToCsv([
      { at: '2026-11-09T10:00:00', from: 'WORKING', to: 'AWAY', durationInPreviousMs: 120000, engine: 'MoveNet', confidence: 0.87 },
    ]);
    const lines = csv.split('\n');
    expect(lines[0]).toBe('timestamp,from_state,to_state,duration_in_previous_state_seconds,engine,confidence');
    expect(lines[1]).toBe('2026-11-09T10:00:00,WORKING,AWAY,120.0,MoveNet,0.87');
  });
});

describe('timeline segments must equal the event log durations', () => {
  // Regression test: the timeline labelled each event's duration by its TO
  // state, but the duration belongs to the FROM state (time spent before
  // the transition).
  const events = [
    { at: '2026-11-09T10:00:03.000Z', from: 'WORKING' as const, to: 'AWAY' as const, durationInPreviousMs: 3000 },
    { at: '2026-11-09T10:00:03.100Z', from: 'AWAY' as const, to: 'WORKING' as const, durationInPreviousMs: 100 },
    { at: '2026-11-09T10:00:08.500Z', from: 'WORKING' as const, to: 'AWAY' as const, durationInPreviousMs: 5400 },
  ];

  it('each segment takes the FROM state and the log duration — not the TO state', () => {
    const segs = timelineSegments(events, 'AWAY', Date.parse('2026-11-09T10:00:08.500Z'), null);
    expect(segs[0]).toEqual({ state: 'WORKING', durationMs: 3000 });
    expect(segs[1]).toEqual({ state: 'AWAY', durationMs: 100 });
    expect(segs[2]).toEqual({ state: 'WORKING', durationMs: 5400 });
    // No segment may claim AWAY lasted 3 s — that 3 s was WORKING.
    expect(segs.find((s) => s.state === 'AWAY')).toEqual({ state: 'AWAY', durationMs: 100 });
  });
  it('timeline total equals the sum of the log durations plus the current stretch', () => {
    const now = Date.parse('2026-11-09T10:00:10.500Z'); // 2 s into AWAY
    const segs = timelineSegments(events, 'AWAY', now, null);
    const total = segs.reduce((sum, s) => sum + s.durationMs, 0);
    expect(total).toBe(3000 + 100 + 5400 + 2000);
    expect(segs[segs.length - 1]).toEqual({ state: 'AWAY', durationMs: 2000 });
  });
  it('with no events yet, the timeline is just the current state since session start', () => {
    const segs = timelineSegments([], 'WORKING', 5000, 1000);
    expect(segs).toEqual([{ state: 'WORKING', durationMs: 4000 }]);
  });
});
