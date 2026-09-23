"""Frame measurements turned into a timeline of acoustic regions.

Every threshold here is a guess about where one kind of sound stops and another
begins. They live together at the top of the file so they can be retuned on real
recordings, and nothing in this module is ever reported as a measurement.
"""

import math

import numpy as np
from parselmouth.praat import call

SILENCE_DROP = 35.0  # decibels below the peak counts as silence
VOICING_HNR = 4.0  # harmonics-to-noise ratio that counts as voiced
FRICATION_COG = 2500.0  # centre of gravity above this suggests frication
FRICATION_ZCR = 3000.0  # zero crossings per second suggesting noise
BURST_RISE = 15.0  # decibel jump marking a release burst
BURST_MAX = 0.030  # a burst longer than 30 ms is something else
MIN_REGION = 0.030  # regions shorter than this merge into their neighbours
BURST_WINDOW = 0.002  # the waveform window used to place a release transient
BURST_JUMP = 5.0  # a transient must stand this many times above the closure
SMOOTH = 2  # frames either side used to vote down one-frame flickers

SILENCE = "silence"
VOICED = "voiced"
FRICATION = "frication"
APERIODIC = "aperiodic"
BURST = "burst"

PLAIN = {
    SILENCE: "near-silence",
    VOICED: "periodic (voiced)",
    FRICATION: "noisy, high-frequency (frication)",
    APERIODIC: "aperiodic, low-frequency",
    BURST: "a short, abrupt burst",
}

# Short nouns, for counting regions in a spoken summary.
SHORT = {
    SILENCE: "silent",
    VOICED: "voiced",
    FRICATION: "frication",
    APERIODIC: "aperiodic",
    BURST: "burst",
}


class Region:
    def __init__(self, kind, start, end):
        self.kind = kind
        self.start = start
        self.end = end
        self.stats = {}

    @property
    def duration(self):
        return self.end - self.start

    def __repr__(self):
        return f"<{self.kind} {self.start:.3f}-{self.end:.3f}>"


def classify(frames):
    """Label every frame, then merge them into regions."""
    intensity = frames.intensity
    usable = intensity[~np.isnan(intensity)]
    if usable.size == 0:
        return []
    peak = float(np.max(usable))
    floor = peak - SILENCE_DROP

    labels = []
    for i in range(len(frames.times)):
        db = intensity[i]
        if np.isnan(db) or db < floor:
            labels.append(SILENCE)
            continue
        f0 = frames.f0[i]
        hnr = frames.hnr[i]
        voiced = not np.isnan(f0) and (np.isnan(hnr) or hnr > VOICING_HNR)
        if voiced:
            labels.append(VOICED)
        elif frames.cog[i] > FRICATION_COG or frames.zcr[i] > FRICATION_ZCR:
            labels.append(FRICATION)
        else:
            labels.append(APERIODIC)

    labels = _smooth(labels)
    regions = _merge_runs(labels, frames.times, frames.window)
    # Bursts are found before short regions are absorbed: a release burst is
    # shorter than the minimum region size, and absorbing it would destroy the
    # one event that marks a stop consonant.
    _mark_bursts(regions, frames)
    regions = _absorb_short(regions)
    regions = _merge_adjacent(regions)
    for r in regions:
        r.stats = measure(r, frames)
    return regions


def _smooth(labels):
    """Let each frame's neighbours outvote it.

    Voicing and frication decisions flicker frame to frame at the edges of a
    sound, which otherwise splinters one segment into several. Silence is left
    alone on both sides: a release burst is only a frame or two long, and
    smoothing it away would lose the very thing worth finding.
    """
    out = list(labels)
    for i, label in enumerate(labels):
        if label == SILENCE:
            continue
        lo = max(0, i - SMOOTH)
        window = [l for l in labels[lo : i + SMOOTH + 1] if l != SILENCE]
        if not window:
            continue
        winner = max(set(window), key=window.count)
        if window.count(winner) > len(window) / 2:
            out[i] = winner
    return out


def _merge_runs(labels, times, window):
    """Runs of identical labels become regions, spanning the whole window."""
    w0, w1 = window
    regions = []
    start_i = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_i]:
            start = w0 if start_i == 0 else times[start_i]
            end = w1 if i == len(labels) else times[i]
            regions.append(Region(labels[start_i], start, end))
            start_i = i
    return regions


def _absorb_short(regions):
    """Fold sub-20 ms flickers into whichever neighbour is longer."""
    changed = True
    while changed and len(regions) > 1:
        changed = False
        for i, r in enumerate(regions):
            if r.duration >= MIN_REGION or len(regions) == 1 or r.kind == BURST:
                continue
            before = regions[i - 1] if i > 0 else None
            after = regions[i + 1] if i < len(regions) - 1 else None
            target = before
            if before is None or (after is not None and after.duration > before.duration):
                target = after
            if target is before:
                target.end = r.end
            else:
                target.start = r.start
            regions.pop(i)
            changed = True
            break
    return regions


def _merge_adjacent(regions):
    """Absorbing a short region can leave two of the same kind side by side."""
    out = [regions[0]] if regions else []
    for region in regions[1:]:
        if region.kind == out[-1].kind:
            out[-1].end = region.end
        else:
            out.append(region)
    return out


def _mark_bursts(regions, frames):
    """A short, loud region right after silence is a release burst."""
    for i, r in enumerate(regions):
        if r.kind in (SILENCE, VOICED) or r.duration > BURST_MAX:
            continue
        if i == 0 or regions[i - 1].kind != SILENCE:
            continue
        a, b = frames.at(r.start), frames.at(min(r.end, frames.times[-1]))
        window = frames.intensity[a : b + 1]
        window = window[~np.isnan(window)]
        before = frames.intensity[max(0, a - 2) : a + 1]
        before = before[~np.isnan(before)]
        if window.size and before.size and window.max() - before.min() >= BURST_RISE:
            r.kind = BURST


def find_burst(sound, start, end):
    """The release transient between start and end, or None if there is none.

    Region boundaries come from Praat's intensity contour, whose window is
    tens of milliseconds wide, so a boundary can sit up to half a window
    before the event it marks - at a 75 hertz pitch floor, up to 21 ms early.
    A release is a step change in the waveform itself, so the samples place it
    far more precisely than the contour can. Returns None when nothing stands
    out above the quiet part, which is the honest answer for a stretch of
    frication with no burst in it.
    """
    sr = sound.sampling_frequency
    values = sound.values[0] if sound.values.ndim > 1 else sound.values
    i0 = max(0, int(start * sr))
    i1 = min(len(values), int(end * sr))
    step = max(1, int(BURST_WINDOW * sr))
    levels = [
        (i, float(np.max(np.abs(values[i : i + step]))))
        for i in range(i0, i1 - step + 1, step)
    ]
    if len(levels) < 3:
        return None
    amps = sorted(level for _, level in levels)
    floor = float(np.median(amps[:3]))  # the quietest part: the closure
    peak = amps[-1]
    if floor <= 0 or peak < floor * BURST_JUMP:
        return None
    threshold = floor + 0.10 * (peak - floor)
    for n, (i, level) in enumerate(levels):
        if level >= threshold:
            # The rise begins in the window before the one that crosses.
            if n and levels[n - 1][1] > floor * 2:
                n -= 1
            return levels[n][0] / sr
    return None


def measure(region, frames):
    """Everything measurable about one region. Facts only."""
    a = frames.at(region.start)
    b = max(a, frames.at(max(region.start, region.end - 1e-6)))
    sl = slice(a, b + 1)
    s = {}

    s["intensity_mean"] = _mean(frames.intensity[sl])
    s["intensity_max"] = _max(frames.intensity[sl])
    s["f0"] = _mean(frames.f0[sl])
    s["f0_min"] = _min(frames.f0[sl])
    s["f0_max"] = _max(frames.f0[sl])
    s["f0_slope"] = _slope(frames.times[sl], frames.f0[sl])
    s["hnr"] = _mean(frames.hnr[sl])
    s["cog"] = _mean(frames.cog[sl])
    s["zcr"] = _mean(frames.zcr[sl])
    s["voiced_fraction"] = float(np.mean(frames.voiced[sl])) if b >= a else 0.0

    for k in range(3):
        s[f"f{k + 1}"] = _mean(frames.formants[sl, k])
        s[f"f{k + 1}_spread"] = _spread(frames.formants[sl, k])

    if frames.bands.size:
        s["bands"] = [_mean(frames.bands[sl, i]) for i in range(frames.bands.shape[1])]
    else:
        s["bands"] = []

    _wave_shape(region, frames, s)
    _periodicity(region, frames, s)
    return s


def _wave_shape(region, frames, s):
    """Peak amplitude and the envelope's attack, steady state and decay."""
    sound = frames.sound
    sr = sound.sampling_frequency
    values = sound.values[0] if sound.values.ndim > 1 else sound.values
    i0 = max(0, int(region.start * sr))
    i1 = min(len(values), int(region.end * sr))
    chunk = values[i0:i1]
    if chunk.size == 0:
        return
    s["peak_amp"] = float(np.max(np.abs(chunk)))
    s["peak_time"] = region.start + float(np.argmax(np.abs(chunk))) / sr
    s["rms"] = float(np.sqrt(np.mean(chunk**2)))
    s["clipped"] = bool(np.mean(np.abs(chunk) > 0.99) > 0.001)
    s["asymmetry"] = float(
        (np.max(chunk) + np.min(chunk)) / max(np.max(np.abs(chunk)), 1e-9)
    )

    a, b = frames.at(region.start), frames.at(max(region.start, region.end - 1e-6))
    env = frames.intensity[a : b + 1]
    times = frames.times[a : b + 1]
    env = np.where(np.isnan(env), -np.inf, env)
    if env.size < 3 or not np.isfinite(env).any():
        return
    top = float(np.max(env[np.isfinite(env)]))
    within = np.where(env >= top - 3.0)[0]
    if within.size:
        s["attack"] = float(times[within[0]] - region.start)
        s["steady"] = float(times[within[-1]] - times[within[0]])
        s["decay"] = float(region.end - times[within[-1]])


def _periodicity(region, frames, s):
    """Cycle count, period and jitter, straight from Praat's point process."""
    if region.kind != VOICED or region.duration < 0.05:
        return
    f0 = s.get("f0")
    if f0:
        s["period"] = 1.0 / f0
        s["cycles"] = region.duration / s["period"]
    try:
        part = frames.sound.extract_part(
            from_time=region.start, to_time=region.end, preserve_times=False
        )
        from .analysis import SETTINGS

        points = call(
            part,
            "To PointProcess (periodic, cc)",
            SETTINGS.pitch_floor,
            SETTINGS.pitch_ceiling,
        )
        n = int(call(points, "Get number of points"))
        if n > 2:
            s["cycles"] = float(n - 1)
            jitter = call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            if jitter is not None and not math.isnan(jitter):
                s["jitter"] = float(jitter)
    except Exception:
        pass


def _mean(a):
    a = a[~np.isnan(a)]
    return float(np.mean(a)) if a.size else None


def _max(a):
    a = a[~np.isnan(a)]
    return float(np.max(a)) if a.size else None


def _min(a):
    a = a[~np.isnan(a)]
    return float(np.min(a)) if a.size else None


def _spread(a):
    a = a[~np.isnan(a)]
    return float(np.std(a)) if a.size > 1 else None


def _slope(times, values):
    """Hertz per second, for describing a rise or fall."""
    ok = ~np.isnan(values)
    if ok.sum() < 3:
        return None
    return float(np.polyfit(times[ok], values[ok], 1)[0])
