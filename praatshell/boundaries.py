"""Where a vowel starts and stops, placed the way a person reads Praat's window.

The 10 millisecond frames that events.py works on are too coarse for a stop
and its vowel: a voice onset time can be 10 milliseconds, and a frame is 10
milliseconds wide. The three boundaries are therefore placed from the signal
itself, each by the cue a person reads off the waveform and spectrogram:

release   The stop's burst: the first sample-level rise out of the quiet
          before it, found in the high-frequency part of the waveform.
onset     Where the vocal folds start: the first pulse of the run of evenly
          spaced pulses that carries on into the vowel, not the irregular
          pulses the tracker finds in aspiration noise.
offset    Where the formants stop: the last moment high-frequency energy is
          still visible in the spectrogram, at Praat's own dynamic range.

The offset depends on what follows. A vowel before a voiced stop ends in an
abrupt step into a closure that keeps a voicing bar, and that step is the end.
A vowel before a voiceless stop devoices, goes silent for the closure, and may
then carry on as breath or release noise. That noise is part of the vowel's
visible high-frequency band, so it is counted as part of the vowel, and the
end of the formants before it is reported alongside as the first offset.

Every threshold lives at the top so it can be retuned. They were set against
Praat measurements of pat, pad, bat and bad from Jacob-Practice.Collection.
"""

import numpy as np
from parselmouth.praat import call

HF_LOW = 2500.0  # hertz: "high-frequency components" are the ones above this
HF_RANGE = 50.0  # decibels below the loudest cell that Praat's display still shows
HF_BRIDGE = 0.004  # a visible gap shorter than this does not end the formants
TAIL_BRIDGE = 0.008  # the same for the ragged noise after a voiceless closure
TAIL_GAP = 0.050  # release noise must begin this soon after the formants stop
VOICED_CLOSURE_DB = -13.0  # a closure this close to the vowel's low band is voiced
COUNT_RELEASE_NOISE = True  # False ends a vowel at its first offset instead
ONSET_LEAD = 0.2 # onset sits this fraction of a period before the first pulse
PULSE_TOLERANCE = 0.2  # a pulse this far off the period is not part of the run
RELEASE_BAND = 1500.0  # hertz: the burst is looked for above this
RELEASE_RISE = 12.0  # decibels above the quiet that count as the release
RELEASE_QUIET = 25.0  # the quiet must sit this far below the loudest part
RELEASE_QUIET_STEPS = 2  # the quiet lasts this many milliseconds beyond its last step
REFINE_RISE = 8.0  # times the quiet's peak that the raw waveform must reach
RELEASE_LOOKBACK = 0.25  # seconds before the onset the release is looked for
LEVEL_STEP = 0.001
LEVEL_WINDOW = 0.002
TAIL_CAP = 0.030  # the offset cannot outrun the tracker-based end by more than this
TRACKER_AGREE = 0.030  # nor fall short of it: the high band is then too weak to trust


class Ends:
    """Where one vowel stops. Times are in the frames' clock."""

    def __init__(self, first, last, closure=None, tail=None):
        self.first = first  # the formants stop: the first offset
        self.last = last  # the last offset, release or breath noise included
        self.closure = closure  # "voiced", "voiceless", or None if unknown
        self.tail = tail  # (start, end) of the noise after the closure, or None


class Bounds:
    """Signal-level measurements of one stretch, made once and shared."""

    def __init__(self, frames):
        from .analysis import SETTINGS

        self.frames = frames
        sound = frames.sound
        self.sr = sound.sampling_frequency
        nyquist = self.sr / 2
        self.values = sound.values[0] if sound.values.ndim > 1 else sound.values

        self.pulses = []
        try:
            points = call(
                sound,
                "To PointProcess (periodic, cc)",
                SETTINGS.pitch_floor,
                SETTINGS.pitch_ceiling,
            )
            n = int(call(points, "Get number of points"))
            self.pulses = [call(points, "Get time from index", i + 1) for i in range(n)]
        except Exception:
            pass

        self.high = self.low = None
        try:
            top = min(nyquist - 50.0, 5500.0)
            self.high = call(
                sound, "Filter (pass Hann band)", RELEASE_BAND, top, 100.0
            ).values[0]
            self.low = call(sound, "Filter (pass Hann band)", 0.0, 600.0, 50.0).values[0]
        except Exception:
            pass

        self.visible = self.cell_times = None
        try:
            emphasised = call(sound, "Filter (pre-emphasis)", 50.0)
            spec = emphasised.to_spectrogram(
                window_length=0.005,
                maximum_frequency=min(5000.0, nyquist),
                time_step=0.002,
                frequency_step=20.0,
            )
            power = spec.values
            freqs, times = spec.ys(), spec.xs()
            db = 10 * np.log10(np.maximum(power, 1e-30))
            inside = (times >= frames.window[0]) & (times <= frames.window[1])
            if inside.any():
                ceiling = db[:, inside].max()
                band = db[freqs >= HF_LOW, :].max(axis=0)
                self.visible = band >= ceiling - HF_RANGE
                self.cell_times = times
        except Exception:
            pass

    # --- the release ---------------------------------------------------

    def _levels(self, signal, start, end):
        """Decibel level of 2 ms windows, stepped every millisecond."""
        n = int(LEVEL_WINDOW * self.sr)
        step = max(1, int(LEVEL_STEP * self.sr))
        i0 = max(0, int(start * self.sr))
        i1 = min(len(signal), int(end * self.sr))
        idx = np.arange(i0, max(i0, i1 - n), step)
        out = np.array(
            [
                20 * np.log10(max(float(np.sqrt(np.mean(signal[i : i + n] ** 2))), 1e-9))
                for i in idx
            ]
        )
        return idx / self.sr, out

    def release(self, onset, earliest):
        """The burst just before onset, or None if nothing rises out of quiet."""
        if self.high is None:
            return None
        start = max(earliest, onset - RELEASE_LOOKBACK)
        times, level = self._levels(self.high, start, onset)
        if level.size < 8:
            return None
        quiet = float(np.percentile(level, 5))
        if level.max() - quiet < RELEASE_QUIET:
            return None
        threshold = quiet + RELEASE_RISE
        i = len(level) - 1
        while i >= 0 and level[i] >= threshold:
            i -= 1
        if i < RELEASE_QUIET_STEPS or i == len(level) - 1:
            return None  # no quiet before the onset, or no rise out of it
        if level[i - RELEASE_QUIET_STEPS : i + 1].max() >= threshold:
            return None  # a dip inside noise, not a quiet a release comes out of
        at = self._refine(float(times[i + 1]))
        return at if at < onset else None

    def _refine(self, coarse):
        """Move a release found in the filtered signal onto the waveform.

        Band-pass filtering smears an abrupt rise a few milliseconds earlier
        than it happens, so the filtered signal only says roughly where to
        look. The raw samples then place it: the first one that stands well
        clear of the quiet that came just before.
        """
        sr = self.sr
        quiet_from = max(0, int((coarse - 0.014) * sr))
        quiet_to = max(quiet_from + 1, int((coarse - 0.004) * sr))
        step = max(1, int(LEVEL_STEP * sr))
        quiet = [
            float(np.max(np.abs(self.values[i : i + step])))
            for i in range(quiet_from, quiet_to, step)
        ]
        if not quiet:
            return coarse
        limit = REFINE_RISE * float(np.median(quiet))
        i0 = max(0, int((coarse - 0.004) * sr))
        i1 = min(len(self.values), int((coarse + 0.008) * sr))
        above = np.flatnonzero(np.abs(self.values[i0:i1]) >= limit)
        return (i0 + int(above[0])) / sr if above.size else coarse

    # --- the onset -----------------------------------------------------

    def onset(self, start, end, earliest):
        """Where steady voicing begins in the voiced region start..end.

        None when there are no pulses to go by. The pulses before start are
        followed back for as long as they keep the vowel's own spacing, which
        is how a vowel the frames called voiced late is caught.
        """
        inside = [p for p in self.pulses if start - 0.002 <= p <= end]
        if len(inside) < 3:
            return None
        period = float(np.median(np.diff(inside)))
        first = self.pulses.index(inside[0])
        while first > 0:
            gap = self.pulses[first] - self.pulses[first - 1]
            if (
                abs(gap - period) > PULSE_TOLERANCE * period
                or self.pulses[first - 1] < earliest
            ):
                break
            first -= 1
        return self.pulses[first] - ONSET_LEAD * period

    # --- the offset ----------------------------------------------------

    def runs(self, bridge=HF_BRIDGE):
        """Stretches where high-frequency energy is visible, as (start, end)."""
        if self.visible is None:
            return []
        out = []
        for t, seen in zip(self.cell_times, self.visible):
            if not seen:
                continue
            if out and t - out[-1][1] <= bridge + 1e-9:
                out[-1][1] = t
            else:
                out.append([t, t])
        return [(a, b) for a, b in out]

    def _low_db(self, start, end):
        seg = self.low[max(0, int(start * self.sr)) : int(end * self.sr)]
        if seg.size == 0:
            return None
        return 20 * np.log10(max(float(np.sqrt(np.mean(seg**2))), 1e-9))

    def ends(self, onset, limit, ceiling):
        """The first and last offsets for a vowel that begins at onset.

        limit is where the next voiced region starts, which nothing here may
        cross. ceiling is the latest the formants may plausibly end, taken from
        the tracker-based end so that a fricative running straight on from the
        vowel cannot drag it out.
        """
        if self.visible is None or self.low is None:
            return None
        run = next((r for r in self.runs() if r[0] <= onset + 0.03 <= r[1]), None)
        if run is None:
            return None
        if run[1] < ceiling - TRACKER_AGREE:
            return None  # a vowel with little energy above HF_LOW: use the tracker
        first = min(run[1], ceiling + TAIL_CAP, limit)

        vowel = self._low_db(onset + 0.03, first - 0.03) if first - onset > 0.08 else None
        gap = self._low_db(first + 0.005, first + 0.030)
        closure = None
        if vowel is not None and gap is not None:
            closure = "voiced" if gap - vowel >= VOICED_CLOSURE_DB else "voiceless"

        tail = None
        if closure == "voiceless":
            for a, b in self.runs(TAIL_BRIDGE):
                if a <= first:
                    continue
                if a - first <= TAIL_GAP and b <= limit:
                    tail = (a, b)
                break
        last = tail[1] if tail else first
        return Ends(first, last, closure, tail)


def bounds(frames):
    """The Bounds for these frames, made on first use."""
    cached = getattr(frames, "_bounds", None)
    if cached is None:
        cached = frames._bounds = Bounds(frames)
    return cached
