"""Praat analyses resampled onto one common frame grid.

Everything downstream (events, description, reports, CSV) reads Frames, so a
measurement is computed once and described many ways.
"""

import math

import numpy as np
from parselmouth.praat import call

TIME_STEP = 0.01  # 10 ms frames
WINDOW = 0.025  # 25 ms analysis window for zero crossings


class Settings:
    """Analysis parameters a phonetics student is expected to adjust."""

    def __init__(self):
        self.pitch_floor = 75.0  # hertz; 75 suits a low male voice, 100+ higher
        self.pitch_ceiling = 600.0
        self.formant_ceiling = 5500.0  # 5500 for a female voice, 5000 for male
        self.formant_count = 5.0

    def describe(self):
        return (
            f"pitch floor {self.pitch_floor:.0f} hertz, "
            f"pitch ceiling {self.pitch_ceiling:.0f} hertz, "
            f"formant ceiling {self.formant_ceiling:.0f} hertz, "
            f"{self.formant_count:.0f} formants"
        )


SETTINGS = Settings()


def band_edges(nyquist):
    """Frequency bands, clipped to what the recording can actually carry.

    Class recordings here are 11025 hertz, so there is no information above
    5512 hertz and a fixed 5-to-8 kilohertz band would report silence.
    """
    edges = [
        (0, 300, "the voicing bar below 300 hertz"),
        (300, 800, "the first formant region, 300 to 800 hertz"),
        (800, 2500, "the second formant region, 800 to 2500 hertz"),
        (2500, 5000, "the high formant region, 2.5 to 5 kilohertz"),
        (5000, 8000, "the frication region above 5 kilohertz"),
    ]
    out = []
    for lo, hi, name in edges:
        if lo >= nyquist:
            break
        out.append((lo, min(hi, nyquist), name))
    return out


class Frames:
    """Frame-aligned measurements for one stretch of sound."""

    def __init__(self, sound, offset=0.0):
        self.sound = sound
        self.offset = offset  # where this sits in the original recording
        self.notes = []  # things that could not be measured, in plain words
        self.window = (0.0, sound.duration)  # the described stretch, in sound's clock
        self.times = np.array([])
        self.intensity = np.array([])
        self.f0 = np.array([])
        self.f0_raw = np.array([])  # before octave-error frames are set aside
        self.hnr = np.array([])
        self.formants = np.zeros((0, 3))
        self.bands = np.zeros((0, 0))
        self.band_names = []
        self.band_ranges = []
        self.cog = np.array([])
        self.zcr = np.array([])

    @property
    def voiced(self):
        return ~np.isnan(self.f0)

    @property
    def span(self):
        """Duration of the described stretch, not of the padded analysis sound."""
        return self.window[1] - self.window[0]

    def window_samples(self):
        sound = self.sound
        values = sound.values[0] if sound.values.ndim > 1 else sound.values
        sr = sound.sampling_frequency
        return values[int(self.window[0] * sr) : int(self.window[1] * sr)]

    def at(self, t):
        """Index of the frame nearest a time (in this excerpt's own clock)."""
        return int(np.argmin(np.abs(self.times - t)))

    def absolute(self, t):
        return t + self.offset


def _safe(fn, frames, what):
    try:
        return fn()
    except Exception:
        frames.notes.append(f"{what} could not be measured on this stretch of sound")
        return None


def analyze_window(sound, start, end, settings=None, pad=None):
    """Analyse start..end, but let Praat see the sound either side of it.

    Praat needs several pitch periods to measure anything, so a 60 millisecond
    vowel analysed on its own comes back as unmeasurable. Praat's own editor
    reads the surrounding signal for exactly this reason; padding here and then
    keeping only the frames inside the selection gives the same answer the
    editor would.
    """
    s = settings or SETTINGS
    pad = pad if pad is not None else max(0.1, 8.0 / s.pitch_floor)
    lo = max(0.0, start - pad)
    hi = min(sound.duration, end + pad)
    if lo <= 0 and hi >= sound.duration:
        part, lo = sound, 0.0
    else:
        part = sound.extract_part(from_time=lo, to_time=hi, preserve_times=False)
    return analyze(part, offset=lo, settings=s, window=(start - lo, end - lo))


def analyze(sound, offset=0.0, settings=None, window=None):
    s = settings or SETTINGS
    f = Frames(sound, offset)
    dur = sound.duration
    nyquist = sound.sampling_frequency / 2

    w0, w1 = window if window else (0.0, dur)
    f.window = (w0, w1)
    span = max(w1 - w0, 1e-6)
    n = max(1, int(span / TIME_STEP))
    f.times = w0 + (np.arange(n) + 0.5) * (span / n)

    values = sound.values[0] if sound.values.ndim > 1 else sound.values
    sr = sound.sampling_frequency

    # Pitch and harmonicity both need several periods of signal to work with.
    min_for_pitch = 6.4 / s.pitch_floor
    pitch = harm = None
    if dur >= min_for_pitch:
        pitch = _safe(
            lambda: sound.to_pitch(
                time_step=TIME_STEP,
                pitch_floor=s.pitch_floor,
                pitch_ceiling=s.pitch_ceiling,
            ),
            f,
            "pitch",
        )
        harm = _safe(
            lambda: sound.to_harmonicity_cc(
                time_step=TIME_STEP, minimum_pitch=s.pitch_floor
            ),
            f,
            "harmonicity",
        )
    else:
        f.notes.append(
            f"this stretch is {dur * 1000:.0f} milliseconds long, shorter than the "
            f"{min_for_pitch * 1000:.0f} milliseconds Praat needs to measure pitch "
            f"with a floor of {s.pitch_floor:.0f} hertz"
        )

    intensity = _safe(
        lambda: sound.to_intensity(minimum_pitch=s.pitch_floor, time_step=TIME_STEP),
        f,
        "intensity",
    )
    formant = _safe(
        lambda: sound.to_formant_burg(
            time_step=TIME_STEP,
            max_number_of_formants=s.formant_count,
            maximum_formant=min(s.formant_ceiling, nyquist),
            window_length=WINDOW,
        ),
        f,
        "formants",
    )
    spec = _safe(
        lambda: sound.to_spectrogram(
            window_length=0.005, maximum_frequency=nyquist, time_step=0.002
        ),
        f,
        "the spectrogram",
    )

    f.f0 = np.full(n, np.nan)
    f.hnr = np.full(n, np.nan)
    f.intensity = np.full(n, np.nan)
    f.formants = np.full((n, 3), np.nan)

    for i, t in enumerate(f.times):
        if pitch is not None:
            v = pitch.get_value_at_time(t)
            if v is not None and not math.isnan(v):
                f.f0[i] = v
        if harm is not None:
            v = harm.get_value(time=t)
            if v is not None and not math.isnan(v) and v > -100:
                f.hnr[i] = v
        if intensity is not None:
            v = intensity.get_value(t)
            if v is not None and not math.isnan(v):
                f.intensity[i] = v
        if formant is not None:
            for k in range(3):
                v = formant.get_value_at_time(k + 1, t)
                if v is not None and not math.isnan(v):
                    f.formants[i, k] = v

    if intensity is None:
        f.intensity = _rms_db(values, sr, f.times)

    f.f0_raw = f.f0.copy()
    _drop_octave_errors(f)

    f.zcr = _zero_crossing_rate(values, sr, f.times)
    f.band_ranges = band_edges(nyquist)
    f.band_names = [name for _, _, name in f.band_ranges]

    if spec is not None:
        f.bands, f.cog = _bands_and_cog(spec, f.times, f.band_ranges)
    else:
        f.bands = np.zeros((n, len(f.band_ranges)))
        f.cog = np.full(n, np.nan)

    return f


def _drop_octave_errors(frames, factor=1.8):
    """Discard pitch frames that jump nearly an octave off the speaker's median.

    Pitch trackers slip to double or half the true frequency on creaky or quiet
    frames. Left in, a handful of those frames drag the mean and the reported
    range far away from the voice. The raw values stay in f0_raw and in the CSV,
    so nothing is hidden.
    """
    voiced = frames.f0[~np.isnan(frames.f0)]
    if voiced.size < 4:
        return
    median = float(np.median(voiced))
    bad = (frames.f0 > median * factor) | (frames.f0 < median / factor)
    count = int(np.count_nonzero(bad & ~np.isnan(frames.f0)))
    if count:
        frames.f0[bad] = np.nan
        frames.notes.append(
            f"{count} pitch frame{'s' if count != 1 else ''} fell more than "
            f"{factor} times away from the median of {median:.0f} hertz and were "
            "set aside as octave errors; they remain in the pitch CSV as f0_raw_hz"
        )


def _rms_db(values, sr, times):
    """Fallback loudness when the excerpt is too short for Praat's intensity."""
    half = int(WINDOW * sr / 2)
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        c = int(t * sr)
        chunk = values[max(0, c - half) : c + half]
        if chunk.size:
            rms = float(np.sqrt(np.mean(chunk**2)))
            out[i] = 20 * math.log10(max(rms, 1e-10) / 2e-5)
    return out


def _zero_crossing_rate(values, sr, times):
    """Crossings per second: high for noise, low for a voiced vowel."""
    half = int(WINDOW * sr / 2)
    out = np.zeros(len(times))
    for i, t in enumerate(times):
        c = int(t * sr)
        chunk = values[max(0, c - half) : c + half]
        if chunk.size > 1:
            out[i] = np.count_nonzero(np.diff(np.signbit(chunk))) / (chunk.size / sr)
    return out


def _bands_and_cog(spec, times, ranges):
    power = spec.values  # (frequency, time), power spectral density
    freqs = spec.ys()
    stimes = spec.xs()
    idx = np.clip(np.searchsorted(stimes, times), 0, len(stimes) - 1)

    bands = np.zeros((len(times), len(ranges)))
    cog = np.full(len(times), np.nan)
    for i, col in enumerate(idx):
        column = np.maximum(power[:, col], 0)
        total = column.sum()
        if total > 0:
            cog[i] = float((freqs * column).sum() / total)
        for b, (lo, hi, _) in enumerate(ranges):
            mask = (freqs >= lo) & (freqs < hi)
            bands[i, b] = 10 * math.log10(max(float(column[mask].sum()), 1e-20))
    return bands, cog


def spectral_peaks(sound, time, count=4, settings=None):
    """The strongest frequency components at one instant."""
    s = settings or SETTINGS
    nyquist = sound.sampling_frequency / 2
    half = WINDOW / 2
    start = max(0.0, time - half)
    end = min(sound.duration, time + half)
    part = sound.extract_part(from_time=start, to_time=end, preserve_times=False)
    spectrum = part.to_spectrum()
    n = int(call(spectrum, "Get number of bins"))
    freqs, powers = [], []
    for i in range(1, n + 1):
        freqs.append(call(spectrum, "Get frequency from bin number", i))
        re = call(spectrum, "Get real value in bin", i)
        im = call(spectrum, "Get imaginary value in bin", i)
        powers.append(re * re + im * im)
    freqs = np.array(freqs)
    powers = np.array(powers)

    peaks = []
    for i in range(1, len(powers) - 1):
        if powers[i] > powers[i - 1] and powers[i] >= powers[i + 1]:
            peaks.append((powers[i], freqs[i]))
    peaks.sort(reverse=True)
    top = [f for _, f in peaks[:count]]
    cog = call(spectrum, "Get centre of gravity", 2)
    return sorted(top), cog, nyquist
