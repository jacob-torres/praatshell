"""The spectrogram command: the picture put into words, plus its measurements.

Four things can be asked of it, alone or together:

    summary    what the spectrogram picture shows, region by region
    power      the power spectrum, at one instant or averaged over the selection
    formants   F1 to Fn with bandwidths, for every voiced region
    harmonics  the harmonics of the voice, with H1 minus H2 and friends

Everything is computed from the sound itself with Praat's own spectrogram and
formant tracker (through parselmouth) and numpy's FFT, so the numbers match what
Praat draws. The spoken reply is short; the detail goes to the report and CSVs.
"""

import math
from collections import namedtuple

import numpy as np

from . import analysis, events, fmt
from .describe import _nearest_vowel
from .session import SessionError

Result = namedtuple("Result", "spoken lines tables")

REFERENCE_POWER = (2e-5) ** 2  # Praat's reference pressure, squared
SECTIONS = ("summary", "power", "formants", "harmonics")

OPTIONS_HELP = (
    "summary, power, formants [N] and harmonics [N] choose what to report (with "
    "none of them you get all four); wide or narrow set the window; "
    "window MS, maxfreq HZ, range DB, preemph DB, ceiling HZ, at TIME, f0 HZ, "
    "pitchfloor HZ, pitchceiling HZ, step MS and peaks N adjust the analysis."
)


class Options:
    def __init__(self):
        s = analysis.SETTINGS
        self.sections = []
        self.window = None  # seconds; None means wide (5 ms) or narrow (30 ms)
        self.narrow = False
        self.maxfreq = 5000.0
        self.dynamic = 50.0
        self.preemph = 6.0
        self.formants = int(s.formant_count)
        self.ceiling = s.formant_ceiling
        self.harmonics = 10
        self.step = 0.01
        self.at = None
        self.f0 = None
        self.pitch_floor = s.pitch_floor
        self.pitch_ceiling = s.pitch_ceiling
        self.peaks = 6
        self.given = []  # what the user typed, for the report

    @property
    def spec_window(self):
        return self.window or (0.03 if self.narrow else 0.005)

    @property
    def bandwidth(self):
        """Analysis bandwidth of the Gaussian window, about 1.3 over its length."""
        return 1.3 / self.spec_window

    def wants(self, section):
        return not self.sections or section in self.sections


def _ms(text):
    """A bare number is milliseconds; 30ms and 0.03s carry their own unit."""
    text = text.strip().lower()
    return fmt.parse_time(text) if text[-1].isalpha() else float(text) / 1000.0


def _number(token, what):
    try:
        return float(token)
    except (TypeError, ValueError):
        raise SessionError(f"Say a number after {what}.")


def parse(arg):
    tokens = arg.lower().split()
    o = Options()
    i = 0

    def take(what):
        nonlocal i
        i += 1
        if i >= len(tokens):
            raise SessionError(f"Say a value after {what}.")
        return tokens[i]

    def optional_count(default):
        nonlocal i
        if i + 1 < len(tokens) and tokens[i + 1].replace(".", "", 1).isdigit():
            i += 1
            return int(float(tokens[i]))
        return default

    while i < len(tokens):
        t = tokens[i]
        if t in ("summary", "power"):
            o.sections.append(t)
        elif t == "formants":
            o.sections.append(t)
            o.formants = optional_count(o.formants)
        elif t == "harmonics":
            o.sections.append(t)
            o.harmonics = optional_count(o.harmonics)
        elif t == "wide":
            o.narrow, o.window = False, None
        elif t == "narrow":
            o.narrow, o.window = True, None
        elif t == "window":
            o.window = _ms(take(t))
        elif t == "step":
            o.step = _ms(take(t))
        elif t == "at":
            o.at = fmt.parse_time(take(t))
        elif t == "maxfreq":
            o.maxfreq = _number(take(t), t)
        elif t == "range":
            o.dynamic = _number(take(t), t)
        elif t == "preemph":
            o.preemph = _number(take(t), t)
        elif t == "ceiling":
            o.ceiling = _number(take(t), t)
        elif t == "f0":
            o.f0 = _number(take(t), t)
        elif t == "pitchfloor":
            o.pitch_floor = _number(take(t), t)
        elif t == "pitchceiling":
            o.pitch_ceiling = _number(take(t), t)
        elif t == "peaks":
            o.peaks = int(_number(take(t), t))
        else:
            raise SessionError(f"spectrogram has no option called {t}. " + OPTIONS_HELP)
        i += 1

    if not 1 <= o.formants <= 10:
        raise SessionError("Ask for between 1 and 10 formants.")
    if not 1 <= o.harmonics <= 60:
        raise SessionError("Ask for between 1 and 60 harmonics.")
    if not 0.001 <= o.spec_window <= 0.2:
        raise SessionError("The window must be between 1 and 200 milliseconds.")
    if o.step <= 0.001:
        raise SessionError("The step must be longer than 1 millisecond.")
    if o.dynamic <= 0 or o.maxfreq <= 0 or o.ceiling <= 0:
        raise SessionError("The range, maxfreq and ceiling must be above zero.")
    if o.pitch_floor >= o.pitch_ceiling:
        raise SessionError("The pitch floor must be below the pitch ceiling.")
    o.given = tokens
    return o


# --- small helpers ----------------------------------------------------------


def _db(power):
    return 10 * math.log10(max(float(power), 1e-30) / REFERENCE_POWER)


def _hz(x):
    return "not measurable" if x is None or np.isnan(x) else f"{x:.0f} hertz"


def _val(x, places=1):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{places}f}"


def _find_peaks(freqs, level, count, min_sep, floor=-np.inf):
    """Strongest local maxima at least min_sep apart, returned in frequency order."""
    idx = [
        i
        for i in range(1, len(level) - 1)
        if level[i] > level[i - 1] and level[i] >= level[i + 1] and level[i] >= floor
    ]
    idx.sort(key=lambda i: -level[i])
    chosen = []
    for i in idx:
        if all(abs(freqs[i] - freqs[j]) >= min_sep for j in chosen):
            chosen.append(i)
        if len(chosen) == count:
            break
    return sorted(chosen)


def _parabola(logamp, i):
    """Peak position (in bins) and height of a three-point parabola."""
    a, b, c = logamp[i - 1], logamp[i], logamp[i + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(i), b
    delta = 0.5 * (a - c) / denom
    return i + delta, b - 0.25 * (a - c) * delta


def _direction(change, ref):
    if abs(change) < max(40.0, 0.04 * ref):
        return "level"
    return "rising" if change > 0 else "falling"


def _resonance_db(f, F, B):
    """How much one formant lifts the spectrum at frequency f, in decibels."""
    b2 = (B / 2.0) ** 2
    num = (F * F + b2) ** 2
    den = ((f - F) ** 2 + b2) * ((f + F) ** 2 + b2)
    return 10 * math.log10(num / den)


# --- the analysis context ---------------------------------------------------


class _Context:
    """The selection plus padding either side, because Praat needs room."""

    def __init__(self, sound, start, end, o):
        self.nyquist = sound.sampling_frequency / 2
        self.sr = sound.sampling_frequency
        self.start, self.end = start, end
        pad = max(0.1, 8.0 / o.pitch_floor, 2 * o.spec_window)
        self.lo = max(0.0, start - pad)
        hi = min(sound.duration, end + pad)
        if self.lo <= 0 and hi >= sound.duration:
            self.part = sound
        else:
            self.part = sound.extract_part(
                from_time=self.lo, to_time=hi, preserve_times=False
            )
        values = self.part.values
        self.samples = values[0] if values.ndim > 1 else values
        self.notes = []
        self.pitch = self._pitch(o)
        self.formant = self._formant(o)

    def local(self, t):
        return t - self.lo

    def _pitch(self, o):
        if self.part.duration < 6.4 / o.pitch_floor:
            return None
        try:
            return self.part.to_pitch(
                time_step=0.01, pitch_floor=o.pitch_floor, pitch_ceiling=o.pitch_ceiling
            )
        except Exception:
            self.notes.append("pitch could not be measured on this stretch")
            return None

    def _formant(self, o):
        try:
            return self.part.to_formant_burg(
                time_step=min(o.step, 0.01),
                max_number_of_formants=float(o.formants),
                maximum_formant=min(o.ceiling, self.nyquist),
                window_length=analysis.WINDOW,
            )
        except Exception:
            self.notes.append("formants could not be measured on this stretch")
            return None

    def f0_over(self, t0, t1):
        """Median pitch over a stretch (absolute seconds), or nan."""
        if self.pitch is None:
            return float("nan")
        ts = np.linspace(t0, t1, 7) if t1 > t0 else [t0]
        vals = []
        for t in ts:
            v = self.pitch.get_value_at_time(self.local(t))
            if v is not None and not math.isnan(v):
                vals.append(v)
        return float(np.median(vals)) if vals else float("nan")

    def formant_at(self, k, t):
        """(frequency, bandwidth) of formant k at an absolute time, or (nan, nan)."""
        if self.formant is None:
            return float("nan"), float("nan")
        lt = self.local(t)
        f = self.formant.get_value_at_time(k, lt)
        b = self.formant.get_bandwidth_at_time(k, lt)
        f = float("nan") if f is None else f
        b = float("nan") if b is None else b
        return f, b


# --- the spectrogram picture ------------------------------------------------


def _picture(ctx, o):
    w = o.spec_window
    spec = ctx.part.to_spectrogram(
        window_length=w,
        maximum_frequency=min(o.maxfreq, ctx.nyquist),
        time_step=0.002 if w < 0.01 else 0.005,
        frequency_step=20.0 if w < 0.01 else 10.0,
    )
    freqs = np.asarray(spec.ys())
    times = np.asarray(spec.xs()) + ctx.lo
    db = 10 * np.log10(np.maximum(np.asarray(spec.values), 1e-30) / 4e-10)
    # Praat's picture boosts high frequencies by default (6 decibels per octave).
    db = db + o.preemph * np.log2(np.maximum(freqs, 1.0) / 1000.0)[:, None]
    keep = (times >= ctx.start) & (times <= ctx.end)
    if not keep.any():
        keep[int(np.argmin(np.abs(times - (ctx.start + ctx.end) / 2)))] = True
    return freqs, times[keep], db[:, keep]


def _region_rows(ctx, o, frames, regions, freqs, times, db):
    """One measured row per region of the timeline, against the picture's scale."""
    top = float(db.max())
    floor = top - o.dynamic
    sep = max(o.bandwidth, 100.0)
    rows = []
    for number, r in enumerate(regions, 1):
        rs, re_ = frames.absolute(r.start), frames.absolute(r.end)
        mask = (times >= rs) & (times <= re_)
        if not mask.any():
            continue
        cells = db[:, mask]
        mean = 10 * np.log10(np.mean(10 ** (cells / 10), axis=1))
        peak_i = int(np.argmax(mean))
        visible_f = np.where(mean >= floor)[0]
        # A silent region has no bands worth naming, only the noise floor.
        ridge_i = [] if r.kind == events.SILENCE else _find_peaks(
            freqs, mean, o.peaks, sep, floor=float(mean.max()) - o.dynamic / 2
        )
        bar = (freqs > 0) & (freqs < 300)
        rows.append(
            {
                "number": number,
                "kind": r.kind,
                "start": rs,
                "end": re_,
                "peak_db": float(mean[peak_i]),
                "peak_hz": float(freqs[peak_i]),
                "upper_hz": float(freqs[visible_f[-1]]) if visible_f.size else None,
                "visible": float(np.mean(cells >= floor)),
                "ridges": [float(freqs[i]) for i in ridge_i],
                "bar": bool(bar.any() and mean[bar].max() >= floor),
                "cog": r.stats.get("cog"),
            }
        )
    return rows


def _summary(ctx, o, name, frames, regions):
    freqs, times, db = _picture(ctx, o)
    top_f, top_t = np.unravel_index(int(np.argmax(db)), db.shape)
    top = float(db.max())
    visible = float(np.mean(db >= top - o.dynamic))
    rows = _region_rows(ctx, o, frames, regions, freqs, times, db)

    kind = "narrowband" if o.spec_window >= 0.02 else "wideband"
    if kind == "wideband":
        reads = (
            "Formants show as dark horizontal bands, and each pulse of the voice "
            "as a thin vertical stripe. Individual harmonics are blurred together."
        )
    else:
        reads = (
            "Harmonics show as thin horizontal lines, evenly spaced by the pitch. "
            "Formants show only as the stretch of lines that are darker."
        )
    f0s = [v for v in frames.f0 if not np.isnan(v)]
    f0 = float(np.mean(f0s)) if f0s else None

    spoken = [
        f"Spectrogram of {name}, {ctx.start:.3f} to {ctx.end:.3f} seconds, "
        f"{kind}: a {o.spec_window * 1000:.0f} millisecond window, which gives a "
        f"frequency resolution of about {o.bandwidth:.0f} hertz, shown up to "
        f"{o.maxfreq:.0f} hertz.",
        f"The darkest point is at {fmt.secs(times[top_t])}, {freqs[top_f]:.0f} "
        f"hertz. {fmt.pct(visible)} of the picture is dark enough to see in a "
        f"{o.dynamic:.0f} decibel dynamic range.",
    ]
    lines = [
        "SPECTROGRAM SUMMARY",
        "",
        f"Type: {kind}. Window {o.spec_window * 1000:.1f} milliseconds, a Gaussian "
        f"shape, so the analysis bandwidth is about {o.bandwidth:.0f} hertz.",
        f"Frequency axis: 0 to {o.maxfreq:.0f} hertz. Dynamic range "
        f"{o.dynamic:.0f} decibels below the darkest point. Pre-emphasis "
        f"{o.preemph:g} decibels per octave above 1000 hertz, which is how Praat "
        "draws the picture by default; set preemph 0 for the raw levels.",
        f"Darkest point: {top:.1f} decibels at {fmt.secs(times[top_t])} and "
        f"{freqs[top_f]:.0f} hertz.",
        f"{fmt.pct(visible)} of the picture is inside the dynamic range; the rest "
        "would be drawn white.",
        reads,
    ]
    if f0:
        if o.bandwidth < 0.8 * f0:
            seen = "so the harmonics of the voice are separated in this picture"
        else:
            seen = "so the harmonics blur together; try narrow to separate them"
        line = (
            f"Mean pitch {f0:.0f} hertz against a resolution of "
            f"{o.bandwidth:.0f} hertz, {seen}."
        )
        lines.append(line)
        spoken.append(line)
    lines.append("")

    for row in rows:
        what = events.PLAIN[row["kind"]]
        head = (
            f"Region {row['number']}, {fmt.stamp(row['start'])} to "
            f"{fmt.stamp(row['end'])}, {what}:"
        )
        body = [f"  Darkest at {row['peak_hz']:.0f} hertz, {row['peak_db']:.1f} decibels."]
        if row["visible"] < 0.005:
            body = ["  Nothing in this region is dark enough to show."]
        else:
            if row["ridges"]:
                body.append(
                    "  Dark bands near " + fmt.join(f"{x:.0f}" for x in row["ridges"])
                    + " hertz."
                )
            if row["upper_hz"] is not None:
                body.append(f"  Energy shows up to {row['upper_hz']:.0f} hertz.")
            if row["kind"] == events.VOICED:
                body.append(
                    "  A voicing bar is visible below 300 hertz."
                    if row["bar"]
                    else "  No voicing bar is visible below 300 hertz."
                )
            if row["cog"]:
                body.append(f"  Centre of gravity {_hz(row['cog'])}.")
            body.append(f"  {fmt.pct(row['visible'])} of this region is visible.")
        lines += [head] + body + [""]
        if row["visible"] >= 0.005:
            said = f"Region {row['number']}, {events.SHORT[row['kind']]}: "
            if row["ridges"] and row["kind"] != events.BURST:
                said += "bands near " + fmt.join(f"{x:.0f}" for x in row["ridges"][:4]) + " hertz"
            else:
                said += f"darkest at {row['peak_hz']:.0f} hertz"
            if row["upper_hz"] is not None:
                said += f", energy up to {row['upper_hz']:.0f} hertz"
            spoken.append(said + ".")
        else:
            spoken.append(f"Region {row['number']}, {events.SHORT[row['kind']]}: too faint to show.")

    ridge_cols = [f"ridge{i}_hz" for i in range(1, o.peaks + 1)]
    columns = [
        "region", "kind", "start_ms", "end_ms", "peak_db", "peak_hz",
        "upper_edge_hz", "visible_pct", "voicing_bar",
    ] + ridge_cols
    table = []
    for row in rows:
        ridges = row["ridges"] + [None] * (o.peaks - len(row["ridges"]))
        table.append(
            [
                row["number"], row["kind"], _val(row["start"] * 1000),
                _val(row["end"] * 1000), _val(row["peak_db"]), _val(row["peak_hz"], 0),
                _val(row["upper_hz"], 0), _val(row["visible"] * 100),
                "yes" if row["bar"] else "no",
            ]
            + [_val(x, 0) for x in ridges]
        )
    return spoken, lines, [(None, columns, table)]


# --- the power spectrum -----------------------------------------------------


def _spectrum_of(ctx, centres, length):
    """Mean amplitude-squared spectrum over Hann-windowed frames."""
    n = max(8, int(round(length * ctx.sr)))
    nfft = 1 << int(math.ceil(math.log2(n * 8)))
    w = np.hanning(n)
    total = np.zeros(nfft // 2 + 1)
    for c in centres:
        a = int(round(ctx.local(c) * ctx.sr)) - n // 2
        seg = np.zeros(n)
        lo, hi = max(a, 0), min(a + n, len(ctx.samples))
        if hi > lo:
            seg[lo - a : hi - a] = ctx.samples[lo:hi]
        amp = np.abs(np.fft.rfft(seg * w, nfft)) * 2.0 / np.sum(w)
        total += amp**2
    freqs = np.fft.rfftfreq(nfft, 1.0 / ctx.sr)
    return freqs, total / max(len(centres), 1)


def _power(ctx, o):
    if o.at is not None:
        length = max(o.window or 0.025, 0.005)
        centres = [o.at]
        where = f"at {fmt.secs(o.at)}"
    else:
        length = min(max(o.window or 0.025, 0.005), max(ctx.end - ctx.start, 0.005))
        hop = length / 2
        count = max(1, int((ctx.end - ctx.start - length) / hop) + 1)
        first = ctx.start + length / 2
        centres = [first + k * hop for k in range(count)] if count > 1 else [
            (ctx.start + ctx.end) / 2
        ]
        where = (
            f"averaged over {len(centres)} windows between {ctx.start:.3f} and "
            f"{ctx.end:.3f} seconds"
        )
    freqs, power = _spectrum_of(ctx, centres, length)
    fmax = min(o.maxfreq, ctx.nyquist)
    sel = (freqs >= 20) & (freqs <= fmax)
    f, p = freqs[sel], power[sel]
    df = float(freqs[1] - freqs[0])
    total = float(p.sum())
    if total <= 0:
        return (
            ["The power spectrum is empty: that stretch is digital silence."],
            ["POWER SPECTRUM", "", "That stretch is digital silence."],
            [],
        )
    level = np.array([_db(x) for x in p])

    cog = float((f * p).sum() / total)
    sd = math.sqrt(float(((f - cog) ** 2 * p).sum() / total))
    skew = float(((f - cog) ** 3 * p).sum() / total / sd**3) if sd > 0 else float("nan")
    kurt = float(((f - cog) ** 4 * p).sum() / total / sd**4 - 3) if sd > 0 else float("nan")
    cum = np.cumsum(p) / total
    roll85 = float(f[np.searchsorted(cum, 0.85)])
    roll95 = float(f[min(np.searchsorted(cum, 0.95), len(f) - 1)])

    # Slope: a straight line through the level in 100 hertz steps.
    edges = np.arange(50.0, fmax, 100.0)
    mids, lv = [], []
    for e in edges:
        m = (f >= e) & (f < e + 100.0)
        if m.any():
            mids.append(e + 50.0)
            lv.append(_db(p[m].sum()))
    slope = float(np.polyfit(np.array(mids) / 1000.0, lv, 1)[0]) if len(mids) > 2 else float("nan")

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(p[m].sum())

    top = min(5000.0, fmax)
    low_p, high_p = band(50, 1000), band(1000, top)
    alpha = 10 * math.log10(high_p / low_p) if low_p > 0 and high_p > 0 else float("nan")

    min_sep = max(2.0 * df, 50.0)
    # 30 decibels down is where the Hann window's own side lobes begin, so
    # anything weaker than that near a strong peak is the window, not the sound.
    top_i = _find_peaks(f, level, o.peaks, min_sep, floor=float(level.max()) - 30)
    raw = []
    logamp = np.log(np.sqrt(np.maximum(p, 1e-60)))
    for i in top_i:
        pos, h = _parabola(logamp, i)
        raw.append((float(f[0] + pos * df), 20 * math.log10(max(math.exp(h), 1e-30) / 2e-5)))

    k = max(1, int(round(300.0 / df)))
    smooth = np.convolve(p, np.ones(k) / k, mode="same")
    env_level = np.array([_db(x) for x in smooth])
    env_i = _find_peaks(f, env_level, min(o.peaks, 5), 300.0, floor=float(env_level.max()) - 30)
    env = [(float(f[i]), float(env_level[i])) for i in env_i]

    lines = [
        "POWER SPECTRUM",
        "",
        f"A {length * 1000:.0f} millisecond Hann window, {where}.",
        f"Frequency range {f[0]:.0f} to {fmax:.0f} hertz, one point every "
        f"{df:.1f} hertz (after zero-padding; the true resolution is about "
        f"{1.44 / length:.0f} hertz).",
        "Levels are in decibels relative to 0.00002, the reference Praat uses.",
        "",
        f"Strongest component: {_hz(raw[0][0]) if raw else 'none'}"
        + (f", {raw[0][1]:.1f} decibels." if raw else "."),
        "Strongest narrow peaks (single harmonics, when the voice is resolved):",
    ]
    lines += [f"  {fr:.0f} hertz, {lv_:.1f} decibels." for fr, lv_ in raw]
    lines += ["Broad peaks, smoothed over 300 hertz (these follow the formants):"]
    lines += [f"  {fr:.0f} hertz, {lv_:.1f} decibels." for fr, lv_ in env]
    lines += [
        "",
        f"Centre of gravity {cog:.0f} hertz.",
        f"Standard deviation {sd:.0f} hertz.",
        f"Skewness {skew:.2f}. Positive means a long tail of energy reaching up in "
        "frequency; negative means the tail runs down.",
        f"Excess kurtosis {kurt:.2f}. Higher means a more sharply peaked spectrum.",
        f"85 percent of the energy lies below {roll85:.0f} hertz and 95 percent below "
        f"{roll95:.0f} hertz.",
        f"Spectral slope {slope:.1f} decibels per kilohertz; a steeper fall means a "
        "breathier, softer or more rounded sound.",
        f"Alpha ratio {alpha:.1f} decibels: energy from 1000 to {top:.0f} hertz "
        "against energy from 50 to 1000 hertz. Positive means the sound is bright.",
        "",
        "Energy in each band, in decibels:",
    ]
    for lo, hi, label in analysis.band_edges(ctx.nyquist):
        if lo < fmax:
            lines.append(f"  {label}: {_db(band(lo, min(hi, fmax))):.1f}.")
    if o.at is None and len(centres) > 1:
        lines += [
            "",
            "This is an average over the whole selection. If the selection spans "
            "different sounds, such as a vowel and a fricative, the average mixes "
            "them. Use at TIME for a single instant, or select one region first.",
        ]

    spoken = [
        f"Power spectrum {where}.",
        f"Strongest narrow peaks at {fmt.join(f'{fr:.0f}' for fr, _ in raw[:4])} hertz."
        if raw else "No clear peaks.",
        f"Broad peaks at {fmt.join(f'{fr:.0f}' for fr, _ in env)} hertz." if env else "",
        f"Centre of gravity {cog:.0f} hertz, standard deviation {sd:.0f}, skewness {skew:.1f}, "
        f"slope {slope:.1f} decibels per kilohertz.",
    ]
    spoken = [s for s in spoken if s]

    step = max(1, len(f) // 2000)
    spectrum_table = [(_val(a, 1), _val(b, 2)) for a, b in zip(f[::step], level[::step])]
    peak_table = (
        [("narrow", _val(a, 1), _val(b, 2)) for a, b in raw]
        + [("broad", _val(a, 1), _val(b, 2)) for a, b in env]
    )
    stats = [
        ("centre_of_gravity_hz", _val(cog, 1)), ("std_dev_hz", _val(sd, 1)),
        ("skewness", _val(skew, 3)), ("excess_kurtosis", _val(kurt, 3)),
        ("rolloff_85_hz", _val(roll85, 1)), ("rolloff_95_hz", _val(roll95, 1)),
        ("slope_db_per_khz", _val(slope, 2)), ("alpha_ratio_db", _val(alpha, 2)),
    ]
    tables = [
        ("spec_power", ["frequency_hz", "level_db"], spectrum_table),
        ("spec_peaks", ["kind", "frequency_hz", "level_db"], peak_table),
        ("spec_power_stats", ["measure", "value"], stats),
    ]
    return spoken, lines, tables


# --- formants ---------------------------------------------------------------


def _formants(ctx, o, frames, regions):
    voiced = [(n, r) for n, r in enumerate(regions, 1) if r.kind == events.VOICED]
    head = [
        "FORMANTS",
        "",
        f"Looking for up to {o.formants} formants below {min(o.ceiling, ctx.nyquist):.0f} "
        "hertz, with Praat's Burg method and a 25 millisecond window.",
        "Each is the median over the middle half of the region. Bandwidth is how "
        "wide the resonance is: narrow means a clear peak, wide means a poorly "
        "defined one. The 25, 50 and 75 percent columns show how the formant "
        "moves through the region.",
    ]
    ratio = min(o.ceiling, ctx.nyquist) / o.formants
    if not 900 <= ratio <= 1300:
        head.append(
            f"Note: {o.formants} formants with a ceiling of {o.ceiling:.0f} hertz is "
            f"{ratio:.0f} hertz per formant. A ratio near 1000 to 1100 is the usual "
            "choice, such as 5 formants to 5000 or 5500 hertz. If F1 to F3 look "
            "wrong, try changing one of the two."
        )
    head.append("")
    if ctx.formant is None or not voiced:
        why = "No voiced region was found, so there is nothing to measure." if not voiced else \
            "Formants could not be measured here."
        return [why], head + [why], []

    spoken, lines, rows = [], head[:], []
    for number, r in voiced:
        rs, re_ = frames.absolute(r.start), frames.absolute(r.end)
        span = re_ - rs
        c0, c1 = rs + 0.25 * span, rs + 0.75 * span
        times = np.arange(c0, c1 + 1e-9, 0.005) if span >= 0.04 else [(rs + re_) / 2]
        points = [rs + p * span for p in (0.25, 0.5, 0.75)]
        f0 = ctx.f0_over(c0, c1)
        lines.append(
            f"Voiced region {number}, {fmt.stamp(rs)} to {fmt.stamp(re_)}"
            + (f", pitch {f0:.0f} hertz:" if not np.isnan(f0) else ":")
        )
        found = {}
        said = []
        for k in range(1, o.formants + 1):
            vals = [ctx.formant_at(k, t) for t in times]
            fs = [v[0] for v in vals if not math.isnan(v[0])]
            bs = [v[1] for v in vals if not math.isnan(v[1])]
            if not fs:
                lines.append(f"  F{k}: not found.")
                continue
            med, bw = float(np.median(fs)), float(np.median(bs)) if bs else float("nan")
            marks = [ctx.formant_at(k, t)[0] for t in points]
            change = (
                marks[2] - marks[0] if not (math.isnan(marks[0]) or math.isnan(marks[2])) else float("nan")
            )
            found[k] = med
            way = _direction(change, med) if not math.isnan(change) else "unknown"
            lines.append(
                f"  F{k}: {med:.0f} hertz, bandwidth {bw:.0f} hertz, "
                f"{way}"
                + (f" ({change:+.0f} hertz from 25 to 75 percent)" if not math.isnan(change) else "")
                + f". At 25, 50 and 75 percent: "
                + ", ".join("not found" if math.isnan(m) else f"{m:.0f}" for m in marks)
                + "."
            )
            said.append(f"F{k} {med:.0f}")
            rows.append(
                [
                    number, k, _val(med, 1), _val(bw, 1),
                    _val(marks[0], 1), _val(marks[1], 1), _val(marks[2], 1),
                    _val(change, 1), way, len(fs),
                ]
            )
        if 1 in found and 2 in found:
            best, runner, _ = _nearest_vowel(found[1], found[2])
            lines.append(f"  Closest reference vowel: {best}. Next closest: {runner}.")
            said.append(f"closest to {best}")
        lines.append("")
        spoken.append(f"Region {number}: " + ", ".join(said) + "." if said else
                      f"Region {number}: no formants found.")

    # Frame by frame, every frame in the selection.
    track_cols = (
        ["time_ms", "f0_hz"]
        + [f"f{k}_hz" for k in range(1, o.formants + 1)]
        + [f"b{k}_hz" for k in range(1, o.formants + 1)]
    )
    track = []
    t = ctx.start + o.step / 2
    while t < ctx.end:
        fb = [ctx.formant_at(k, t) for k in range(1, o.formants + 1)]
        f0 = ctx.f0_over(t, t)
        track.append(
            [_val(t * 1000, 1), _val(f0)]
            + [_val(x[0], 0) for x in fb]
            + [_val(x[1], 0) for x in fb]
        )
        t += o.step
    cols = [
        "region", "formant", "median_hz", "bandwidth_hz", "at_25_hz", "at_50_hz",
        "at_75_hz", "change_hz", "direction", "frames",
    ]
    return spoken, lines, [("spec_formants", cols, rows), ("spec_tracks", track_cols, track)]


# --- harmonics --------------------------------------------------------------


def _measure_harmonics(ctx, o, t, f0):
    """Peaks near each multiple of f0, with parabolic refinement."""
    length = min(max(0.025, 3.5 / f0), max(ctx.part.duration - 0.001, 0.01))
    freqs, power = _spectrum_of(ctx, [t], length)
    amp = np.sqrt(np.maximum(power, 1e-60))
    logamp = np.log(amp)
    df = float(freqs[1] - freqs[0])
    out = []
    est = f0
    limit = ctx.nyquist * 0.98
    for k in range(1, o.harmonics + 1):
        nominal = k * est
        if nominal > limit:
            break
        half = 0.4 * est
        lo_i = max(1, int((nominal - half) / df))
        hi_i = min(len(freqs) - 2, int((nominal + half) / df) + 1)
        if hi_i <= lo_i:
            break
        i = lo_i + int(np.argmax(amp[lo_i : hi_i + 1]))
        i = min(max(i, 1), len(freqs) - 2)
        pos, h = _parabola(logamp, i)
        freq = pos * df
        level = 20 * math.log10(max(math.exp(h), 1e-30) / 2e-5)
        out.append({"k": k, "nominal": k * f0, "freq": freq, "level": level})
        est = freq / k
    return out, length


def _harmonics(ctx, o, frames, regions):
    if o.at is not None:
        spots = [(None, o.at)]
    else:
        spots = [
            (n, (frames.absolute(r.start) + frames.absolute(r.end)) / 2)
            for n, r in enumerate(regions, 1)
            if r.kind == events.VOICED
        ]
    head = [
        "HARMONICS",
        "",
        f"The first {o.harmonics} harmonics, found as peaks near whole multiples of "
        "the pitch in a Hann-windowed spectrum. The window is long enough to hold "
        "at least three and a half cycles, so neighbouring harmonics separate.",
        "H1 minus H2 compares the first two harmonics: positive means a breathy "
        "or soft voice, negative means a pressed or creaky one. A1, A2 and A3 are "
        "the harmonics nearest the first three formants. The starred values "
        "(H1* and H2*) have the lift from F1 to F3 taken out, which is what makes "
        "them comparable between vowels.",
        "",
    ]
    if not spots:
        return ["No voiced region was found, so there are no harmonics to measure."], head + [
            "No voiced region was found."
        ], []

    spoken, lines = [], head[:]
    harm_rows, measure_rows = [], []
    for number, t in spots:
        if o.f0:
            f0 = o.f0
        else:
            region = next((r for n, r in enumerate(regions, 1) if n == number), None)
            if region is not None:
                f0 = ctx.f0_over(frames.absolute(region.start), frames.absolute(region.end))
            else:
                f0 = ctx.f0_over(t - 0.02, t + 0.02)
        label = f"region {number}" if number else "the instant asked for"
        if f0 is None or math.isnan(f0):
            lines += [f"At {fmt.stamp(t)} ({label}): no pitch was found, so harmonics "
                      "cannot be located. Give one with f0 HERTZ.", ""]
            spoken.append(f"At {fmt.stamp(t)}: no pitch found. Give one with f0.")
            continue
        found, length = _measure_harmonics(ctx, o, t, f0)
        if len(found) < 2:
            lines += [f"At {fmt.stamp(t)}: fewer than two harmonics fit below the top "
                      "of the recording.", ""]
            continue
        h1, h2 = found[0], found[1]
        strongest = max(found, key=lambda h: h["level"])
        formants = []
        for k in range(1, min(o.formants, 3) + 1):
            fr, bw = ctx.formant_at(k, t)
            if not (math.isnan(fr) or math.isnan(bw)):
                formants.append((k, fr, bw))

        def corrected(h):
            return h["level"] - sum(_resonance_db(h["freq"], fr, bw) for _, fr, bw in formants)

        h1h2 = h1["level"] - h2["level"]
        h1h2c = corrected(h1) - corrected(h2) if formants else float("nan")
        a_vals = {}
        for k, fr, _ in formants:
            near = min(found, key=lambda h: abs(h["freq"] - fr))
            a_vals[k] = near
        slope = (
            float(np.polyfit([h["k"] for h in found], [h["level"] for h in found], 1)[0])
            if len(found) > 2 else float("nan")
        )
        lines.append(
            f"At {fmt.stamp(t)} ({label}), pitch {f0:.1f} hertz, period "
            f"{1000 / f0:.2f} milliseconds, window {length * 1000:.0f} milliseconds:"
        )
        for h in found:
            off = h["freq"] - h["nominal"]
            lines.append(
                f"  H{h['k']}: {h['freq']:.1f} hertz, {h['level']:.1f} decibels, "
                f"{h['level'] - h1['level']:+.1f} against H1"
                + (f", {off:+.1f} hertz from {h['nominal']:.0f}" if abs(off) >= 1 else "")
                + "."
            )
            harm_rows.append(
                [_val(t * 1000, 1), number or "", h["k"], _val(h["nominal"], 1),
                 _val(h["freq"], 1), _val(h["level"], 2), _val(h["level"] - h1["level"], 2),
                 _val(off, 1)]
            )
        lines.append(f"  H1 minus H2: {h1h2:.1f} decibels.")
        if formants:
            lines.append(
                f"  H1* minus H2*: {h1h2c:.1f} decibels (corrected for "
                f"{fmt.join(f'F{k}' for k, _, _ in formants)})."
            )
        rel = {}
        for k, near in a_vals.items():
            rel[k] = h1["level"] - near["level"]
            lines.append(
                f"  H1 minus A{k}: {rel[k]:.1f} decibels, A{k} being H{near['k']} at "
                f"{near['freq']:.0f} hertz, next to F{k}."
            )
        lines.append(
            f"  Strongest harmonic: H{strongest['k']} at {strongest['freq']:.0f} hertz."
        )
        lines.append(
            f"  Levels fall by {-slope:.1f} decibels per harmonic on average."
            if slope < 0 else f"  Levels rise by {slope:.1f} decibels per harmonic on average."
        )
        lines.append("")
        spoken.append(
            f"At {fmt.stamp(t)}: pitch {f0:.0f} hertz, {len(found)} harmonics, strongest is "
            f"H{strongest['k']} at {strongest['freq']:.0f} hertz. H1 minus H2 is "
            f"{h1h2:.1f} decibels"
            + (f", corrected {h1h2c:.1f}" if formants else "")
            + (f". H1 minus A1 is {rel[1]:.1f}" if 1 in rel else "")
            + "."
        )
        measure_rows.append(
            [_val(t * 1000, 1), number or "", _val(f0, 2), len(found),
             _val(h1["level"], 2), _val(h2["level"], 2), _val(h1h2, 2),
             _val(h1h2c, 2)]
            + [_val(rel.get(k), 2) for k in (1, 2, 3)]
            + [strongest["k"], _val(slope, 2)]
        )

    hcols = ["time_ms", "region", "harmonic", "nominal_hz", "frequency_hz",
             "level_db", "re_h1_db", "offset_hz"]
    mcols = ["time_ms", "region", "f0_hz", "harmonics_found", "h1_db", "h2_db",
             "h1_h2_db", "h1c_h2c_db", "h1_a1_db", "h1_a2_db", "h1_a3_db",
             "strongest_harmonic", "slope_db_per_harmonic"]
    return spoken, lines, [("spec_harmonics", hcols, harm_rows),
                           ("spec_harmonic_measures", mcols, measure_rows)]


# --- the whole command ------------------------------------------------------


def run(sound, start, end, frames, regions, name, o):
    nyq = sound.sampling_frequency / 2
    notes = []
    if o.maxfreq > nyq:
        notes.append(f"maxfreq was lowered to {nyq:.0f} hertz, the top of this recording")
        o.maxfreq = nyq
    if o.ceiling > nyq:
        notes.append(f"ceiling was lowered to {nyq:.0f} hertz, the top of this recording")
        o.ceiling = nyq
    if o.at is not None and not start <= o.at <= end:
        raise SessionError(
            f"The time after at must fall inside the selection, {start:.3f} to "
            f"{end:.3f} seconds. Use select first to cover it."
        )

    ctx = _Context(sound, start, end, o)
    spoken = []
    lines = [
        "SPECTROGRAM COMMAND",
        "",
        "Options: " + (" ".join(o.given) if o.given else "none, so the defaults were used") + ".",
        f"Formants {o.formants}, ceiling {o.ceiling:.0f} hertz. Pitch floor "
        f"{o.pitch_floor:.0f} hertz, ceiling {o.pitch_ceiling:.0f} hertz. "
        f"Maximum frequency {o.maxfreq:.0f} hertz.",
    ]
    lines += [f"Note: {n}." for n in notes + ctx.notes]
    lines.append("")
    tables = []
    steps = (
        ("summary", lambda: _summary(ctx, o, name, frames, regions)),
        ("power", lambda: _power(ctx, o)),
        ("formants", lambda: _formants(ctx, o, frames, regions)),
        ("harmonics", lambda: _harmonics(ctx, o, frames, regions)),
    )
    for section, fn in steps:
        if not o.wants(section):
            continue
        s, l, t = fn()
        spoken += s
        lines += l + [""]
        tables += t
    for n in notes + ctx.notes:
        spoken.append(f"Note: {n}.")
    return Result(spoken, lines, tables)
