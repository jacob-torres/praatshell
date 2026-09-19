"""Measurements turned into prose, with measurement and guesswork kept apart.

Two rules govern this module:

1. Anything under MEASURED can be quoted in an assignment as fact.
2. Anything under LIKELY is the tool reasoning from thresholds, and always
   carries a confidence, the reason for it, and the runner-up reading.
"""

import numpy as np

from . import events, fmt

# Adult male averages (Peterson and Barney). Formants scale with vocal tract
# length, so these fit a smaller or larger speaker badly - which is exactly why
# a vowel guess is never reported without its confidence.
VOWELS = [
    ("the vowel in beat", 270, 2290),
    ("the vowel in bit", 390, 1990),
    ("the vowel in bet", 530, 1840),
    ("the vowel in bat", 660, 1720),
    ("the vowel in pot", 730, 1090),
    ("the vowel in bought", 570, 840),
    ("the vowel in put", 440, 1020),
    ("the vowel in boot", 300, 870),
    ("the vowel in but", 640, 1190),
    ("the vowel in bird", 490, 1350),
]

HIGH, MODERATE, LOW = "high", "moderate", "low"
_RANK = {HIGH: 2, MODERATE: 1, LOW: 0}

PEAK_PROMINENCE = 10.0  # decibels a loudness peak must stand out by to be counted
TOO_SHORT = "measurable over too few frames to characterise"


class Guess:
    """A claim the tool is making, with how much it trusts itself."""

    def __init__(self, claim, level=HIGH):
        self.claim = claim
        self.level = level
        self.reasons = []
        self.runner_up = None

    def downgrade(self, level, reason):
        if _RANK[level] < _RANK[self.level]:
            self.level = level
        self.reasons.append(reason)
        return self

    def note(self, reason):
        self.reasons.append(reason)
        return self

    def lines(self):
        claim = self.claim
        if self.level == LOW:
            claim = f"possibly {claim}, but treat this as an open question"
        out = [f"  {claim}."]
        sentence = f"  Confidence: {self.level}."
        if self.reasons:
            sentence += " " + " ".join(self.reasons)
        if self.runner_up:
            sentence += f" Next most plausible reading: {self.runner_up}."
        out.append(sentence)
        return out


# --- overview ---------------------------------------------------------


def overview(frames, sound_name, start, end):
    snd = frames.sound
    lines = [
        f"Sound: {sound_name}.",
        f"Stretch described: {fmt.secs(start)} to {fmt.secs(end)}, "
        f"lasting {fmt.duration(end - start)}.",
        f"Sampling frequency {snd.sampling_frequency:.0f} hertz, so the recording "
        f"carries no information above {snd.sampling_frequency / 2:.0f} hertz.",
        f"Channels: {snd.n_channels}.",
    ]
    values = frames.window_samples()
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    lines.append(f"Peak amplitude {fmt.amp(peak)}.")
    if peak > 0.99:
        lines.append(
            "Warning: the waveform reaches full scale, so it may be clipped and "
            "the measurements distorted."
        )
    for note in frames.notes:
        lines.append(f"Note: {note}.")
    return lines


# --- regions ----------------------------------------------------------


def describe_region(region, frames, index, total):
    s = region.stats
    lines = [
        f"Region {index} of {total}: {fmt.secs(frames.absolute(region.start))} to "
        f"{fmt.secs(frames.absolute(region.end))}, lasting "
        f"{fmt.duration(region.duration)}.",
        f"  Character: {events.PLAIN[region.kind]}.",
    ]

    if region.kind == events.VOICED:
        lines += _voiced_lines(s)
    elif region.kind == events.SILENCE:
        lines.append(
            f"  Mean level {fmt.db(s.get('intensity_mean'))}, which is at or below "
            "the silence threshold for this recording."
        )
    else:
        lines += _noise_lines(s)

    if region.kind != events.SILENCE:
        lines += _amplitude_lines(s)
        lines += _band_lines(s, frames)

    return lines


def _voiced_lines(s):
    lines = []
    if s.get("cycles") and s.get("period"):
        lines.append(
            f"  Periodic: about {s['cycles']:.0f} cycles, mean period "
            f"{s['period'] * 1000:.1f} milliseconds."
        )
    if s.get("f0"):
        line = f"  Fundamental frequency averages {fmt.hz_plain(s['f0'])}"
        if s.get("f0_min") and s.get("f0_max"):
            line += (
                f", ranging {s['f0_min']:.0f} to {s['f0_max']:.0f} hertz"
                f" and {_contour_word(s.get('f0_slope'))}"
            )
        lines.append(line + ".")
    if s.get("jitter") is not None:
        j = s["jitter"]
        steadiness = "very regular" if j < 0.01 else "regular" if j < 0.02 else "irregular"
        lines.append(
            f"  Cycle-to-cycle variation is {fmt.pct(j)}, so the vibration is "
            f"{steadiness}."
        )
    if s.get("hnr") is not None:
        lines.append(
            f"  Harmonics-to-noise ratio {fmt.db(s['hnr'])}: the higher this is, "
            "the more the sound is tone rather than noise."
        )
    f1, f2, f3 = s.get("f1"), s.get("f2"), s.get("f3")
    if f1 and f2:
        line = f"  Formants: F1 {f1:.0f} hertz, F2 {f2:.0f} hertz"
        if f3:
            line += f", F3 {f3:.0f} hertz"
        lines.append(line + ".")
    return lines


def _noise_lines(s):
    lines = []
    if s.get("cog"):
        lines.append(
            f"  Energy centre of gravity {fmt.hz_plain(s['cog'])}: the higher this "
            "is, the more the energy sits at the top of the frequency range."
        )
    if s.get("zcr"):
        lines.append(
            f"  The waveform crosses zero about {s['zcr']:.0f} times per second, "
            "a measure of how noisy rather than periodic it is."
        )
    if s.get("voiced_fraction"):
        lines.append(
            f"  Pitch was detectable in {fmt.pct(s['voiced_fraction'])} of frames."
        )
    return lines


def _amplitude_lines(s):
    lines = []
    if s.get("peak_amp") is not None:
        lines.append(
            f"  Amplitude peaks at {fmt.amp(s['peak_amp'])} at "
            f"{fmt.secs(s['peak_time'])}; mean level {fmt.db(s.get('intensity_mean'))}."
        )
    if s.get("clipped"):
        lines.append("  Warning: this region touches full scale and may be clipped.")
    if "attack" in s:
        lines.append(f"  Envelope shape: it {envelope_phrase(s)}.")
    if s.get("asymmetry") is not None and abs(s["asymmetry"]) > 0.25:
        direction = "positive" if s["asymmetry"] > 0 else "negative"
        lines.append(
            f"  The waveform leans {direction}: its peaks are markedly larger on "
            "one side of zero than the other."
        )
    return lines


def envelope_phrase(s):
    """How the loudness rises, holds and falls, skipping the zero-length parts."""
    parts = []
    if s["attack"] >= 0.005:
        parts.append(f"rises over the first {fmt.ms(s['attack'])}")
    else:
        parts.append("starts already at its loudest")
    parts.append(f"holds near its peak for {fmt.ms(s['steady'])}")
    if s["decay"] >= 0.005:
        parts.append(f"falls away over the last {fmt.ms(s['decay'])}")
    else:
        parts.append("is cut off while still loud")
    return ", then ".join(parts)


def _band_lines(s, frames):
    if not s.get("bands") or not frames.band_names:
        return []
    bands = [b for b in s["bands"] if b is not None]
    if not bands:
        return []
    loudest = int(np.argmax(bands))
    top = max(bands)
    strong = [
        frames.band_names[i] for i, b in enumerate(bands) if b is not None and b > top - 10
    ]
    lines = [f"  Most energy sits in {frames.band_names[loudest]}."]
    if len(strong) > 1:
        lines.append(f"  Bands within 10 decibels of the strongest: {fmt.join(strong)}.")
    weak = [
        frames.band_names[i] for i, b in enumerate(bands) if b is not None and b < top - 25
    ]
    if weak:
        lines.append(f"  Little energy in {fmt.join(weak)}.")
    return lines


def _contour_word(slope):
    if slope is None:
        return "holding steady"
    if slope > 25:
        return "rising"
    if slope < -25:
        return "falling"
    return "holding roughly level"


# --- the guessing layer ----------------------------------------------


def guess_region(region, frames):
    s = region.stats
    if region.kind == events.SILENCE:
        g = Guess("a pause, or the closure phase of a stop consonant")
        g.note(
            "It is measured as silence because its level is more than "
            f"{events.SILENCE_DROP:.0f} decibels below the loudest part of the sound."
        )
        g.runner_up = "background noise between words"
        return g

    if region.kind == events.BURST:
        g = Guess("a stop release burst, as in p, t or k")
        g.note(
            f"It lasts only {fmt.ms(region.duration)}, follows silence, and jumps "
            f"at least {events.BURST_RISE:.0f} decibels in level."
        )
        g.runner_up = "a click or other transient in the recording"
        if region.duration < 0.005:
            g.downgrade(LOW, "It is extremely short, so it may be a recording artefact.")
        return g

    if region.kind == events.FRICATION:
        g = Guess("voiceless frication, as in s, sh or f")
        cog = s.get("cog")
        if cog:
            margin = cog - events.FRICATION_COG
            g.note(
                f"Its centre of gravity, {fmt.hz_plain(cog)}, sits "
                f"{abs(margin):.0f} hertz "
                f"{'above' if margin > 0 else 'below'} the "
                f"{events.FRICATION_COG:.0f} hertz threshold this tool uses."
            )
            if margin < 400:
                g.downgrade(
                    MODERATE, "That is a narrow margin, so the reading is not clear-cut."
                )
        if s.get("voiced_fraction", 0) > 0.3:
            g.downgrade(
                MODERATE,
                f"Pitch was detectable in {fmt.pct(s['voiced_fraction'])} of frames, "
                "which points to a voiced fricative instead.",
            )
            g.runner_up = "a voiced fricative such as z or v"
        else:
            g.runner_up = "aspiration following a stop release"
        _duration_and_level(g, region, s)
        return g

    if region.kind == events.APERIODIC:
        g = Guess("a weak or transitional sound", MODERATE)
        g.note(
            "It is above the silence threshold but has neither clear periodicity "
            "nor the high-frequency energy of frication."
        )
        g.runner_up = "a nasal, an approximant, or the edge of a neighbouring sound"
        _duration_and_level(g, region, s)
        return g

    # voiced
    g = Guess("a voiced sound")
    hnr = s.get("hnr")
    if hnr is not None:
        g.note(
            f"Pitch is measurable and the harmonics-to-noise ratio is "
            f"{fmt.db(hnr)}, against a voicing threshold of "
            f"{events.VOICING_HNR:.0f} decibels."
        )
        if hnr < events.VOICING_HNR + 3:
            g.downgrade(MODERATE, "That is close to the threshold.")

    f1, f2 = s.get("f1"), s.get("f2")
    if f1 and f2 and region.duration >= 0.04:
        name, runner_up, ratio = _nearest_vowel(f1, f2)
        g.claim = f"a vowel, closest to {name}"
        g.runner_up = runner_up
        g.note(
            f"F1 {f1:.0f} and F2 {f2:.0f} hertz are nearest the reference values "
            f"for {name}."
        )
        if ratio > 0.8:
            g.downgrade(
                MODERATE,
                "The next vowel along is almost equally close, so the vowel "
                "identity is genuinely ambiguous here.",
            )
        g.note(
            "Reference values are adult male averages, so a higher or lower voice "
            "will shift the match."
        )
        spread1, spread2 = s.get("f1_spread"), s.get("f2_spread")
        if spread2 and f2 and spread2 / f2 > 0.15:
            g.downgrade(
                MODERATE,
                f"F2 moves by {spread2:.0f} hertz across the region, so this may be "
                "a diphthong or a formant transition rather than a steady vowel.",
            )
        elif spread1 and f1 and spread1 / f1 > 0.2:
            g.downgrade(MODERATE, "Formant tracking was unsteady across the region.")
    elif region.duration < 0.04:
        g.runner_up = "a voiced consonant such as a nasal or a glide"

    _duration_and_level(g, region, s)
    return g


def _duration_and_level(guess, region, stats):
    if region.duration < 0.030:
        guess.downgrade(
            LOW,
            f"The region is only {fmt.ms(region.duration)} long, too short to "
            "measure formants reliably.",
        )
    mean = stats.get("intensity_mean")
    peak = stats.get("intensity_max")
    if mean is not None and peak is not None and peak - mean > 12:
        guess.downgrade(
            MODERATE, "Its level varies a lot, so a single reading averages over change."
        )


def _nearest_vowel(f1, f2):
    """Nearest reference vowel in log frequency, plus how close the runner-up is."""
    scored = sorted(
        (
            (np.hypot(np.log(f1 / rf1), np.log(f2 / rf2)), name)
            for name, rf1, rf2 in VOWELS
        )
    )
    best_d, best = scored[0]
    second_d, second = scored[1]
    ratio = best_d / second_d if second_d > 0 else 1.0
    return best, second, ratio


# --- assembled reports ------------------------------------------------


def full_report(frames, regions, sound_name, start, end):
    lines = ["ACOUSTIC DESCRIPTION", ""]
    lines += overview(frames, sound_name, start, end)
    lines += ["", f"The tool found {len(regions)} regions.", ""]
    for i, region in enumerate(regions, 1):
        lines.append("MEASURED")
        lines += describe_region(region, frames, i, len(regions))
        lines.append("LIKELY (the tool's guess, not a measurement)")
        lines += guess_region(region, frames).lines()
        lines.append("")
    lines += contour_summary(frames)
    return lines


def contour_summary(frames):
    """Pitch and loudness across the whole stretch, not region by region."""
    lines = ["MEASURED: the stretch as a whole"]
    f0 = frames.f0[~np.isnan(frames.f0)]
    peaks = None
    if f0.size:
        slope = np.polyfit(np.arange(f0.size), f0, 1)[0] * f0.size
        lines += [
            f"  Pitch was measurable in {fmt.pct(f0.size / max(len(frames.f0), 1))} "
            "of frames.",
            f"  Fundamental frequency runs from {f0.min():.0f} to {f0.max():.0f} "
            f"hertz. Median {np.median(f0):.0f} hertz, mean {f0.mean():.0f} hertz.",
            f"  Net change across the voiced parts: {slope:+.0f} hertz.",
        ]
    else:
        lines.append("  No pitch could be measured anywhere in this stretch.")

    intensity = frames.intensity[~np.isnan(frames.intensity)]
    if intensity.size:
        peaks = _count_peaks(frames.intensity)
        lines += [
            f"  Level runs from {intensity.min():.0f} to {intensity.max():.0f} "
            f"decibels, mean {intensity.mean():.0f} decibels.",
            f"  The loudness curve has {peaks} clear peak"
            f"{'s' if peaks != 1 else ''}, counting only rises and falls of at "
            f"least {PEAK_PROMINENCE:.0f} decibels.",
        ]

    lines.append("LIKELY (the tool's guess, not a measurement)")
    shape = _overall_shape(f0) if f0.size else None
    if f0.size and shape != TOO_SHORT:
        guess = Guess(f"the pitch contour over this stretch is best described as {shape}")
        if f0.size < len(frames.f0) * 0.4:
            guess.downgrade(
                MODERATE,
                f"Pitch was only measurable in {fmt.pct(f0.size / len(frames.f0))} "
                "of frames, so the contour is pieced together from gaps.",
            )
        lines += guess.lines()
    if peaks is not None:
        guess = Guess(
            f"this stretch contains about {peaks} syllable"
            f"{'s' if peaks != 1 else ''}",
            MODERATE,
        )
        guess.note(
            "Syllables usually show up as peaks in loudness, but one syllable can "
            "produce more than one peak and an unstressed syllable can produce none."
        )
        lines += guess.lines()
    return lines


def _overall_shape(f0):
    n = f0.size
    if n < 6:
        return TOO_SHORT
    third = max(1, n // 3)
    a, b, c = f0[:third].mean(), f0[third:-third].mean(), f0[-third:].mean()
    span = f0.max() - f0.min()
    if span < 10:
        return "level"
    if b > a + 5 and b > c + 5:
        return "a rise then a fall"
    if b < a - 5 and b < c - 5:
        return "a fall then a rise"
    if c > a + 5:
        return "rising"
    if c < a - 5:
        return "falling"
    return "level"


def _count_peaks(intensity, prominence=None):
    prominence = PEAK_PROMINENCE if prominence is None else prominence
    v = np.where(np.isnan(intensity), -np.inf, intensity)
    if v.size < 3:
        return 0
    count = 0
    for i in range(1, v.size - 1):
        if v[i] >= v[i - 1] and v[i] > v[i + 1]:
            left = v[:i].min() if i else v[i]
            right = v[i + 1 :].min()
            if v[i] - max(left, right) >= prominence:
                count += 1
    return count


def summary(frames, regions, start, end):
    """The two to four sentences the shell speaks."""
    kinds = {}
    for r in regions:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    parts = [f"{n} {events.SHORT[k]}" for k, n in kinds.items()]
    f0 = frames.f0[~np.isnan(frames.f0)]
    out = [
        f"Described {fmt.duration(end - start)} of sound as {len(regions)} "
        f"region{'s' if len(regions) != 1 else ''}: {fmt.join(parts)}."
    ]
    if f0.size:
        shape = _overall_shape(f0)
        out.append(
            f"Median pitch {np.median(f0):.0f} hertz, "
            + (
                "over too few frames to describe a contour."
                if shape == TOO_SHORT
                else f"and the contour looks {shape}."
            )
        )
    longest = max(regions, key=lambda r: r.duration) if regions else None
    if longest is not None and longest.kind == events.VOICED:
        s = longest.stats
        if s.get("f1") and s.get("f2"):
            name, _, _ = _nearest_vowel(s["f1"], s["f2"])
            out.append(
                f"The longest region is voiced, with F1 {s['f1']:.0f} and F2 "
                f"{s['f2']:.0f} hertz, closest to {name}."
            )
    return out
