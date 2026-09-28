"""Measurements turned into prose, with measurement and guesswork kept apart.

Two rules govern this module:

1. Anything under MEASURED can be quoted in an assignment as fact.
2. Anything under LIKELY is the tool reasoning from thresholds, and always
   carries a confidence, the reason for it, and the runner-up reading.
"""

import datetime

import numpy as np

from . import analysis, events, fmt

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
# English aspirated stops rarely pass 120 ms, so a longer gap between a noise
# and the voicing after it is probably not a release and its vowel at all.
VOT_IMPLAUSIBLE = 0.120
# A vowel before a voiceless stop devoices at its end while its formants hold.
# Its tail ends where the level finally falls this far below the vowel's own.
VOWEL_TAIL_DROP = 15.0
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
            f"F1 {f1:.0f} hertz and F2 {f2:.0f} hertz are nearest the reference "
            f"values for {name}."
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


# --- voice onset time -------------------------------------------------


class VOT:
    """A voice onset time, or the reason there is not one.

    Falsy when nothing could be measured, so `if not v: say(v.reason)`.
    Times are in the frames' clock.
    """

    def __init__(self, seconds=None, release=None, anchor=None, onset=None, reason=None):
        self.seconds = seconds
        self.release = release  # the region the release run starts in
        self.anchor = anchor  # where the release actually is, in that region
        self.onset = onset  # where voicing starts
        self.reason = reason

    def __bool__(self):
        return self.seconds is not None

    @property
    def implausible(self):
        return bool(self) and self.seconds > VOT_IMPLAUSIBLE


def vot_at(frames, regions, index):
    """Voice onset time for the voiced region at index.

    The release is the run of non-silent, non-voiced regions immediately
    before the voicing: the burst, plus any aspiration or frication that
    follows it before the voice starts. A burst that runs straight into
    aspiration is often too long to be labelled a burst, so the label is not
    what identifies it - its position is. Silence in between means the two
    events belong to different syllables, and no VOT is measured across it.

    The run's own start is only as precise as the intensity contour that drew
    it, so the release transient is looked for in the waveform and used
    instead when there is one. Only a positive VOT can be found this way:
    prevoicing sits inside the voiced region itself and leaves no separate
    release to anchor on.
    """
    i = index - 1
    if i < 0:
        return VOT(reason="voicing opens the stretch, so no release precedes it")
    if regions[i].kind == events.SILENCE:
        return VOT(reason="silence runs straight into the voicing, with no release")
    while i > 0 and regions[i - 1].kind not in (events.SILENCE, events.VOICED):
        i -= 1
    release = regions[i]
    onset = regions[index].start
    anchor = events.find_burst(frames.sound, release.start, onset)
    if anchor is None or not release.start <= anchor < onset:
        anchor = release.start
    return VOT(onset - anchor, release, anchor, onset)


def vot(frames, regions):
    """Voice onset time for the first vowel in the stretch. See vot_at.

    Select a stretch starting just before a later stop to measure that one
    instead, or use the vowels command, which measures every vowel at once.
    """
    first = next((i for i, r in enumerate(regions) if r.kind == events.VOICED), None)
    if first is None:
        return VOT(reason="no voicing was found in this stretch")
    return vot_at(frames, regions, first)


def vowel_end(frames, regions, index):
    """Where the vowel at index really ends, past the frames that lost voicing.

    A vowel before a voiceless stop devoices at its end: the pitch tracker
    gives up while F1 and F2 carry on unchanged and the sound is still loud.
    Voicing alone therefore cuts a vowel short, so the tail is followed for as
    long as the formants hold their place and the level stays up. The start is
    left alone: voicing onset is where the vowel begins by definition, and it
    is the same instant that ends the voice onset time.
    """
    region = regions[index]
    a = frames.at(region.start)
    b = max(a, frames.at(max(region.start, region.end - 1e-6)))
    core = slice(a, b + 1)
    db = frames.intensity[core]
    db = db[~np.isnan(db)]
    f1 = frames.formants[core, 0]
    f2 = frames.formants[core, 1]
    f1, f2 = f1[~np.isnan(f1)], f2[~np.isnan(f2)]
    if not db.size or not f1.size or not f2.size:
        return region.end
    f1_mean, f2_mean = float(f1.mean()), float(f2.mean())
    floor = db.mean() - VOWEL_TAIL_DROP
    limit = next(
        (r.start for r in regions[index + 1 :] if r.kind == events.VOICED),
        frames.window[1],
    )

    i = b + 1
    while i < len(frames.times) and frames.times[i] < limit:
        level = frames.intensity[i]
        if np.isnan(level) or level < floor:
            break
        one, two = frames.formants[i, 0], frames.formants[i, 1]
        if np.isnan(one) or np.isnan(two):
            break
        if abs(two - f2_mean) > max(150.0, 0.15 * f2_mean):
            break
        if abs(one - f1_mean) > max(150.0, 0.20 * f1_mean):
            break
        i += 1
    if i == b + 1:
        return region.end
    return float(min(frames.times[i] if i < len(frames.times) else limit, limit))


# --- the vowel table --------------------------------------------------

VOWEL_COLUMNS = [
    ("vowel", "No."),
    ("start_s", "Start"),
    ("end_s", "End"),
    ("duration_ms", "Dur"),
    ("voiced_ms", "Voiced"),
    ("vot_ms", "VOT"),
    ("f0_mean_hz", "F0"),
    ("f0_min_hz", "F0 min"),
    ("f0_max_hz", "F0 max"),
    ("period_ms", "Period"),
    ("f1_hz", "F1"),
    ("f2_hz", "F2"),
    ("f3_hz", "F3"),
    ("closest_reference_vowel", "Closest reference vowel"),
]


def vowel_rows(frames, regions):
    """One row per voiced region: the vowels, measured.

    A voiced region is not always a vowel - nasals, laterals and voiced
    fricatives are voiced too - so every row is a candidate, and the closest
    reference vowel is a guess, never a measurement.
    """
    rows = []
    for i, r in enumerate(regions):
        if r.kind != events.VOICED:
            continue
        s = r.stats
        v = vot_at(frames, regions, i)
        end = vowel_end(frames, regions, i)
        row = {
            "number": len(rows) + 1,
            "start": frames.absolute(r.start),
            "end": frames.absolute(end),
            "duration": end - r.start,
            "voiced_end": frames.absolute(r.end),
            "voiced_duration": r.duration,
            "tail": end - r.end,
            "vot": v.seconds,
            "vot_reason": v.reason,
            "release": frames.absolute(v.anchor) if v else None,
            "release_kind": v.release.kind if v else None,
            "f0": s.get("f0"),
            "f0_min": s.get("f0_min"),
            "f0_max": s.get("f0_max"),
            "f0_slope": s.get("f0_slope"),
            "period": 1.0 / s["f0"] if s.get("f0") else None,
            "f1": s.get("f1"),
            "f2": s.get("f2"),
            "f3": s.get("f3"),
            "f1_spread": s.get("f1_spread"),
            "f2_spread": s.get("f2_spread"),
            "intensity": s.get("intensity_mean"),
            "hnr": s.get("hnr"),
            "jitter": s.get("jitter"),
            "nearest": None,
            "runner_up": None,
            "ambiguous": False,
        }
        if row["f1"] and row["f2"]:
            name, runner, ratio = _nearest_vowel(row["f1"], row["f2"])
            row["nearest"] = name
            row["runner_up"] = runner
            row["ambiguous"] = ratio > 0.8
        rows.append(row)
    return rows


def vowel_csv_rows(rows):
    """The same rows as the CSV's columns, in VOWEL_COLUMNS order."""
    out = []
    for r in rows:
        out.append([
            r["number"],
            _num(r["start"], 3),
            _num(r["end"], 3),
            _num(r["duration"] * 1000, 1),
            _num(r["voiced_duration"] * 1000, 1),
            _num(r["vot"] * 1000 if r["vot"] is not None else None, 1),
            _num(r["f0"], 1),
            _num(r["f0_min"], 1),
            _num(r["f0_max"], 1),
            _num(r["period"] * 1000 if r["period"] else None, 2),
            _num(r["f1"], 1),
            _num(r["f2"], 1),
            _num(r["f3"], 1),
            r["nearest"] or "",
        ])
    return out


def _num(v, places):
    """A number for the CSV, or an empty cell. Never a nan."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.{places}f}"


def _table(headings, rows, align):
    """Space-aligned columns. No drawn characters: a screen reader reads them."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [
        max([len(h)] + [len(row[i]) for row in cells])
        for i, h in enumerate(headings)
    ]

    def line(values):
        out = []
        for value, width, how in zip(values, widths, align):
            out.append(value.ljust(width) if how == "l" else value.rjust(width))
        return "  " + "  ".join(out).rstrip()

    return [line(headings), line(["-" * w for w in widths])] + [
        line(row) for row in cells
    ]


def vowel_table(frames, regions, rows=None):
    """The heading, the notes and the aligned table.

    describe carries this too, so the two commands cannot drift apart: the
    table is built here once and both reports print the same one.
    """
    rows = vowel_rows(frames, regions) if rows is None else rows
    if not rows:
        return [
            "VOWEL TABLE",
            "",
            "No voiced region was found here, so there are no vowels to list.",
        ]
    lines = [
        "VOWEL TABLE",
        "",
        f"Vowels found: {len(rows)}.",
        "",
        "One row per voiced region. Times are in seconds, durations, periods "
        "and voice onset times in milliseconds, frequencies in hertz. An empty "
        "cell is a measurement that could not be made.",
        "",
        "Dur runs from the onset of voicing to the offset of the formants. "
        "Voiced is the part of that the pitch tracker found a pitch in. The "
        "two differ when a vowel devoices into a voiceless consonant after it, "
        "which is ordinary in English and not a fault in the recording.",
        "",
        "Period is the time a single cycle of the vocal folds takes, in "
        "milliseconds. It is one second divided by F0, so to get the "
        "frequency back divide 1000 by it, not 1: 100 hertz is a period of "
        "10 milliseconds, and 1000 divided by 10 is 100 again. It carries "
        "the same information as F0 and is given because a waveform is "
        "measured in time, not in frequency.",
        "",
    ]

    display = [
        [
            str(r["number"]),
            f"{r['start']:.3f}",
            f"{r['end']:.3f}",
            f"{r['duration'] * 1000:.0f}",
            f"{r['voiced_duration'] * 1000:.0f}",
            "" if r["vot"] is None else f"{r['vot'] * 1000:.0f}",
            _cell(r["f0"]),
            _cell(r["f0_min"]),
            _cell(r["f0_max"]),
            "" if r["period"] is None else f"{r['period'] * 1000:.2f}",
            _cell(r["f1"]),
            _cell(r["f2"]),
            _cell(r["f3"]),
            r["nearest"] or "",
        ]
        for r in rows
    ]
    lines += _table([h for _, h in VOWEL_COLUMNS], display, "rrrrrrrrrrrrrl")

    lines += [
        "",
        "MEASURED and LIKELY are mixed in that table, so read the last column "
        "with care: every number in it is measured, but the closest reference "
        "vowel is the tool's guess. Voiced regions include nasals, laterals "
        "and voiced fricatives, so a row is a vowel candidate, not a vowel.",
        "",
        "Voice onset time is measured from the noise immediately before the "
        f"vowel, so a figure over {fmt.ms(VOT_IMPLAUSIBLE)} usually means that "
        "noise was not a stop release. Prevoicing does not show: only a "
        "positive voice onset time can be found this way.",
    ]
    return lines


def vowel_csv(frames, regions, rows=None):
    """The vowel table as (columns, rows) for a CSV, or None if no vowels."""
    rows = vowel_rows(frames, regions) if rows is None else rows
    if not rows:
        return None
    return [name for name, _ in VOWEL_COLUMNS], vowel_csv_rows(rows)


def vowel_report(frames, regions, rows=None):
    """The vowel table as report lines, plus the columns and rows for the CSV.

    The sound and the stretch are not repeated here: every report carries them
    already, in the header written around these lines.
    """
    rows = vowel_rows(frames, regions) if rows is None else rows
    lines = vowel_table(frames, regions, rows)
    if rows:
        lines += ["", "VOWEL BY VOWEL", ""]
        for r in rows:
            lines += _vowel_block(r)
            lines.append("")
    return lines, [name for name, _ in VOWEL_COLUMNS], vowel_csv_rows(rows)


def _cell(v):
    return "" if v is None else f"{v:.0f}"


def _vowel_block(r):
    """One vowel, a fact per line, measurement kept apart from guesswork."""
    lines = [
        "MEASURED",
        f"Vowel {r['number']}: {fmt.secs(r['start'])} to {fmt.secs(r['end'])}, "
        f"lasting {fmt.duration(r['duration'])}.",
    ]
    if r["tail"] > 1e-6:
        lines.append(
            f"  Voicing stops at {fmt.secs(r['voiced_end'])}, "
            f"{fmt.duration(r['tail'])} before the end: the vowel devoices into "
            f"the consonant after it while F1 and F2 hold their place. Voiced "
            f"for {fmt.duration(r['voiced_duration'])} of its "
            f"{fmt.duration(r['duration'])}."
        )
    if r["vot"] is None:
        lines.append(f"  Voice onset time: not measurable, {r['vot_reason']}.")
    else:
        lines.append(
            f"  Voice onset time: {fmt.ms(r['vot'])}, from the release at "
            f"{fmt.secs(r['release'])}, labelled {events.PLAIN[r['release_kind']]}."
        )
        if r["vot"] > VOT_IMPLAUSIBLE:
            lines.append(
                f"  Treat that with suspicion: {fmt.ms(VOT_IMPLAUSIBLE)} is already "
                "long for an aspirated stop, so the noise before this vowel is "
                "more likely a fricative or another word than a release. Select "
                "a shorter stretch around the stop you mean."
            )
    if r["f0"]:
        lines.append(
            f"  Pitch: mean {r['f0']:.0f} hertz, from {r['f0_min']:.0f} to "
            f"{r['f0_max']:.0f} hertz."
        )
        lines.append(
            f"  Pitch period: {r['period'] * 1000:.2f} milliseconds, the time one "
            "cycle takes. Divide 1000 by it to get the mean above back in hertz."
        )
        if r["f0_slope"]:
            lines.append(f"  Pitch slope: {r['f0_slope']:+.0f} hertz per second.")
    else:
        lines.append("  Pitch: not measurable in this region.")
    lines.append(
        f"  Formants: F1 {fmt.hz_plain(r['f1'])}, F2 {fmt.hz_plain(r['f2'])}, "
        f"F3 {fmt.hz_plain(r['f3'])}."
    )
    if r["f1_spread"] is not None and r["f2_spread"] is not None:
        lines.append(
            f"  Formants vary across the region by {r['f1_spread']:.0f} hertz in "
            f"F1 and {r['f2_spread']:.0f} hertz in F2."
        )
    if r["intensity"]:
        lines.append(f"  Mean level: {fmt.db(r['intensity'])}.")
    if r["jitter"] is not None:
        lines.append(f"  Jitter: {fmt.pct(r['jitter'])}.")

    if r["nearest"]:
        guess = Guess(f"this is {r['nearest']}")
        guess.runner_up = r["runner_up"]
        guess.note(
            f"F1 {r['f1']:.0f} hertz and F2 {r['f2']:.0f} hertz are nearest the "
            "reference values for it, which are adult male averages."
        )
        if r["ambiguous"]:
            guess.downgrade(
                MODERATE,
                "The next vowel along is almost equally close, so the identity "
                "is genuinely ambiguous here.",
            )
        if r["f2_spread"] and r["f2"] and r["f2_spread"] / r["f2"] > 0.15:
            guess.downgrade(
                MODERATE,
                f"F2 moves by {r['f2_spread']:.0f} hertz across the region, so "
                "this may be a diphthong or a formant transition.",
            )
        lines.append("LIKELY (the tool's guess, not a measurement)")
        lines += guess.lines()
    return lines


def vowel_summary(rows):
    """The two or three sentences the shell speaks."""
    if not rows:
        return ["No voiced region was found, so there are no vowels to measure."]
    out = [
        f"Measured {len(rows)} vowel{'s' if len(rows) != 1 else ''}, "
        f"{fmt.join(fmt.duration(r['duration']) for r in rows)} long."
    ]
    with_vot = [r for r in rows if r["vot"] is not None]
    if with_vot:
        out.append(
            "Voice onset time: "
            + fmt.join(
                f"{fmt.ms(r['vot'])} for vowel {r['number']}"
                + (", which is too long to trust" if r["vot"] > VOT_IMPLAUSIBLE else "")
                for r in with_vot
            )
            + "."
        )
    named = [r for r in rows if r["nearest"]]
    if named:
        out.append(
            "Closest reference vowels: "
            + fmt.join(f"vowel {r['number']}, {r['nearest']}" for r in named)
            + "."
        )
    return out


# --- the consonant table ----------------------------------------------

CONSONANT_COLUMNS = [
    ("consonant", "No."),
    ("start_s", "Start"),
    ("end_s", "End"),
    ("duration_ms", "Dur"),
    ("kind", "Kind"),
    ("vot_ms", "VOT"),
    ("cog_hz", "CoG"),
    ("zero_crossings_per_s", "Zero X"),
    ("intensity_db", "Level"),
    ("voiced_percent", "Voiced %"),
    ("likely", "Likely"),
]

# Short names for the table, where the full phrases would not fit.
CONSONANT_KIND = {
    events.SILENCE: "closure",
    events.BURST: "burst",
    events.FRICATION: "frication",
    events.APERIODIC: "aperiodic",
}

CONSONANT_LIKELY = {
    events.SILENCE: "a stop closure, or a pause",
    events.BURST: "a stop release",
    events.FRICATION: "a fricative, or aspiration after a stop",
    events.APERIODIC: "a weak fricative, an approximant, or aspiration",
}

SIBILANT_COG = 3500.0  # a centre of gravity this high suggests s or sh


def consonant_rows(frames, regions):
    """One row per region that is not a vowel.

    Leading and trailing silence is left out, being room tone rather than a
    consonant. Silence between two sounds stays, because a stop closure is
    silence and the length of it is a real measurement.

    Voiced consonants do not appear at all. Nasals, laterals and voiced
    fricatives are periodic, so they sit inside a voiced region and this
    classifier cannot tell them from the vowel beside them. What is listed is
    the voiceless consonants, the closures, the bursts and the aspiration.
    """
    if not regions:
        return []
    speech = [i for i, r in enumerate(regions) if r.kind != events.SILENCE]
    if not speech:
        return []
    first, last = speech[0], speech[-1]

    # Where each voice onset time is anchored, so that the row holding the
    # release carries it and the two tables cannot disagree.
    anchors = {}
    for i, r in enumerate(regions):
        if r.kind != events.VOICED:
            continue
        v = vot_at(frames, regions, i)
        if not v:
            continue
        for n, region in enumerate(regions):
            if region.start <= v.anchor < region.end:
                anchors[n] = v.seconds
                break

    # A vowel is followed through the devoiced tail past its region's end, so
    # the region after a vowel can begin inside it. Trim that overlap, or the
    # same stretch is counted twice: once ending a vowel, once as a consonant.
    vowel_spans = [
        (r.start, vowel_end(frames, regions, i))
        for i, r in enumerate(regions)
        if r.kind == events.VOICED
    ]

    rows = []
    for i, r in enumerate(regions):
        if r.kind == events.VOICED:
            continue
        if r.kind == events.SILENCE and not first < i < last:
            continue
        start = r.start
        for vowel_start, vowel_finish in vowel_spans:
            # Only a vowel that began before this region can run into it.
            if vowel_start < r.start and vowel_finish > start:
                start = min(vowel_finish, r.end)
        if start >= r.end:
            continue  # a vowel's tail covers the whole of this region
        # Measured again over what is left, so a trimmed region's centre of
        # gravity and level describe the consonant rather than the vowel.
        s = r.stats if start == r.start else events.measure(
            events.Region(r.kind, start, r.end), frames
        )
        rows.append(
            {
                "number": len(rows) + 1,
                "start": frames.absolute(start),
                "end": frames.absolute(r.end),
                "duration": r.end - start,
                "kind": r.kind,
                "vot": anchors.get(i),
                "cog": s.get("cog"),
                "zcr": s.get("zcr"),
                "intensity": s.get("intensity_mean"),
                "peak_amp": s.get("peak_amp"),
                "voiced_fraction": s.get("voiced_fraction"),
            }
        )
    return rows


def consonant_csv_rows(rows):
    """The same rows as the CSV's columns, in CONSONANT_COLUMNS order."""
    out = []
    for r in rows:
        out.append(
            [
                r["number"],
                _num(r["start"], 3),
                _num(r["end"], 3),
                _num(r["duration"] * 1000, 1),
                CONSONANT_KIND[r["kind"]],
                _num(r["vot"] * 1000 if r["vot"] is not None else None, 1),
                _num(r["cog"], 1),
                _num(r["zcr"], 1),
                _num(r["intensity"], 1),
                _num(
                    r["voiced_fraction"] * 100
                    if r["voiced_fraction"] is not None
                    else None,
                    1,
                ),
                CONSONANT_LIKELY[r["kind"]],
            ]
        )
    return out


def consonant_csv(frames, regions, rows=None):
    """The consonant table as (columns, rows) for a CSV, or None if empty."""
    rows = consonant_rows(frames, regions) if rows is None else rows
    if not rows:
        return None
    return [name for name, _ in CONSONANT_COLUMNS], consonant_csv_rows(rows)


def consonant_table(frames, regions, rows=None):
    """The heading, the notes and the aligned table. describe carries it too."""
    rows = consonant_rows(frames, regions) if rows is None else rows
    if not rows:
        return [
            "CONSONANT TABLE",
            "",
            "No consonant was found here: nothing in this stretch falls "
            "outside the vowels and the silence around them.",
        ]
    lines = [
        "CONSONANT TABLE",
        "",
        f"Consonants found: {len(rows)}.",
        "",
        "One row per region that is not a vowel. Times are in seconds, "
        "durations and voice onset times in milliseconds, frequencies in "
        "hertz. CoG is the centre of gravity of the energy, the clearest "
        "clue to where in the mouth the constriction is: the further forward "
        "the constriction, the higher it usually sits. Zero X counts zero "
        "crossings per second, which rises with noise.",
        "",
        "Voiced consonants are missing from this table. Nasals, laterals and "
        "voiced fricatives are periodic, so they sit inside a voiced region "
        "and this tool cannot tell them from the vowel beside them. What is "
        "listed is the voiceless consonants, the stop closures, the bursts "
        "and the aspiration. Check the rows against what you know was said.",
        "",
        "A stop is spread across consecutive rows: a closure, then a burst, "
        "then the aspiration before the vowel. Its voice onset time is given "
        "once, on the row the release itself falls in.",
        "",
    ]

    display = [
        [
            str(r["number"]),
            f"{r['start']:.3f}",
            f"{r['end']:.3f}",
            f"{r['duration'] * 1000:.0f}",
            CONSONANT_KIND[r["kind"]],
            "" if r["vot"] is None else f"{r['vot'] * 1000:.0f}",
            _cell(r["cog"]),
            _cell(r["zcr"]),
            _cell(r["intensity"]),
            ""
            if r["voiced_fraction"] is None
            else f"{r['voiced_fraction'] * 100:.0f}",
            CONSONANT_LIKELY[r["kind"]],
        ]
        for r in rows
    ]
    lines += _table([h for _, h in CONSONANT_COLUMNS], display, "rrrrlrrrrrl")

    lines += [
        "",
        "Everything in that table is measured except the last column, which "
        "is the tool reading a manner of articulation off the region's "
        "character. It never names a place of articulation: CoG leans that "
        "way, but one number cannot carry a decision that needs the formant "
        "transitions into the vowel as well.",
    ]
    return lines


def _consonant_block(r):
    """One consonant, a fact per line, measurement kept apart from guesswork."""
    lines = [
        "MEASURED",
        f"Consonant {r['number']}: {fmt.secs(r['start'])} to "
        f"{fmt.secs(r['end'])}, lasting {fmt.duration(r['duration'])}.",
        f"  Character: {events.PLAIN[r['kind']]}.",
    ]
    if r["vot"] is not None:
        lines.append(
            f"  Voice onset time: {fmt.ms(r['vot'])}, from this release to the "
            "voicing after it."
        )
    if r["cog"]:
        lines.append(f"  Centre of gravity: {fmt.hz_plain(r['cog'])}.")
    if r["zcr"]:
        lines.append(f"  Zero crossings: {r['zcr']:.0f} per second.")
    if r["intensity"]:
        lines.append(f"  Mean level: {fmt.db(r['intensity'])}.")
    if r["peak_amp"] is not None:
        lines.append(f"  Peak amplitude: {fmt.amp(r['peak_amp'])}.")
    if r["voiced_fraction"]:
        lines.append(
            f"  Pitch was detectable in {fmt.pct(r['voiced_fraction'])} of frames."
        )

    guess = Guess(f"this is {CONSONANT_LIKELY[r['kind']]}", MODERATE)
    if r["kind"] == events.SILENCE:
        guess.note(
            "A stop closure and a pause are both silence; only what surrounds "
            "them tells the two apart."
        )
    if r["kind"] in (events.FRICATION, events.APERIODIC) and r["cog"]:
        if r["cog"] > SIBILANT_COG:
            guess.note(
                f"Its energy centres at {r['cog']:.0f} hertz, high enough for a "
                "sibilant such as s or sh."
            )
        else:
            guess.note(
                f"Its energy centres at {r['cog']:.0f} hertz, low for a "
                "sibilant, which fits a quieter fricative such as f or th, or "
                "aspiration."
            )
    if r["voiced_fraction"] and r["voiced_fraction"] > 0.5:
        guess.downgrade(
            LOW,
            f"Pitch was found in {fmt.pct(r['voiced_fraction'])} of its frames, "
            "so this may be a voiced consonant that the classifier only partly "
            "separated from the vowel beside it.",
        )
    lines.append("LIKELY (the tool's guess, not a measurement)")
    lines += guess.lines()
    return lines


def consonant_report(frames, regions, rows=None):
    """The consonant table as report lines, plus the columns and CSV rows."""
    rows = consonant_rows(frames, regions) if rows is None else rows
    lines = consonant_table(frames, regions, rows)
    if rows:
        lines += ["", "CONSONANT BY CONSONANT", ""]
        for r in rows:
            lines += _consonant_block(r)
            lines.append("")
    return lines, [name for name, _ in CONSONANT_COLUMNS], consonant_csv_rows(rows)


def consonant_summary(rows):
    """The two or three sentences the shell speaks."""
    if not rows:
        return ["No consonant was found in this stretch."]
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    parts = [f"{n} {CONSONANT_KIND[k]}" for k, n in kinds.items()]
    out = [
        f"Measured {len(rows)} consonant{'s' if len(rows) != 1 else ''}: "
        f"{fmt.join(parts)}."
    ]
    with_vot = [r for r in rows if r["vot"] is not None]
    if with_vot:
        out.append(
            "Voice onset time: "
            + fmt.join(
                f"{fmt.ms(r['vot'])} at consonant {r['number']}"
                for r in with_vot[:4]
            )
            + "."
        )
    out.append(
        "Voiced consonants are not in this table; the report says why."
    )
    return out


# --- tables for the single-measure commands ---------------------------


def events_csv(frames, regions):
    """The region timeline, as the events command reports it."""
    columns = [
        "region", "start_s", "end_s", "duration_ms", "kind",
        "intensity_db", "f0_hz",
    ]
    rows = []
    for i, r in enumerate(regions, 1):
        s = r.stats
        rows.append([
            i,
            _num(frames.absolute(r.start), 3),
            _num(frames.absolute(r.end), 3),
            _num(r.duration * 1000, 1),
            r.kind,
            _num(s.get("intensity_mean"), 1),
            _num(s.get("f0"), 1),
        ])
    return columns, rows


def wave_csv(frames, regions):
    """The waveform's shape region by region, as the wave command reports it."""
    columns = [
        "region", "start_s", "end_s", "duration_ms", "kind",
        "peak_amplitude", "peak_time_s", "rms", "asymmetry",
        "attack_ms", "steady_ms", "decay_ms", "cycles", "period_ms", "clipped",
    ]
    rows = []
    for i, r in enumerate(regions, 1):
        s = r.stats
        peak_time = s.get("peak_time")
        rows.append([
            i,
            _num(frames.absolute(r.start), 3),
            _num(frames.absolute(r.end), 3),
            _num(r.duration * 1000, 1),
            r.kind,
            _num(s.get("peak_amp"), 4),
            _num(frames.absolute(peak_time) if peak_time is not None else None, 3),
            _num(s.get("rms"), 4),
            _num(s.get("asymmetry"), 3),
            _num(s["attack"] * 1000 if "attack" in s else None, 1),
            _num(s["steady"] * 1000 if "steady" in s else None, 1),
            _num(s["decay"] * 1000 if "decay" in s else None, 1),
            _num(s.get("cycles"), 1),
            _num(s["period"] * 1000 if s.get("period") else None, 2),
            "yes" if s.get("clipped") else "no",
        ])
    return columns, rows


def spectrum_csv(frames, regions):
    """Energy per band region by region, as the spectrum command reports it."""
    bands = [f"band_{lo:.0f}_{hi:.0f}_hz_db" for lo, hi, _ in frames.band_ranges]
    columns = (
        ["region", "start_s", "end_s", "kind"]
        + bands
        + ["centre_of_gravity_hz", "zero_crossings_per_s"]
    )
    rows = []
    for i, r in enumerate(regions, 1):
        s = r.stats
        row = [
            i,
            _num(frames.absolute(r.start), 3),
            _num(frames.absolute(r.end), 3),
            r.kind,
        ]
        values = list(s.get("bands") or [])
        row += [_num(v, 1) for v in values] + [""] * (len(bands) - len(values))
        row += [_num(s.get("cog"), 1), _num(s.get("zcr"), 1)]
        rows.append(row)
    return columns, rows


def slice_csv(time, peaks, cog):
    """The strongest components at one instant, as slice reports them."""
    columns = ["time_s", "rank", "frequency_hz", "centre_of_gravity_hz"]
    return columns, [
        [_num(time, 3), n, _num(p, 1), _num(cog, 1)]
        for n, p in enumerate(peaks, 1)
    ]


def compare_csv(name_a, name_b, stats_a, stats_b):
    """The two sides of a comparison, measure by measure."""
    columns = ["measure", "unit", "a", "a_value", "b", "b_value", "difference"]
    rows = []
    for key, unit in (
        ("duration", "seconds"),
        ("f0", "hertz"),
        ("f1", "hertz"),
        ("f2", "hertz"),
        ("f3", "hertz"),
        ("intensity", "decibels"),
    ):
        va, vb = stats_a.get(key), stats_b.get(key)
        if va is None or vb is None:
            continue
        rows.append([
            key, unit, name_a, _num(va, 2), name_b, _num(vb, 2),
            _num(abs(va - vb), 2),
        ])
    return columns, rows


# --- assembled reports ------------------------------------------------


def stats_list(frames, regions, start, end):
    """The measurements as a plain list, one per line, nothing interpreted."""
    snd = frames.sound
    values = frames.window_samples()
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    lines = [
        "MEASUREMENTS",
        f"  Duration: {fmt.duration(end - start)}.",
    ]

    v = vot(frames, regions)
    if not v:
        lines.append(f"  Voice onset time: not measurable, {v.reason}.")
    else:
        lines.append(
            f"  Voice onset time: {fmt.ms(v.seconds)}, from the release at "
            f"{fmt.secs(frames.absolute(v.anchor))}, labelled "
            f"{events.PLAIN[v.release.kind]}, to voicing at "
            f"{fmt.secs(frames.absolute(v.onset))}."
        )

    lines += [
        f"  Sampling frequency: {snd.sampling_frequency:.0f} hertz.",
        f"  Channels: {snd.n_channels}.",
        f"  Peak amplitude: {fmt.amp(peak)}.",
    ]

    kinds = {}
    for r in regions:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    parts = [f"{n} {events.SHORT[k]}" for k, n in kinds.items()]
    lines.append(f"  Regions: {len(regions)}, {fmt.join(parts)}.")
    voiced_time = sum(r.duration for r in regions if r.kind == events.VOICED)
    lines.append(
        f"  Voiced: {fmt.duration(voiced_time)}, "
        f"{fmt.pct(voiced_time / max(end - start, 1e-9))} of the stretch."
    )

    f0 = frames.f0[~np.isnan(frames.f0)]
    if f0.size:
        slope = np.polyfit(np.arange(f0.size), f0, 1)[0] * f0.size
        lines += [
            f"  Pitch measurable in: {fmt.pct(f0.size / max(len(frames.f0), 1))} "
            "of frames.",
            f"  Pitch median: {np.median(f0):.0f} hertz.",
            f"  Pitch period, median: {1000 / np.median(f0):.2f} milliseconds, "
            "the time one cycle takes. Divide 1000 by it to get the median "
            "above back in hertz.",
            f"  Pitch mean: {f0.mean():.0f} hertz.",
            f"  Pitch range: {f0.min():.0f} to {f0.max():.0f} hertz.",
            f"  Pitch period range: {1000 / f0.max():.2f} to "
            f"{1000 / f0.min():.2f} milliseconds. A shorter period is a higher "
            "frequency, so this range runs the opposite way round.",
            f"  Pitch net change: {slope:+.0f} hertz.",
        ]
    else:
        lines.append("  Pitch: not measurable anywhere in this stretch.")
    hnr = frames.hnr[~np.isnan(frames.hnr)]
    if hnr.size:
        lines.append(f"  Harmonics-to-noise ratio, mean: {hnr.mean():.1f} decibels.")

    intensity = frames.intensity[~np.isnan(frames.intensity)]
    if intensity.size:
        lines += [
            f"  Level mean: {intensity.mean():.0f} decibels.",
            f"  Level range: {intensity.min():.0f} to {intensity.max():.0f} decibels.",
            f"  Loudness peaks: {_count_peaks(frames.intensity)}, counting rises "
            f"and falls of at least {PEAK_PROMINENCE:.0f} decibels.",
        ]

    voiced = [r for r in regions if r.kind == events.VOICED]
    if voiced:
        longest = max(voiced, key=lambda r: r.duration)
        st = longest.stats
        lines += [
            f"  Longest voiced region: {fmt.secs(frames.absolute(longest.start))} "
            f"to {fmt.secs(frames.absolute(longest.end))}, "
            f"{fmt.duration(longest.duration)}.",
            f"  Its formants: F1 {fmt.hz_plain(st.get('f1'))}, F2 "
            f"{fmt.hz_plain(st.get('f2'))}, F3 {fmt.hz_plain(st.get('f3'))}.",
        ]
        if st.get("jitter") is not None:
            lines.append(f"  Its jitter: {fmt.pct(st['jitter'])}.")
    return lines


def full_report(frames, regions, sound_name, start, end):
    lines = ["ACOUSTIC DESCRIPTION", ""]
    lines += stats_list(frames, regions, start, end)
    lines += [""] + vowel_table(frames, regions)
    lines += [""] + consonant_table(frames, regions)
    lines += ["", "DESCRIPTION", ""]
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


def whole_sound_summary(frames, regions, sound_name, source, texts=(), csvs=()):
    """A short orientation to the entire recording, plus what else was written.

    describe works on a selection, so this is the file that answers "what is in
    this recording at all" - one line per region with the tool's reading of it,
    and an index of the detailed reports.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    start, end = frames.window
    lines = [
        f"WHOLE SOUND SUMMARY: {sound_name}",
        "",
        f"Written by praatshell on {now}.",
        f"Source file: {source or 'created in this session'}.",
        f"Analysis settings: {analysis.SETTINGS.describe()}.",
        "",
        "This file describes the entire recording. The other files listed at the "
        "end describe whichever stretch was selected at the time.",
        "",
    ]
    lines += overview(frames, sound_name, frames.absolute(start), frames.absolute(end))
    lines += ["", f"TIMELINE: {len(regions)} regions across the whole recording.", ""]

    for i, region in enumerate(regions, 1):
        guess = guess_region(region, frames)
        lines.append(
            f"Region {i}: {frames.absolute(region.start):.3f} to "
            f"{frames.absolute(region.end):.3f} seconds, "
            f"{fmt.duration(region.duration)}, {events.PLAIN[region.kind]}."
        )
        s = region.stats
        detail = []
        # Formant tracking returns numbers for silence and noise too, but they
        # describe nothing. Only quote them where they mean something.
        if region.kind == events.VOICED:
            if s.get("f0"):
                detail.append(f"F0 {s['f0']:.0f} hertz")
            if s.get("f1") and s.get("f2"):
                detail.append(f"F1 {s['f1']:.0f} hertz, F2 {s['f2']:.0f} hertz")
        elif region.kind != events.SILENCE and s.get("cog"):
            detail.append(f"energy centred at {s['cog']:.0f} hertz")
        if s.get("intensity_mean"):
            detail.append(f"mean level {s['intensity_mean']:.0f} decibels")
        if detail:
            lines.append(f"  MEASURED: {fmt.join(detail)}.")
        lines.append(f"  LIKELY: {guess.claim}. Confidence: {guess.level}.")

    lines += [""] + contour_summary(frames)

    lines += ["", "OTHER REPORTS FOR THIS SOUND"]
    if texts:
        lines.append("Descriptions, in the reports folder:")
        lines += [f"  {name}" for name in texts]
    if csvs:
        lines.append("Measurements, in the reports csv folder:")
        lines += [f"  {name}" for name in csvs]
    if not texts and not csvs:
        lines.append("None yet. Select a stretch and use describe to make one.")
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
    """Count loudness peaks that stand clear of the dips beside them.

    Measuring a peak against the quietest point anywhere in the recording
    counts every ripple; it has to be measured against the troughs immediately
    around it. This walks the curve and only registers a peak once the level
    has fallen back by the full prominence, then waits for an equal rise before
    looking for the next one.
    """
    prominence = PEAK_PROMINENCE if prominence is None else prominence
    v = intensity[~np.isnan(intensity)]
    if v.size < 3:
        return 0
    # Bracket with the quietest level so a peak at either end still falls away.
    v = np.concatenate(([v.min()], v, [v.min()]))

    count = 0
    climbing = True
    best = trough = v[0]
    for x in v:
        if climbing:
            if x > best:
                best = x
            elif best - x >= prominence:
                count += 1
                climbing = False
                trough = x
        else:
            if x < trough:
                trough = x
            elif x - trough >= prominence:
                climbing = True
                best = x
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
    v = vot(frames, regions)
    if v:
        out.append(
            f"Voice onset time {fmt.ms(v.seconds)}"
            + (", which is too long to trust" if v.implausible else "")
            + "."
        )
    longest = max(regions, key=lambda r: r.duration) if regions else None
    if longest is not None and longest.kind == events.VOICED:
        s = longest.stats
        if s.get("f1") and s.get("f2"):
            name, _, _ = _nearest_vowel(s["f1"], s["f2"])
            out.append(
                f"The longest region is voiced, with F1 {s['f1']:.0f} hertz and "
                f"F2 {s['f2']:.0f} hertz, closest to {name}."
            )
    return out
