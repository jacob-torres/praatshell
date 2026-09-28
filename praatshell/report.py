"""Writes the prose report and the matching CSV measurements."""

import csv
import datetime
import os

import numpy as np

from . import analysis


def _stem(sound_name, start, end):
    """Times in whole milliseconds, so the filename holds no extra full stops."""
    return f"{sound_name}__{start * 1000:.0f}-{end * 1000:.0f}ms"


def _dir(root):
    path = os.path.join(root, "reports")
    os.makedirs(path, exist_ok=True)
    return path


def _csv_dir(root):
    """Numbers live in reports/csv, so the reports folder itself stays readable."""
    path = os.path.join(_dir(root), "csv")
    os.makedirs(path, exist_ok=True)
    return path


def header(sound_name, source, start, end):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return [
        f"Written by praatshell on {now}.",
        f"Sound: {sound_name}.",
        f"Source file: {source or 'created in this session'}.",
        f"Stretch: {start:.3f} to {end:.3f} seconds.",
        f"Analysis settings: {analysis.SETTINGS.describe()}.",
        f"Frame spacing: {analysis.TIME_STEP * 1000:.0f} milliseconds.",
        "",
    ]


def write(root, sound_name, source, start, end, lines, frames=None, suffix="",
          preview="", layers=None, stem=None, tables=None):
    """Write the .txt, plus a .csv per analysis layer. Returns the paths.

    suffix names a single-layer report, so `pitch` writes spkr1__0-1204ms_pitch.txt
    beside its spkr1__0-1204ms_pitch.csv.

    preview names the edit a report describes rather than the report itself, so
    `pitch reverse` writes spkr1__0-1204ms_pitch_reverse.txt beside
    spkr1__0-1204ms_reverse_pitch.csv. It has to stay apart from suffix: the
    preview belongs to the numbers, and folding it into the report's name would
    let a preview overwrite the real sound's CSV.

    tables is a list of (name, columns, rows) for numbers that form a table
    rather than one value per frame, as the vowel and consonant tables do. A
    name makes that table a layer alongside the frame CSVs, so a description
    writes its vowels and consonants beside its pitch and formants; a name of
    None makes it the report's own CSV, taking the report's own filename.
    """
    folder = _dir(root)
    base = stem or _stem(sound_name, start, end)
    written = []

    report_name = "_".join([base] + [p for p in (suffix, preview) if p])
    txt = os.path.join(folder, report_name + ".txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header(sound_name, source, start, end) + lines) + "\n")
    written.append(txt)

    # A preview's numbers are kept apart from the real sound's, by name:
    # pat__0-1204ms_reverse_pitch.csv.
    csv_base = f"{base}_{preview}" if preview else base
    if frames is not None:
        written += _write_csvs(_csv_dir(root), csv_base, frames, layers)
    for table_name, columns, rows in tables or ():
        # Named, a table is one more layer beside the frame CSVs. Unnamed, it is
        # the whole of the report's numbers and takes the report's own name.
        # Every CSV is named the same way round, whether it came from a table
        # or from the frames: the base, then the preview, then what was
        # measured. So *_pitch.csv and *_vowels.csv both glob as you expect.
        name = f"{csv_base}_{table_name or suffix}.csv" if (table_name or suffix) else f"{report_name}.csv"
        written.append(_csv(_csv_dir(root), name, columns, rows))
    return written


def write_summary(root, sound_name, lines):
    """The whole-sound summary, named so it sorts ahead of the timed reports."""
    path = os.path.join(_dir(root), f"{sound_name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def related_files(root, sound_name):
    """Every report already written for this sound, for the summary's index."""
    folder = _dir(root)
    prefix = f"{sound_name}__"
    texts = sorted(
        f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".txt")
    )
    csv_folder = _csv_dir(root)
    csvs = sorted(
        f for f in os.listdir(csv_folder) if f.startswith(prefix) and f.endswith(".csv")
    )
    return texts, csvs


def _write_csvs(folder, stem, frames, layers=None):
    written = []
    times = [frames.absolute(t) for t in frames.times]
    wanted = layers or ["pitch", "formants", "intensity", "bands"]

    if "pitch" in wanted:
        written.append(
            _csv(
                folder,
                stem + "_pitch.csv",
                ["time_s", "f0_hz", "period_ms", "f0_raw_hz", "hnr_db", "voiced"],
                zip(
                    times,
                    _col(frames.f0),
                    _periods_ms(frames.f0),
                    _col(frames.f0_raw if frames.f0_raw.size else frames.f0),
                    _col(frames.hnr),
                    ["yes" if v else "no" for v in frames.voiced],
                ),
            )
        )
    if "formants" in wanted:
        written.append(
            _csv(
                folder,
                stem + "_formants.csv",
                ["time_s", "f1_hz", "f2_hz", "f3_hz"],
                ([times[i]] + _col(frames.formants[i]) for i in range(len(times))),
            )
        )
    if "intensity" in wanted:
        written.append(
            _csv(
                folder,
                stem + "_intensity.csv",
                ["time_s", "intensity_db"],
                zip(times, _col(frames.intensity)),
            )
        )
    if "bands" in wanted and frames.bands.size:
        names = [f"band_{lo:.0f}_{hi:.0f}_hz_db" for lo, hi, _ in frames.band_ranges]
        written.append(
            _csv(
                folder,
                stem + "_bands.csv",
                ["time_s"] + names + ["centre_of_gravity_hz", "zero_crossings_per_s"],
                (
                    [times[i]]
                    + _col(frames.bands[i])
                    + [_val(frames.cog[i]), _val(frames.zcr[i])]
                    for i in range(len(times))
                ),
            )
        )
    return written


def _csv(folder, name, columns, rows):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [f"{v:.4f}" if isinstance(v, float) else v for v in row]
            )
    return path


def _col(array):
    return [_val(v) for v in np.asarray(array).ravel()]


def _periods_ms(f0):
    """How long one cycle takes, beside the frequency it is the reciprocal of.

    Empty wherever no pitch was found, so the column lines up with f0_hz.
    """
    return [
        "" if np.isnan(v) or v <= 0 else 1000.0 / float(v)
        for v in np.asarray(f0).ravel()
    ]


def _val(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
