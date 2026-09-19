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


def write(root, sound_name, source, start, end, lines, frames=None, suffix="", layers=None, stem=None):
    """Write the .txt, plus a .csv per analysis layer. Returns the paths.

    suffix names a single-layer report, so `pitch` writes spkr1__0-1204ms_pitch.txt
    beside its spkr1__0-1204ms_pitch.csv.
    """
    folder = _dir(root)
    base = stem or _stem(sound_name, start, end)
    written = []

    txt = os.path.join(folder, f"{base}_{suffix}.txt" if suffix else base + ".txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header(sound_name, source, start, end) + lines) + "\n")
    written.append(txt)

    if frames is not None:
        written += _write_csvs(folder, base, frames, layers)
    return written


def _write_csvs(folder, stem, frames, layers=None):
    written = []
    times = [frames.absolute(t) for t in frames.times]
    wanted = layers or ["pitch", "formants", "intensity", "bands"]

    if "pitch" in wanted:
        written.append(
            _csv(
                folder,
                stem + "_pitch.csv",
                ["time_s", "f0_hz", "f0_raw_hz", "hnr_db", "voiced"],
                zip(
                    times,
                    _col(frames.f0),
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


def _val(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
