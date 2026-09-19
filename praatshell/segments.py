"""Segment boundaries, and reading and writing real Praat TextGrids."""

import os

import parselmouth
from parselmouth.praat import call

from .session import Segment, SessionError


def from_regions(regions):
    """Turn an automatic region timeline into numbered, editable segments."""
    return [
        Segment(r.start, r.end, str(i)) for i, r in enumerate(regions, 1)
    ]


def split_at(segments, time, label, total_duration):
    """Add a boundary, splitting whichever segment contains it."""
    for seg in segments:
        if seg.start < time < seg.end:
            tail = Segment(time, seg.end, label)
            seg.end = time
            segments.append(tail)
            break
    else:
        start = max(
            [s.end for s in segments if s.end <= time] + [0.0]
        )
        end = min(
            [s.start for s in segments if s.start > time] + [total_duration]
        )
        segments.append(Segment(max(start, time), end, label))
    segments.sort(key=lambda s: s.start)
    return segments


def to_textgrid(sound, segments, path, tier_name="segments"):
    """Write a TextGrid that real Praat can open."""
    grid = call(sound, "To TextGrid", tier_name, "")
    for seg in sorted(segments, key=lambda s: s.start):
        for edge in (seg.start, seg.end):
            if 0 < edge < sound.duration:
                try:
                    call(grid, "Insert boundary", 1, edge)
                except Exception:
                    pass  # a boundary already sits there
    for seg in segments:
        middle = (seg.start + seg.end) / 2
        index = int(call(grid, "Get interval at time", 1, middle))
        call(grid, "Set interval text", 1, index, seg.label)
    call(grid, "Save as text file", os.path.abspath(path))
    return path


def from_textgrid(path, tier=1):
    """Read labelled intervals out of a TextGrid file."""
    if not os.path.exists(path):
        raise SessionError(f"There is no file at {path}.")
    grid = call("Read from file", os.path.abspath(path))
    if isinstance(grid, list):
        grid = next((g for g in grid if type(g).__name__ == "TextGrid"), None)
    if grid is None or type(grid).__name__ != "TextGrid":
        raise SessionError(f"{path} is not a TextGrid.")
    tiers = int(call(grid, "Get number of tiers"))
    if tier > tiers:
        raise SessionError(f"That TextGrid has only {tiers} tiers.")
    count = int(call(grid, "Get number of intervals", tier))
    segments = []
    for i in range(1, count + 1):
        label = call(grid, "Get label of interval", tier, i)
        if not label.strip():
            continue
        segments.append(
            Segment(
                call(grid, "Get start time of interval", tier, i),
                call(grid, "Get end time of interval", tier, i),
                label.strip(),
            )
        )
    return segments


def describe_list(segments):
    lines = []
    for seg in sorted(segments, key=lambda s: s.start):
        lines.append(
            f"{seg.label}: {seg.start:.3f} to {seg.end:.3f} seconds, "
            f"{seg.duration * 1000:.0f} milliseconds."
        )
    return lines or ["No segments are defined. Try autoseg, or seg add."]
