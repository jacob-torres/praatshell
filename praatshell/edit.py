"""Cutting, splicing and resynthesis, all through Praat's own operations."""

from parselmouth.praat import call

from .analysis import SETTINGS
from .session import SessionError


def extract(sound, start, end):
    if end <= start:
        raise SessionError("The end of a stretch must come after its start.")
    return sound.extract_part(from_time=start, to_time=end, preserve_times=False)


def match_rate(clip, target_sound):
    """Praat refuses to join sounds recorded at different sampling rates."""
    if abs(clip.sampling_frequency - target_sound.sampling_frequency) < 1:
        return clip
    return call(clip, "Resample", target_sound.sampling_frequency, 50)


def join(pieces):
    pieces = [p for p in pieces if p is not None and p.duration > 0]
    if not pieces:
        raise SessionError("There is nothing left to join.")
    if len(pieces) == 1:
        return pieces[0]
    return call(pieces, "Concatenate")


def replace_range(sound, start, end, clip):
    """Put clip in place of start..end. Pass clip=None to delete the stretch."""
    head = extract(sound, 0, start) if start > 0 else None
    tail = extract(sound, end, sound.duration) if end < sound.duration else None
    middle = match_rate(clip, sound) if clip is not None else None
    return join([p for p in (head, middle, tail) if p is not None])


def insert_at(sound, time, clip):
    clip = match_rate(clip, sound)
    if time <= 0:
        return join([clip, sound])
    if time >= sound.duration:
        return join([sound, clip])
    return join([extract(sound, 0, time), clip, extract(sound, time, sound.duration)])


def swap(sound_a, span_a, sound_b, span_b):
    """Exchange one stretch of two sounds. Returns the two new sounds."""
    clip_a = extract(sound_a, *span_a)
    clip_b = extract(sound_b, *span_b)
    return (
        replace_range(sound_a, span_a[0], span_a[1], clip_b),
        replace_range(sound_b, span_b[0], span_b[1], clip_a),
    )


def reverse(sound):
    copy = sound.copy()
    call(copy, "Reverse")
    return copy


def normalize(sound, peak=0.99):
    copy = sound.copy()
    call(copy, "Scale peak", peak)
    return copy


def _manipulation(sound):
    return call(
        sound, "To Manipulation", 0.01, SETTINGS.pitch_floor, SETTINGS.pitch_ceiling
    )


def stretch(sound, factor):
    """Change duration without changing pitch."""
    if factor <= 0:
        raise SessionError("The stretch factor must be greater than zero.")
    try:
        return call(
            sound,
            "Lengthen (overlap-add)",
            SETTINGS.pitch_floor,
            SETTINGS.pitch_ceiling,
            factor,
        )
    except Exception as exc:
        raise SessionError(
            "Praat could not stretch this sound. It needs enough voiced signal to "
            f"find pitch periods. Praat said: {exc}"
        )


def pitch_shift(sound, semitones):
    """Change pitch without changing duration."""
    factor = 2 ** (semitones / 12.0)
    try:
        man = _manipulation(sound)
        tier = call(man, "Extract pitch tier")
        call(tier, "Multiply frequencies", 0, sound.duration, factor)
        call([man, tier], "Replace pitch tier")
        return call(man, "Get resynthesis (overlap-add)")
    except Exception as exc:
        raise SessionError(f"Praat could not shift the pitch here. It said: {exc}")


def flatten(sound, hertz=None):
    """Replace the pitch contour with a monotone."""
    try:
        man = _manipulation(sound)
        if hertz is None:
            pitch = sound.to_pitch(
                pitch_floor=SETTINGS.pitch_floor, pitch_ceiling=SETTINGS.pitch_ceiling
            )
            # Median, not mean: one octave-error frame drags a mean badly.
            hertz = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
        tier = call("Create PitchTier", "flat", 0, sound.duration)
        call(tier, "Add point", 0.0, hertz)
        call(tier, "Add point", sound.duration, hertz)
        call([man, tier], "Replace pitch tier")
        return call(man, "Get resynthesis (overlap-add)"), hertz
    except Exception as exc:
        raise SessionError(f"Praat could not flatten the pitch here. It said: {exc}")
