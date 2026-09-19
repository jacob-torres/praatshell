"""Playback through the Windows sound API in the standard library."""

import os
import tempfile

import parselmouth
from parselmouth.praat import call

try:
    import winsound
except ImportError:  # not Windows
    winsound = None

_TEMP = os.path.join(tempfile.gettempdir(), "praatshell_play.wav")


def play(sound, speed=1.0):
    """Play a sound. speed below 1 stretches it without changing the pitch."""
    if winsound is None:
        raise RuntimeError("Playback needs Windows.")
    if speed != 1.0:
        sound = _lengthen(sound, 1.0 / speed)
    sound.save(_TEMP, parselmouth.SoundFileFormat.WAV)
    winsound.PlaySound(_TEMP, winsound.SND_FILENAME | winsound.SND_ASYNC)
    return sound.duration


def stop():
    if winsound is not None:
        winsound.PlaySound(None, winsound.SND_PURGE)


def _lengthen(sound, factor):
    """PSOLA time-stretch, so slowed speech keeps its pitch.

    Praat needs enough cycles to find periods; on a very short or unvoiced
    snippet it refuses, and resampling is the acceptable fallback even though
    it drags the pitch down with it.
    """
    try:
        return call(sound, "Lengthen (overlap-add)", 75, 600, factor)
    except Exception:
        return parselmouth.Sound(
            sound.values, sampling_frequency=sound.sampling_frequency / factor
        )
