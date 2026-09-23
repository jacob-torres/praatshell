"""Session state: loaded sounds, the current selection, segments, undo."""

import copy
import os

import parselmouth
from parselmouth.praat import call


class Segment:
    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label

    @property
    def duration(self):
        return self.end - self.start


class SessionError(Exception):
    """Raised with a sentence the user can act on."""


class Session:
    def __init__(self, root):
        self.root = root
        self.sounds = {}
        self.segments = {}
        self.current = None
        self.sel = None
        self.clipboard = None
        self._undo = []

    # --- sounds -------------------------------------------------------

    def load(self, path):
        """Load a .wav, or every Sound in a Praat .Collection. Returns names."""
        if not os.path.exists(path):
            raise SessionError(f"There is no file at {path}.")
        objects = call("Read from file", os.path.abspath(path))
        if not isinstance(objects, list):
            objects = [objects]
        sounds = [o for o in objects if isinstance(o, parselmouth.Sound)]
        if not sounds:
            kinds = ", ".join(sorted({type(o).__name__ for o in objects}))
            raise SessionError(f"{path} holds no sounds. It contains: {kinds}.")
        names = []
        for snd in sounds:
            names.append(self.add(snd.name or os.path.basename(path), snd))
        self.current = names[0]
        self.sel = None
        return names

    def add(self, name, sound):
        name = self._unique(self._clean(name))
        self.sounds[name] = sound
        self.segments[name] = []
        return name

    def _clean(self, name):
        keep = "".join(c if c.isalnum() or c in "-_." else "-" for c in name)
        return keep.strip("-") or "sound"

    def _unique(self, name):
        if name not in self.sounds:
            return name
        n = 2
        while f"{name}-{n}" in self.sounds:
            n += 1
        return f"{name}-{n}"

    def use(self, name):
        if name not in self.sounds:
            raise SessionError(
                f"No sound is called {name}. Type list to hear what is loaded."
            )
        self.current = name
        self.sel = None

    def sound(self, name=None):
        name = name or self.current
        if name is None:
            raise SessionError("No sound is loaded. Use load followed by a filename.")
        if name not in self.sounds:
            raise SessionError(f"No sound is called {name}.")
        return self.sounds[name]

    def replace(self, sound, name=None):
        """Swap in a new version of a sound, saving the old one for undo.

        A selection survives an edit that keeps the same duration, so
        normalizing or shifting pitch does not lose your place. An edit that
        changes the length invalidates the old times, so the selection goes.
        """
        name = name or self.current
        self.push_undo(name)
        same_length = abs(self.sounds[name].duration - sound.duration) < 1e-6
        self.sounds[name] = sound
        if name == self.current and not same_length:
            self.sel = None

    # --- selection ----------------------------------------------------

    def set_selection(self, start, end):
        snd = self.sound()
        if end <= start:
            raise SessionError("The end of a selection must come after its start.")
        if start < 0 or end > snd.duration + 1e-9:
            raise SessionError(
                f"That selection runs outside the sound, which is "
                f"{snd.duration:.3f} seconds long."
            )
        self.sel = (start, min(end, snd.duration))

    def selection(self):
        """(start, end) of the current selection, whole sound if unset."""
        snd = self.sound()
        return self.sel if self.sel else (0.0, snd.duration)

    def selected(self):
        """The current selection as its own Sound."""
        snd = self.sound()
        start, end = self.selection()
        if start <= 0 and end >= snd.duration:
            return snd
        return snd.extract_part(from_time=start, to_time=end, preserve_times=False)

    def selection_label(self):
        start, end = self.selection()
        return f"{start:.3f}-{end:.3f}"

    # --- segments -----------------------------------------------------

    def segs(self, name=None):
        return self.segments.setdefault(name or self.current, [])

    def find_segment(self, label, name=None):
        for seg in self.segs(name):
            if seg.label == label:
                return seg
        raise SessionError(
            f"No segment is labelled {label}. Type seg list to hear the labels."
        )

    def set_segments(self, segments, name=None):
        self.segments[name or self.current] = segments

    # --- what-if ------------------------------------------------------

    def snapshot(self):
        """Everything needed to put the session back as it is now.

        Edits build new Sound objects rather than changing old ones in place,
        so holding on to the old references is enough.
        """
        return (
            dict(self.sounds),
            {k: [copy.copy(s) for s in v] for k, v in self.segments.items()},
            self.current,
            self.sel,
            self.clipboard,
            list(self._undo),
        )

    def restore(self, state):
        (self.sounds, self.segments, self.current, self.sel,
         self.clipboard, self._undo) = state

    # --- undo ---------------------------------------------------------

    def push_undo(self, name=None):
        name = name or self.current
        self._undo.append(
            (name, self.sounds[name], [copy.copy(s) for s in self.segments[name]])
        )
        del self._undo[:-20]

    def undo(self):
        if not self._undo:
            raise SessionError("There is nothing to undo.")
        name, sound, segments = self._undo.pop()
        self.sounds[name] = sound
        self.segments[name] = segments
        self.current = name
        self.sel = None
        return name
