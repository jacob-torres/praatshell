"""The interactive command shell.

Output rules, because this is read aloud by a screen reader: one fact per line,
no tables or drawn characters, units spelled out, and at most a few sentences
per reply. Detail goes to a file instead.
"""

import cmd
import os

import numpy as np
import parselmouth

from . import analysis, audio, describe, edit, events, fmt, praatbridge, report, segments
from .session import Session, SessionError

HELP = [
    ("load FILE", "Load a .wav, or every sound inside a Praat .Collection."),
    ("list", "Say which sounds are loaded."),
    ("use NAME", "Make a different loaded sound the current one."),
    ("info", "Duration, sampling frequency, channels and peak level."),
    ("save NAME FILE", "Write a loaded sound to a .wav file."),
    ("undo", "Undo the last change to a sound."),
    ("", ""),
    ("select START END", "Work on part of the sound. Times in seconds, or like 120ms."),
    ("select all", "Go back to working on the whole sound."),
    ("select seg LABEL", "Work on a labelled segment."),
    ("play", "Play the current selection."),
    ("play slow", "Play it at half speed, keeping the pitch."),
    ("stop", "Stop playing."),
    ("", ""),
    ("describe", "Full description of the selection: regions, pitch, formants, shape."),
    ("events", "Just the timeline of regions."),
    ("wave", "The waveform's physical shape: amplitude, periodicity, envelope."),
    ("pitch", "Fundamental frequency over the selection."),
    ("formants", "F1, F2 and F3 over the selection."),
    ("intensity", "Loudness over the selection."),
    ("spectrum", "Where energy sits in frequency."),
    ("slice TIME", "The strongest frequencies at one instant."),
    ("compare A B", "Compare two segments, or two loaded sounds."),
    ("", ""),
    ("autoseg", "Propose segment boundaries from the region timeline."),
    ("seg list", "Say the segments and their times."),
    ("seg add TIME LABEL", "Add a boundary at a time and name what follows."),
    ("seg del LABEL", "Remove a segment."),
    ("seg rename OLD NEW", "Rename a segment."),
    ("seg save FILE", "Write the segments as a Praat TextGrid."),
    ("seg load FILE", "Read segments from a Praat TextGrid."),
    ("", ""),
    ("cut LABEL", "Remove a segment from the sound and hold it on the clipboard."),
    ("copy [LABEL]", "Copy the selection, or a segment, to the clipboard."),
    ("paste TIME", "Insert the clipboard at a time."),
    ("swap A B", "Exchange two segments. Use sound:label to cross between sounds."),
    ("splice TARGET SOURCE", "Replace the target segment with the source segment."),
    ("stretch FACTOR", "Change duration without changing pitch. 2 is twice as long."),
    ("pitchshift SEMITONES", "Raise or lower pitch without changing duration."),
    ("flatten [HERTZ]", "Replace the pitch contour with a monotone."),
    ("reverse", "Reverse the sound."),
    ("normalize", "Scale the peak up to just under full scale."),
    ("concat A B", "Join loaded sounds end to end into a new sound."),
    ("", ""),
    ("set NAME VALUE", "Change pitchfloor, pitchceiling or formantceiling."),
    ("settings", "Say the current analysis settings."),
    ("praat gui", "Open the current sound in the real Praat program."),
    ("praat script FILE", "Run a .praat script and read back what it printed."),
    ("quit", "Leave."),
]


class Shell(cmd.Cmd):
    intro = (
        "praatshell. Type commands to analyse speech. "
        "Type help for the command list, or load followed by a filename to start."
    )

    def __init__(self, root):
        super().__init__()
        self.session = Session(root)
        self.root = root
        self.sources = {}
        self._cache = None
        self._set_prompt()

    # --- plumbing -----------------------------------------------------

    def say(self, *lines):
        for line in lines:
            print(line, flush=True)

    def _set_prompt(self):
        name = self.session.current
        self.prompt = f"{name}> " if name else "praatshell> "

    def postcmd(self, stop, line):
        self._set_prompt()
        return stop

    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except SessionError as exc:
            self.say(str(exc))
        except Exception as exc:  # a Praat refusal should not end the session
            self.say(f"That did not work: {exc}")

    def emptyline(self):
        pass

    def default(self, line):
        self.say(f"There is no command called {line.split()[0]}. Type help for the list.")

    def do_help(self, arg):
        """Say the command list, one command per line."""
        if arg:
            return super().do_help(arg)
        self.say("Commands, one per line.")
        for name, text in HELP:
            self.say(f"{name}: {text}" if name else "")

    def do_quit(self, arg):
        """Leave praatshell."""
        self.say("Goodbye.")
        return True

    do_exit = do_quit
    do_EOF = do_quit

    # --- analysis cache -----------------------------------------------

    def analysed(self):
        """Frames and regions for the current selection, computed once."""
        name = self.session.current
        start, end = self.session.selection()
        key = (name, start, end, id(self.session.sound()), vars(analysis.SETTINGS).copy())
        if self._cache and self._cache[0] == key:
            return self._cache[1], self._cache[2]
        frames = analysis.analyze_window(self.session.sound(), start, end)
        regions = events.classify(frames)
        self._cache = (key, frames, regions)
        return frames, regions

    def invalidate(self):
        self._cache = None

    def source_of(self, name):
        return self.sources.get(name, "")

    def write_report(self, lines, suffix="", layers=None, frames=None):
        name = self.session.current
        start, end = self.session.selection()
        paths = report.write(
            self.root,
            name,
            self.source_of(name),
            start,
            end,
            lines,
            frames=frames,
            suffix=suffix,
            layers=layers,
        )
        rel = [os.path.relpath(p, self.root) for p in paths]
        self.say(f"Written: {fmt.join(rel)}.")

    # --- loading ------------------------------------------------------

    def do_load(self, arg):
        """load FILE. A .wav, or every sound inside a Praat .Collection."""
        if not arg.strip():
            raise SessionError("Say which file to load, for example load pat.wav.")
        path = arg.strip().strip('"')
        names = self.session.load(path)
        for name in names:
            self.sources[name] = path
        self.invalidate()
        snd = self.session.sound()
        if len(names) == 1:
            self.say(
                f"Loaded {names[0]}. {snd.duration:.3f} seconds, "
                f"{snd.sampling_frequency:.0f} hertz, "
                f"{snd.n_channels} channel{'s' if snd.n_channels > 1 else ''}."
            )
        else:
            self.say(
                f"Loaded {len(names)} sounds from {os.path.basename(path)}.",
                f"The current one is {names[0]}.",
                "Type list to hear all of their names.",
            )

    def do_list(self, arg):
        """list. Say which sounds are loaded."""
        if not self.session.sounds:
            self.say("Nothing is loaded yet.")
            return
        for name, snd in self.session.sounds.items():
            marker = " (current)" if name == self.session.current else ""
            self.say(f"{name}: {snd.duration:.3f} seconds{marker}.")

    def do_use(self, arg):
        """use NAME. Make a different loaded sound the current one."""
        self.session.use(arg.strip())
        self.invalidate()
        self.say(f"Now working on {self.session.current}.")

    def do_info(self, arg):
        """info. Duration, sampling frequency, channels and peak level."""
        snd = self.session.sound()
        start, end = self.session.selection()
        values = snd.values[0] if snd.values.ndim > 1 else snd.values
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        self.say(
            f"{self.session.current}: {snd.duration:.3f} seconds, "
            f"{snd.sampling_frequency:.0f} hertz, {snd.n_channels} channel"
            f"{'s' if snd.n_channels > 1 else ''}, "
            f"{snd.n_samples} samples.",
            f"Peak amplitude {fmt.amp(peak)}. Nothing above "
            f"{snd.sampling_frequency / 2:.0f} hertz is recorded.",
            f"Current selection: {start:.3f} to {end:.3f} seconds.",
        )
        if self.session.segs():
            self.say(f"{len(self.session.segs())} segments are defined.")

    def do_save(self, arg):
        """save NAME FILE. Write a loaded sound to a .wav file."""
        parts = arg.split()
        if len(parts) != 2:
            raise SessionError("Say save, then the sound's name, then the filename.")
        name, path = parts
        snd = self.session.sound(name)
        snd.save(os.path.abspath(path), parselmouth.SoundFileFormat.WAV)
        self.say(f"Wrote {name} to {path}.")

    def do_undo(self, arg):
        """undo. Undo the last change to a sound."""
        name = self.session.undo()
        self.invalidate()
        self.say(f"Undone. {name} is back to its previous state.")

    # --- selection and playback ---------------------------------------

    def do_select(self, arg):
        """select START END, select all, or select seg LABEL."""
        parts = arg.split()
        if not parts:
            start, end = self.session.selection()
            self.say(f"The selection is {start:.3f} to {end:.3f} seconds.")
            return
        if parts[0] == "all":
            self.session.sel = None
            self.invalidate()
            self.say(f"Selected the whole sound, {self.session.sound().duration:.3f} seconds.")
            return
        if parts[0] == "seg":
            if len(parts) < 2:
                raise SessionError("Say select seg, then the segment's label.")
            seg = self.session.find_segment(parts[1])
            self.session.set_selection(seg.start, seg.end)
        else:
            if len(parts) < 2:
                raise SessionError("Say select, then a start time and an end time.")
            self.session.set_selection(fmt.parse_time(parts[0]), fmt.parse_time(parts[1]))
        self.invalidate()
        start, end = self.session.selection()
        self.say(
            f"Selected {start:.3f} to {end:.3f} seconds, "
            f"{fmt.duration(end - start)} of sound."
        )

    def do_play(self, arg):
        """play, play slow, or play seg LABEL."""
        parts = arg.split()
        speed = 1.0
        if parts and parts[0] == "slow":
            speed = 0.5
            parts = parts[1:]
        if parts and parts[0] == "seg":
            if len(parts) < 2:
                raise SessionError("Say play seg, then the segment's label.")
            seg = self.session.find_segment(parts[1])
            self.session.set_selection(seg.start, seg.end)
            self.invalidate()
        sound = self.session.selected()
        played = audio.play(sound, speed)
        how = " at half speed" if speed != 1.0 else ""
        self.say(f"Playing {fmt.duration(played)}{how}.")

    def do_stop(self, arg):
        """stop. Stop playing."""
        audio.stop()
        self.say("Stopped.")

    # --- description --------------------------------------------------

    def do_describe(self, arg):
        """describe. Full description of the selection."""
        frames, regions = self.analysed()
        start, end = self.session.selection()
        lines = describe.full_report(
            frames, regions, self.session.current, start, end
        )
        self.say(*describe.summary(frames, regions, start, end))
        self.write_report(lines, frames=frames)

    def do_events(self, arg):
        """events. The timeline of regions, without the detail."""
        frames, regions = self.analysed()
        lines = [f"The selection divides into {len(regions)} regions."]
        for i, r in enumerate(regions, 1):
            lines.append(
                f"Region {i}: {frames.absolute(r.start):.3f} to "
                f"{frames.absolute(r.end):.3f} seconds, "
                f"{fmt.duration(r.duration)}, {events.PLAIN[r.kind]}."
            )
        self.say(*lines)
        self.write_report(lines, suffix="events", layers=[])

    def do_wave(self, arg):
        """wave. The waveform's physical shape."""
        frames, regions = self.analysed()
        start, end = self.session.selection()
        lines = describe.overview(frames, self.session.current, start, end)
        lines.append("")
        for i, r in enumerate(regions, 1):
            s = r.stats
            if r.kind == events.SILENCE:
                continue
            lines.append(
                f"Region {i}, {fmt.duration(r.duration)}, {events.PLAIN[r.kind]}:"
            )
            lines += describe._amplitude_lines(s)
            if s.get("cycles"):
                lines.append(
                    f"  About {s['cycles']:.0f} cycles at a mean period of "
                    f"{s.get('period', 0) * 1000:.1f} milliseconds."
                )

        loud = [r for r in regions if r.kind != events.SILENCE]
        if not loud:
            self.say("This stretch is silent throughout.")
        else:
            biggest = max(loud, key=lambda r: r.stats.get("peak_amp") or 0)
            s = biggest.stats
            spoken = [
                f"Loudest point: {fmt.amp(s.get('peak_amp', 0))} at "
                f"{fmt.secs(frames.absolute(s.get('peak_time', biggest.start)))}, "
                f"in {fmt.article(events.PLAIN[biggest.kind])} region."
            ]
            if s.get("cycles"):
                spoken.append(
                    f"That region holds about {s['cycles']:.0f} cycles at a mean "
                    f"period of {s['period'] * 1000:.1f} milliseconds."
                )
            if "attack" in s:
                spoken.append(f"Its envelope {describe.envelope_phrase(s)}.")
            self.say(*spoken)
        self.write_report(lines, suffix="wave", layers=["intensity"], frames=frames)

    def do_pitch(self, arg):
        """pitch. Fundamental frequency over the selection."""
        frames, _ = self.analysed()
        f0 = frames.f0[~np.isnan(frames.f0)]
        if not f0.size:
            self.say("No pitch could be measured here. The sound may be voiceless.")
            return
        lines = [
            f"Pitch measured in {fmt.pct(f0.size / len(frames.f0))} of frames.",
            f"Mean {f0.mean():.1f} hertz. Range {f0.min():.1f} to {f0.max():.1f} hertz.",
            f"Standard deviation {f0.std():.1f} hertz.",
            f"The contour is {describe._overall_shape(f0)}.",
            "",
            "Frame by frame, time in seconds then fundamental frequency in hertz:",
        ]
        for t, v in zip(frames.times, frames.f0):
            if not np.isnan(v):
                lines.append(f"  {frames.absolute(t):.3f}: {v:.1f}")
        self.say(*lines[:4])
        self.write_report(lines, suffix="pitch", layers=["pitch"], frames=frames)

    def do_formants(self, arg):
        """formants. F1, F2 and F3 over the selection."""
        frames, regions = self.analysed()
        lines = []
        voiced = [r for r in regions if r.kind == events.VOICED]
        if not voiced:
            self.say("No voiced region was found, so formants would not be meaningful.")
            return
        spoken = []
        for i, r in enumerate(voiced, 1):
            s = r.stats
            f1, f2, f3 = s.get("f1"), s.get("f2"), s.get("f3")
            head = (
                f"Voiced region {i}, {frames.absolute(r.start):.3f} to "
                f"{frames.absolute(r.end):.3f} seconds:"
            )
            body = (
                f"  F1 {fmt.hz_plain(f1)}, F2 {fmt.hz_plain(f2)}, F3 {fmt.hz_plain(f3)}."
            )
            lines += [head, body]
            if f1 and f2:
                name, runner, ratio = describe._nearest_vowel(f1, f2)
                lines.append(
                    f"  Closest reference vowel: {name}. Next closest: {runner}."
                )
                spoken.append(f"Region {i}: F1 {f1:.0f}, F2 {f2:.0f}, closest to {name}.")
            if s.get("f2_spread"):
                lines.append(
                    f"  F2 varies by {s['f2_spread']:.0f} hertz across the region."
                )
        lines.append("")
        lines.append("Frame by frame, time then F1, F2, F3 in hertz:")
        for i, t in enumerate(frames.times):
            row = frames.formants[i]
            if not np.isnan(row).all():
                lines.append(
                    f"  {frames.absolute(t):.3f}: "
                    + ", ".join("" if np.isnan(v) else f"{v:.0f}" for v in row)
                )
        self.say(*spoken[:3])
        self.write_report(lines, suffix="formants", layers=["formants"], frames=frames)

    def do_intensity(self, arg):
        """intensity. Loudness over the selection."""
        frames, _ = self.analysed()
        db = frames.intensity[~np.isnan(frames.intensity)]
        if not db.size:
            self.say("Intensity could not be measured here.")
            return
        peaks = describe._count_peaks(frames.intensity)
        lines = [
            f"Mean level {db.mean():.1f} decibels.",
            f"Range {db.min():.1f} to {db.max():.1f} decibels.",
            f"The loudness curve has {peaks} clear peak{'s' if peaks != 1 else ''}, "
            "which often matches the number of syllables.",
            "",
            "Frame by frame, time in seconds then level in decibels:",
        ]
        for t, v in zip(frames.times, frames.intensity):
            if not np.isnan(v):
                lines.append(f"  {frames.absolute(t):.3f}: {v:.1f}")
        self.say(*lines[:3])
        self.write_report(lines, suffix="intensity", layers=["intensity"], frames=frames)

    def do_spectrum(self, arg):
        """spectrum. Where energy sits in frequency."""
        frames, regions = self.analysed()
        lines = [
            "Energy by frequency band. Bands stop at "
            f"{frames.sound.sampling_frequency / 2:.0f} hertz, the highest "
            "frequency this recording carries.",
            "",
        ]
        spoken = []
        for i, r in enumerate(regions, 1):
            if r.kind == events.SILENCE:
                continue
            lines.append(
                f"Region {i}, {frames.absolute(r.start):.3f} to "
                f"{frames.absolute(r.end):.3f} seconds, {events.PLAIN[r.kind]}:"
            )
            band_lines = describe._band_lines(r.stats, frames)
            lines += band_lines
            if r.stats.get("cog"):
                lines.append(
                    f"  Centre of gravity {fmt.hz_plain(r.stats['cog'])}."
                )
            if band_lines and not spoken:
                spoken.append(f"Region {i}: {band_lines[0].strip()}")
        self.say(
            f"Described energy in {len(frames.band_names)} frequency bands across "
            f"{len(regions)} regions.",
            *spoken,
        )
        self.write_report(lines, suffix="spectrum", layers=["bands"], frames=frames)

    def do_slice(self, arg):
        """slice TIME. The strongest frequencies at one instant."""
        if not arg.strip():
            raise SessionError("Say slice, then a time, for example slice 0.25.")
        time = fmt.parse_time(arg.strip())
        snd = self.session.sound()
        if not 0 <= time <= snd.duration:
            raise SessionError(
                f"That time is outside the sound, which is {snd.duration:.3f} seconds."
            )
        peaks, cog, nyquist = analysis.spectral_peaks(snd, time)
        lines = [
            f"Spectral slice at {fmt.secs(time)}, measured over a "
            f"{analysis.WINDOW * 1000:.0f} millisecond window.",
            f"Strongest frequency components: "
            f"{fmt.join(f'{p:.0f} hertz' for p in peaks)}.",
            f"Centre of gravity {fmt.hz_plain(cog)}.",
            f"The recording carries nothing above {nyquist:.0f} hertz.",
        ]
        self.say(*lines)
        self.write_report(lines, suffix=f"slice{time * 1000:.0f}ms", layers=[])

    def do_compare(self, arg):
        """compare A B. Compare two segments, or two loaded sounds."""
        parts = arg.split()
        if len(parts) != 2:
            raise SessionError("Say compare, then two segment labels or two sound names.")
        a, b = (self._resolve(p) for p in parts)
        lines = ["COMPARISON", ""]
        spoken = []
        for label, (name, frames, regions) in (("A", a), ("B", b)):
            lines.append(f"{label}: {name}")
            lines += self._compare_block(frames, regions)
            lines.append("")
        stats = [self._headline(f, r) for _, f, r in (a, b)]
        for key, unit in (("f0", "hertz"), ("f1", "hertz"), ("f2", "hertz"),
                          ("duration", "seconds"), ("intensity", "decibels")):
            va, vb = stats[0].get(key), stats[1].get(key)
            if va and vb:
                line = (
                    f"{key.upper()}: {a[0]} {va:.0f} {unit}, {b[0]} {vb:.0f} {unit}, "
                    f"a difference of {abs(va - vb):.0f} {unit}."
                )
                lines.append(line)
                if key in ("f1", "f2"):
                    spoken.append(line)
        self.say(f"Compared {a[0]} with {b[0]}.", *spoken)
        name = self.session.current
        stem = f"{parts[0]}-vs-{parts[1]}".replace(":", "-")
        paths = report.write(
            self.root,
            name,
            self.source_of(name),
            0,
            0,
            lines,
            suffix="compare",
            stem=stem,
        )
        self.say(f"Written: {os.path.relpath(paths[0], self.root)}.")

    def _resolve(self, token):
        """A token is a segment label in the current sound, or a sound name."""
        if token in self.session.sounds:
            sound = self.session.sound(token)
            frames = analysis.analyze(sound)
            return token, frames, events.classify(frames)
        seg = self.session.find_segment(token)
        frames = analysis.analyze_window(self.session.sound(), seg.start, seg.end)
        return f"segment {token}", frames, events.classify(frames)

    def _compare_block(self, frames, regions):
        lines = [f"  Duration {fmt.duration(frames.span)}."]
        f0 = frames.f0[~np.isnan(frames.f0)]
        if f0.size:
            lines.append(f"  Mean pitch {f0.mean():.0f} hertz.")
        voiced = [r for r in regions if r.kind == events.VOICED]
        if voiced:
            longest = max(voiced, key=lambda r: r.duration)
            s = longest.stats
            if s.get("f1") and s.get("f2"):
                name, runner, _ = describe._nearest_vowel(s["f1"], s["f2"])
                lines.append(
                    f"  Longest voiced region: F1 {s['f1']:.0f} hertz, F2 "
                    f"{s['f2']:.0f} hertz, closest to {name}, then {runner}."
                )
        db = frames.intensity[~np.isnan(frames.intensity)]
        if db.size:
            lines.append(f"  Mean level {db.mean():.0f} decibels.")
        return lines

    def _headline(self, frames, regions):
        out = {"duration": frames.span}
        f0 = frames.f0[~np.isnan(frames.f0)]
        if f0.size:
            out["f0"] = float(f0.mean())
        db = frames.intensity[~np.isnan(frames.intensity)]
        if db.size:
            out["intensity"] = float(db.mean())
        voiced = [r for r in regions if r.kind == events.VOICED]
        if voiced:
            s = max(voiced, key=lambda r: r.duration).stats
            for k in ("f1", "f2", "f3"):
                if s.get(k):
                    out[k] = s[k]
        return out

    # --- segments -----------------------------------------------------

    def do_autoseg(self, arg):
        """autoseg. Propose segment boundaries from the region timeline."""
        frames, regions = self.analysed()
        segs = segments.from_regions(regions)
        for seg, region in zip(segs, regions):
            seg.start = frames.absolute(seg.start)
            seg.end = frames.absolute(seg.end)
        self.session.set_segments(segs)
        self.say(
            f"Created {len(segs)} segments, numbered 1 to {len(segs)}.",
            "Type seg list to hear them, or select seg 1 to work on the first.",
        )

    def do_seg(self, arg):
        """seg list, seg add TIME LABEL, seg del LABEL, seg rename OLD NEW,
        seg save FILE, seg load FILE."""
        parts = arg.split()
        if not parts:
            raise SessionError("Say seg, then list, add, del, rename, save or load.")
        action = parts[0]
        segs = self.session.segs()

        if action == "list":
            self.say(*segments.describe_list(segs))
        elif action == "add":
            if len(parts) < 3:
                raise SessionError("Say seg add, then a time, then a label.")
            time = fmt.parse_time(parts[1])
            segments.split_at(segs, time, parts[2], self.session.sound().duration)
            self.say(f"Added a boundary at {fmt.secs(time)}, labelled {parts[2]}.")
        elif action == "del":
            seg = self.session.find_segment(parts[1])
            segs.remove(seg)
            self.say(f"Removed segment {parts[1]}.")
        elif action == "rename":
            if len(parts) < 3:
                raise SessionError("Say seg rename, then the old label and the new one.")
            seg = self.session.find_segment(parts[1])
            seg.label = parts[2]
            self.say(f"Renamed {parts[1]} to {parts[2]}.")
        elif action == "save":
            if len(parts) < 2:
                raise SessionError("Say seg save, then a filename ending in .TextGrid.")
            path = segments.to_textgrid(self.session.sound(), segs, parts[1])
            self.say(f"Wrote {len(segs)} segments to {path}. Real Praat can open it.")
        elif action == "load":
            if len(parts) < 2:
                raise SessionError("Say seg load, then a TextGrid filename.")
            loaded = segments.from_textgrid(parts[1])
            self.session.set_segments(loaded)
            self.say(f"Loaded {len(loaded)} labelled segments from {parts[1]}.")
        else:
            raise SessionError(f"seg has no {action} action.")

    # --- editing ------------------------------------------------------

    def _span(self, token):
        """A token is sound:label, a label, or nothing for the selection."""
        if ":" in token:
            name, label = token.split(":", 1)
            self.session.sound(name)  # raises if unknown
            seg = self.session.find_segment(label, name)
            return name, (seg.start, seg.end)
        seg = self.session.find_segment(token)
        return self.session.current, (seg.start, seg.end)

    def do_cut(self, arg):
        """cut LABEL. Remove a segment and hold it on the clipboard."""
        name, (start, end) = self._span(arg.strip())
        sound = self.session.sound(name)
        self.session.clipboard = edit.extract(sound, start, end)
        self.session.replace(edit.replace_range(sound, start, end, None), name)
        self.invalidate()
        self.say(
            f"Cut {fmt.duration(end - start)} out of {name}. "
            f"It is on the clipboard, and {name} is now "
            f"{self.session.sound(name).duration:.3f} seconds."
        )

    def do_copy(self, arg):
        """copy [LABEL]. Copy the selection, or a segment, to the clipboard."""
        if arg.strip():
            _, (start, end) = self._span(arg.strip())
            self.session.clipboard = edit.extract(self.session.sound(), start, end)
        else:
            self.session.clipboard = self.session.selected()
        self.say(
            f"Copied {fmt.duration(self.session.clipboard.duration)} to the clipboard."
        )

    def do_paste(self, arg):
        """paste TIME. Insert the clipboard at a time."""
        if self.session.clipboard is None:
            raise SessionError("The clipboard is empty. Use copy or cut first.")
        if not arg.strip():
            raise SessionError("Say paste, then a time in seconds.")
        time = fmt.parse_time(arg.strip())
        sound = self.session.sound()
        self.session.replace(edit.insert_at(sound, time, self.session.clipboard))
        self.invalidate()
        self.say(
            f"Pasted {fmt.duration(self.session.clipboard.duration)} at "
            f"{fmt.secs(time)}. {self.session.current} is now "
            f"{self.session.sound().duration:.3f} seconds."
        )

    def do_swap(self, arg):
        """swap A B. Exchange two segments. Use sound:label to cross sounds."""
        parts = arg.split()
        if len(parts) != 2:
            raise SessionError(
                "Say swap, then two segments, for example swap pot:2 pat:2."
            )
        name_a, span_a = self._span(parts[0])
        name_b, span_b = self._span(parts[1])
        if name_a == name_b:
            raise SessionError(
                "Both segments are in the same sound. Swapping within one sound "
                "would shift the second segment's own boundaries, so use cut and "
                "paste for that."
            )
        new_a, new_b = edit.swap(
            self.session.sound(name_a), span_a, self.session.sound(name_b), span_b
        )
        self.session.replace(new_a, name_a)
        self.session.replace(new_b, name_b)
        self.invalidate()
        self.say(
            f"Swapped {fmt.duration(span_a[1] - span_a[0])} of {name_a} with "
            f"{fmt.duration(span_b[1] - span_b[0])} of {name_b}.",
            f"{name_a} is now {self.session.sound(name_a).duration:.3f} seconds, "
            f"{name_b} is {self.session.sound(name_b).duration:.3f} seconds.",
            "Type play to hear the current one.",
        )

    def do_splice(self, arg):
        """splice TARGET SOURCE. Replace the target segment with the source."""
        parts = arg.split()
        if len(parts) != 2:
            raise SessionError("Say splice, then the target segment and the source.")
        name_t, span_t = self._span(parts[0])
        name_s, span_s = self._span(parts[1])
        clip = edit.extract(self.session.sound(name_s), *span_s)
        target = self.session.sound(name_t)
        self.session.replace(
            edit.replace_range(target, span_t[0], span_t[1], clip), name_t
        )
        self.invalidate()
        self.say(
            f"Replaced {fmt.duration(span_t[1] - span_t[0])} of {name_t} with "
            f"{fmt.duration(span_s[1] - span_s[0])} from {name_s}.",
            f"{name_t} is now {self.session.sound(name_t).duration:.3f} seconds.",
        )

    def do_stretch(self, arg):
        """stretch FACTOR. Change duration without changing pitch."""
        factor = float(arg.strip() or 0)
        before = self.session.sound().duration
        self.session.replace(edit.stretch(self.session.sound(), factor))
        self.invalidate()
        self.say(
            f"Stretched by {factor}. {self.session.current} went from "
            f"{before:.3f} to {self.session.sound().duration:.3f} seconds. "
            "The pitch is unchanged."
        )

    def do_pitchshift(self, arg):
        """pitchshift SEMITONES. Raise or lower pitch, keeping the duration."""
        semitones = float(arg.strip() or 0)
        self.session.replace(edit.pitch_shift(self.session.sound(), semitones))
        self.invalidate()
        direction = "up" if semitones > 0 else "down"
        self.say(
            f"Shifted {self.session.current} {direction} by {abs(semitones)} "
            "semitones. The duration is unchanged."
        )

    def do_flatten(self, arg):
        """flatten [HERTZ]. Replace the pitch contour with a monotone."""
        target = float(arg.strip()) if arg.strip() else None
        sound, used = edit.flatten(self.session.sound(), target)
        self.session.replace(sound)
        self.invalidate()
        self.say(f"Flattened the pitch of {self.session.current} to {used:.0f} hertz.")

    def do_reverse(self, arg):
        """reverse. Reverse the sound."""
        self.session.replace(edit.reverse(self.session.sound()))
        self.invalidate()
        self.say(f"Reversed {self.session.current}.")

    def do_normalize(self, arg):
        """normalize. Scale the peak up to just under full scale."""
        self.session.replace(edit.normalize(self.session.sound()))
        self.invalidate()
        self.say(f"Scaled {self.session.current} so its peak is just under full scale.")

    def do_concat(self, arg):
        """concat A B. Join loaded sounds end to end into a new sound."""
        parts = arg.split()
        if len(parts) < 2:
            raise SessionError("Say concat, then two or more loaded sound names.")
        pieces = [self.session.sound(p) for p in parts]
        joined = edit.join([edit.match_rate(p, pieces[0]) for p in pieces])
        name = self.session.add("-".join(parts), joined)
        self.session.use(name)
        self.invalidate()
        self.say(f"Joined {fmt.join(parts)} into {name}, {joined.duration:.3f} seconds.")

    # --- settings and the real Praat ----------------------------------

    def do_settings(self, arg):
        """settings. Say the current analysis settings."""
        self.say(analysis.SETTINGS.describe() + ".")

    def do_set(self, arg):
        """set NAME VALUE. pitchfloor, pitchceiling or formantceiling."""
        parts = arg.split()
        names = {
            "pitchfloor": "pitch_floor",
            "pitchceiling": "pitch_ceiling",
            "formantceiling": "formant_ceiling",
            "formants": "formant_count",
        }
        if len(parts) != 2 or parts[0] not in names:
            raise SessionError(
                "Say set, then one of pitchfloor, pitchceiling, formantceiling or "
                "formants, then a number."
            )
        setattr(analysis.SETTINGS, names[parts[0]], float(parts[1]))
        self.invalidate()
        self.say(f"Set {parts[0]} to {parts[1]}. {analysis.SETTINGS.describe()}.")

    def do_praat(self, arg):
        """praat gui, or praat script FILE."""
        parts = arg.split()
        if not parts:
            raise SessionError("Say praat gui, or praat script followed by a filename.")
        if parts[0] == "gui":
            path = praatbridge.open_gui(self.session.selected(), self.session.current)
            self.say(f"Opened the selection in the real Praat, from {path}.")
        elif parts[0] == "script":
            if len(parts) < 2:
                raise SessionError("Say praat script, then the script's filename.")
            rest = list(parts[2:])
            trust = "trust" in rest
            if trust:
                rest.remove("trust")
            self.say(praatbridge.run_script(parts[1], rest, trust=trust))
        else:
            raise SessionError("praat takes either gui or script.")


def run(root, argv=None):
    shell = Shell(root)
    if argv:
        shell.onecmd(" ".join(argv))
        return 0
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    return 0
