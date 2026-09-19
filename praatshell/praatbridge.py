"""The escape hatch: the real Praat program, for scripts and for a sighted check."""

import os
import shutil
import subprocess
import tempfile

import parselmouth

CANDIDATES = [
    r"C:\Program Files\Praat\Praat.exe",
    r"C:\Program Files (x86)\Praat\Praat.exe",
]


def find():
    for path in CANDIDATES:
        if os.path.exists(path):
            return path
    found = shutil.which("praat") or shutil.which("Praat")
    if found:
        return found
    raise FileNotFoundError(
        "Praat.exe could not be found. Install Praat, or add its folder to PATH."
    )


def run_script(path, args=()):
    """Run a .praat script and return whatever it printed."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"There is no script at {path}.")
    result = subprocess.run(
        [find(), "--run", os.path.abspath(path), *[str(a) for a in args]],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        error = (result.stderr or "").strip()
        return f"Praat reported an error.\n{error or output}"
    return output or "The script ran and printed nothing."


def open_gui(sound, name="sound"):
    """Hand a sound to the real Praat, for a sighted person to look at."""
    path = os.path.join(tempfile.gettempdir(), f"{name}.wav")
    sound.save(path, parselmouth.SoundFileFormat.WAV)
    subprocess.Popen([find(), "--open", path])
    return path
