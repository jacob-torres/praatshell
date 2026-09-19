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


def run_script(path, args=(), trust=False):
    """Run a .praat script and return whatever it printed.

    trust adds Praat's --FULL-TRUST, which is what lets a script write files.
    It stays off by default: it is the flag that removes Praat's own guard
    against a script touching the disk, and it should be a deliberate choice.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"There is no script at {path}.")
    command = [find()]
    if trust:
        command.append("--FULL-TRUST")
    command += ["--run", os.path.abspath(path), *[str(a) for a in args]]

    result = subprocess.run(command, capture_output=True, timeout=300)
    output = _decode(result.stdout).strip()
    error = _decode(result.stderr).strip()

    if result.returncode != 0 or "not completed" in output:
        message = f"Praat reported a problem.\n{error or output}"
        if "FULL-TRUST" in output or "FULL-TRUST" in error:
            message += (
                "\nThat script tried to write a file, which Praat blocks unless "
                "you allow it. To allow it, say: praat script "
                f"{os.path.basename(path)} trust"
            )
        return message
    return output or "The script ran and printed nothing."


def _decode(raw):
    """Praat on Windows prints UTF-16, which the usual decoders mangle."""
    if not raw:
        return ""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw.count(b"\x00") > len(raw) // 4:
        for encoding in ("utf-16", "utf-16-le"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                pass
    return raw.decode("utf-8", errors="replace")


def open_gui(sound, name="sound"):
    """Hand a sound to the real Praat, for a sighted person to look at."""
    path = os.path.join(tempfile.gettempdir(), f"{name}.wav")
    sound.save(path, parselmouth.SoundFileFormat.WAV)
    subprocess.Popen([find(), "--open", path])
    return path
