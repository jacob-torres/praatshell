praatshell
A screen-reader-first command shell for phonetic analysis, built on Praat.


WHAT IT IS

Praat keeps nearly all of its analysis inside graphical windows. The waveform
and the spectrogram are pictures with no spoken equivalent, so a screen reader
cannot get at them. praatshell does the same measurements through Praat's own
engine and reports them as text: you type a command, it answers in a sentence or
two, and it writes the full detail to a file you can read at your own pace.

The part that matters most is "describe". It divides a recording into regions,
measures each one, and says in words what the waveform is physically doing:
whether it is periodic, how many cycles it contains, how its loudness rises and
falls, and where its energy sits in frequency. That is the information a
spectrogram carries, delivered as prose.


WHAT YOU NEED

Python, and two Python packages. Install them once:

    pip install --user praat-parselmouth numpy

parselmouth contains Praat's own analysis code, so the numbers match what Praat
itself produces. The separate Praat program is not required for analysis, but if
it is installed, the "praat" command here can hand a sound to it.


STARTING IT

From this folder, in Git Bash or PowerShell:

    python -m praatshell

Or run the launcher: ./praatshell.sh in Git Bash, praatshell.bat elsewhere.

You can also run one command and exit, which is useful from a script:

    python -m praatshell describe

The prompt tells you which sound you are working on. It reads "praatshell" until
you load something, then the sound's name.

Type help for the full command list, one command per line.


A FIRST SESSION

    load Sept16-1.Collection
    list
    describe
    events
    autoseg
    seg list
    select seg 7
    play
    describe

load accepts a .wav file, or a Praat .Collection, in which case every sound
inside it is loaded under its own name.

describe writes a full report and speaks a short summary. events gives just the
timeline. autoseg turns that timeline into numbered segments you can select,
play, measure and edit.


MEASUREMENT VERSUS GUESSWORK

Every report separates two things.

MEASURED is fact. Durations, frequencies, amplitudes, decibels: these come from
Praat and you can quote them in an assignment.

LIKELY is the tool reasoning from thresholds, and it always carries a
confidence, the reason for that confidence, and the next most plausible reading.
A guess marked "low" is phrased as an open question. Never quote a LIKELY line
without its confidence, and never quote one as a measurement.

The vowel guesses compare your formants against adult male reference values. A
higher or lower voice shifts the match, which is why the confidence line says so
every time.


WHAT GETS WRITTEN

Descriptions land in the reports folder beside this file. Numbers go one level
down, in reports/csv, so the reports folder stays a list of things to read.

    reports/
      NAME.txt                           the whole-sound summary
      NAME__START-ENDms.txt              the prose description of one stretch
      NAME__START-ENDms_pitch.txt        single-layer reports, as you ask for them
      csv/
        NAME__START-ENDms_pitch.csv      fundamental frequency, frame by frame
        NAME__START-ENDms_formants.csv   F1, F2, F3
        NAME__START-ENDms_intensity.csv  loudness in decibels
        NAME__START-ENDms_bands.csv      energy per frequency band

NAME.txt is the one to open first. Its name has no time span in it, so it sorts
above every other file for that sound. describe rewrites it each time, and it
always covers the entire recording even when you were describing a short
selection: an overview, a one-line reading of every region with its confidence,
the pitch and loudness contours, and an index of every other report written for
that sound.

The CSV files open in R or a spreadsheet. The pitch CSV has two columns for
frequency: f0_hz, with octave errors removed, and f0_raw_hz exactly as Praat
measured it, so nothing is hidden from you.


TIMES

Times are in seconds, or in milliseconds with an ms suffix. These are the same:

    select 0.768 0.828
    select 768ms 828ms


EDITING AND SPLICING

    copy 7            copy segment 7 to the clipboard
    paste 0.4         insert it at 0.4 seconds
    cut 7             remove segment 7, keeping it on the clipboard
    swap pot:7 dock:7 exchange segment 7 between two loaded sounds
    splice pot:7 dock:7   replace pot's segment 7 with dock's

swap and splice take sound:label, so you can move a vowel from one word into
another. undo steps back one change at a time.

    stretch 1.5       make it half again as long, pitch unchanged
    pitchshift 3      raise by three semitones, duration unchanged
    flatten           replace the pitch contour with a monotone

These use Praat's overlap-add resynthesis, the same as the Manipulation window.


SETTINGS

    settings                 say the current values
    set pitchfloor 100       raise the floor for a higher voice
    set pitchceiling 300     lower the ceiling to stop octave errors
    set formantceiling 5000  5000 suits a male voice, 5500 a female one

If pitch readings look like they are doubling, lower the ceiling. If a low voice
is being missed, lower the floor. The floor also sets how much sound Praat needs
before it can measure pitch at all: at 75 hertz that is 85 milliseconds.

Selections shorter than that still work. praatshell analyses a padded stretch
either side of your selection and then keeps only the frames inside it, which is
what Praat's own editor does, so a 60 millisecond vowel still gets a pitch.


THE REAL PRAAT

    praat gui              open the current selection in the Praat program
    praat script FILE      run a .praat script and read back what it printed

Useful when a classmate or instructor wants to look at the same file, or for
anything praatshell does not cover.


IF SOMETHING GOES WRONG

Nothing echoes when you type, in Git Bash: run it as

    winpty python -u -m praatshell

MinTTY sometimes needs that for interactive programs. Inside the VS Code
terminal it is usually unnecessary.

"No pitch could be measured": the stretch may genuinely be voiceless, or the
pitch floor may be too high for the voice. Try set pitchfloor 60.

A command fails with a message from Praat: Praat refuses operations it has too
little signal for, especially resynthesis on very short or unvoiced stretches.
Select a longer stretch and try again.
