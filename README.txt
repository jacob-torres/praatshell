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

    pip install praat-parselmouth numpy

The package is called praat-parselmouth on PyPI even though you import it as
parselmouth. Do not "pip install parselmouth": that is an unrelated, broken
Google Ads package. Avoid --user on Windows: some Python installs do not put
the user site directory on the import path, so the install silently vanishes.

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

Pressing enter on its own repeats the previous command. It says "Again:" and
the command first, so you always know what is about to run. A command that
failed is not repeated.


A FIRST SESSION

    load Sept16-1.Collection
    list
    describe
    vowels
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
play, measure and edit. autoseg followed by a number cuts the selection into
equal segments of that many milliseconds instead:

    autoseg 50
    autoseg 50ms
    select seg 4

A bare number is milliseconds; 50ms or 0.05s also work. The last segment keeps
whatever is left over, so no sound goes unlabelled.

describe can also take a selection or an edit as its argument:

    describe select 0 0.35
    describe select seg 3
    describe reverse
    describe flatten 120

The command after describe runs first and its result is described. Then the
sound and the selection are put back as they were, nothing is played, and
nothing is added to the undo history, so this is a way to hear what a stretch
holds, or what an edit would do, without committing to it. A previewed edit's
report is named after the edit, such as spkr1-pot__0-1204ms_reverse.txt, so
it never overwrites the report for the real sound.


EVERY VOWEL AT ONCE

vowels measures every vowel in the selection and writes them as a table:

    vowels
    vowels select seg 3
    vowels flatten 150

One row per voiced region, with the vowel's duration, its voice onset time,
its pitch (mean, lowest and highest), and F1, F2 and F3. It goes to
NAME__START-ENDms_vowels.txt as an aligned table followed by a fact-per-line
block for each vowel, and to csv/NAME__START-ENDms_vowels.csv for R or a
spreadsheet. It takes the same arguments as describe, listed above.

The table gives two durations. Dur runs from the onset of voicing to the
offset of the formants, which is the vowel. Voiced is the part of that the
pitch tracker found a pitch in. They differ when a vowel devoices at its end
into a voiceless consonant, as the vowel in pat does before its final t: the
pitch disappears while F1 and F2 carry on unchanged and the sound is still
loud. Voicing alone would cut such a vowel short by 50 milliseconds or more,
so the tail is followed for as long as the formants hold their place and the
level stays up. Quote Dur as the vowel's duration; Voiced is there so you can
see how much of it was modal.

A row is a vowel candidate, not a vowel: nasals, laterals and voiced
fricatives are voiced too, so check the rows against what you know was said.
The closest reference vowel in the last column is a guess, and the block for
that vowel gives its confidence and the runner-up.


WHAT A DESCRIBE REPORT HOLDS

The report opens with MEASUREMENTS: a plain list, one number per line, that
you can quote directly. Duration, voice onset time, sampling frequency, peak
amplitude, the region count, how much of the stretch is voiced, pitch median,
mean, range and net change, harmonics-to-noise ratio, level mean and range,
loudness peaks, and the formants of the longest voiced region. The prose
DESCRIPTION follows it: the overview, every region in turn, and the contours.

Voice onset time is measured from the release to the voicing that follows it.
The release is the run of non-silent, non-voiced regions immediately before
the vowel: the burst, plus any aspiration after it. A burst that runs into
aspiration is often labelled aperiodic rather than burst, so position, not
label, is what identifies it. Silence in between means the two belong to
different syllables and no VOT is measured across it. Only a positive VOT can
be found this way: prevoicing does not show.

Where that run starts is then settled on the waveform rather than on the
region boundary. Regions are drawn from Praat's intensity contour, whose
window is 43 milliseconds wide at a 75 hertz pitch floor, so a boundary can
sit up to 21 milliseconds before the event it marks. A release is a step
change in the samples themselves, so the samples place it far more precisely:
in pat the contour put the release at 0.676 seconds, while the waveform jumps
thirty-fold at 0.690. Voicing onset stays on the frame grid, so a VOT is only
as fine as the 10 millisecond frame spacing at that end.

describe reports the VOT of the first vowel in the stretch. vowels reports it
for every vowel. A figure over 120 milliseconds is flagged in the report,
because it usually means the noise before that vowel was not a stop release.
To pin down one stop, select a stretch that starts just before its release.


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

Changing a selection, and every edit above, plays the result straight away, so
you hear what you just did without asking. To work in silence:

    autoplay off      stop playing automatically
    autoplay on       start again
    stop              cut off whatever is playing now

Edits are held in memory. Every later command works on the changed sound, and
keeps doing so until you undo. Nothing on disk changes: the file you loaded is
never written back, and the only commands that write audio are save and seg
save. If you want to keep an edit, save it under a new name.

A selection survives an edit that keeps the same length, such as normalize or
pitchshift, so you stay where you were. An edit that changes the length, such
as stretch or cut, moves everything after it, so the selection resets to the
whole sound.

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
