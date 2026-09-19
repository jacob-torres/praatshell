# ============================================================
# demo_measure.praat
# A simple demonstration script for Praat, run from the command line.
#
# Run it from Git Bash like this:
#     praat --run demo_measure.praat
#
# It prints the measurements to the command line AND writes them to
# praat_measurements.txt in the folder you run it from.
# ============================================================

# --- CHOOSE YOUR SOUND ---
# To analyze your OWN file, put its path between the quotes below.
# Use a drive letter and forward slashes, for example:
#     soundFile$ = "C:/Users/Jacob/praat-work/sample.wav"
# Leave it empty, as it is now, to use a built-in test tone so the
# script runs with no setup at all.
soundFile$ = ""

# --- Get a sound: either the file above, or a built-in test tone ---
if soundFile$ = ""
    sound = Create Sound as pure tone: "demoTone", 1, 0, 0.5, 44100, 220, 0.2, 0.01, 0.01
    source$ = "built-in 220 Hz test tone (no file supplied)"
else
    sound = Read from file: soundFile$
    source$ = soundFile$
endif

# --- Basic properties ---
selectObject: sound
duration = Get total duration
samplingFrequency = Get sampling frequency
numberOfSamples = Get number of samples

# --- Pitch (the fundamental frequency, F0) ---
selectObject: sound
pitch = To Pitch: 0, 75, 600
meanF0 = Get mean: 0, 0, "Hertz"
minF0 = Get minimum: 0, 0, "Hertz", "Parabolic"
maxF0 = Get maximum: 0, 0, "Hertz", "Parabolic"

# --- Intensity (loudness, in dB) ---
selectObject: sound
intensity = To Intensity: 100, 0, "yes"
meanIntensity = Get mean: 0, 0, "energy"

# --- Formants at the temporal midpoint (F1, F2, F3) ---
selectObject: sound
formant = To Formant (burg): 0, 5, 5500, 0.025, 50
midpoint = duration / 2
f1 = Get value at time: 1, midpoint, "Hertz", "Linear"
f2 = Get value at time: 2, midpoint, "Hertz", "Linear"
f3 = Get value at time: 3, midpoint, "Hertz", "Linear"

# --- Build one report, then send it to BOTH the screen and a file ---
report$ = "Praat acoustic measurements" + newline$
report$ = report$ + "Source: " + source$ + newline$
report$ = report$ + "Duration (s): " + fixed$(duration, 3) + newline$
report$ = report$ + "Sampling frequency (Hz): " + fixed$(samplingFrequency, 0) + newline$
report$ = report$ + "Number of samples: " + fixed$(numberOfSamples, 0) + newline$
report$ = report$ + "Mean F0 (Hz): " + fixed$(meanF0, 2) + newline$
report$ = report$ + "Min F0 (Hz): " + fixed$(minF0, 2) + newline$
report$ = report$ + "Max F0 (Hz): " + fixed$(maxF0, 2) + newline$
report$ = report$ + "Mean intensity (dB): " + fixed$(meanIntensity, 2) + newline$
report$ = report$ + "F1 at midpoint (Hz): " + fixed$(f1, 2) + newline$
report$ = report$ + "F2 at midpoint (Hz): " + fixed$(f2, 2) + newline$
report$ = report$ + "F3 at midpoint (Hz): " + fixed$(f3, 2) + newline$

# Print to the command line
writeInfoLine: report$

# Write the same text to a file in the current directory (overwrites each run)
writeFile: "praat_measurements.txt", report$
