
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# =========================================================
# PROJECT: AI Music Generator for a Student Focus Playlist
# =========================================================

SAMPLE_RATE = 22050
DURATION = 30
OUTPUT_DIR = "focus_music"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. MUSIC GENERATION
# ---------------------------------------------------------

def generate_music(bpm=82, seed=1):
    rng = np.random.default_rng(seed)
    total_samples = SAMPLE_RATE * DURATION
    music = np.zeros(total_samples, dtype=np.float64)

    beat = 60.0 / bpm

    def add_note(start_time, duration, frequency, volume=0.1):
        start = int(start_time * SAMPLE_RATE)
        end = min(
            total_samples,
            int((start_time + duration) * SAMPLE_RATE)
        )

        if start >= total_samples or end <= start:
            return

        t = np.arange(end - start) / SAMPLE_RATE

        # Synthesized piano-like sound
        tone = (
            np.sin(2 * np.pi * frequency * t)
            + 0.30 * np.sin(2 * np.pi * frequency * 2 * t)
            + 0.10 * np.sin(2 * np.pi * frequency * 3 * t)
        )

        envelope = np.exp(-4.0 * t)
        attack = np.minimum(1.0, t / 0.015)

        music[start:end] += volume * tone * envelope * attack

    # Chord progression: C, Am, F, G
    chords = [
        [130.81, 164.81, 196.00],
        [110.00, 130.81, 164.81],
        [87.31, 110.00, 130.81],
        [98.00, 123.47, 146.83]
    ]

    bar_duration = 4 * beat

    # Background chords
    for bar, start_time in enumerate(
        np.arange(0, DURATION, bar_duration)
    ):
        chord = chords[bar % len(chords)]

        start = int(start_time * SAMPLE_RATE)
        end = min(
            total_samples,
            int((start_time + bar_duration) * SAMPLE_RATE)
        )

        t = np.arange(end - start) / SAMPLE_RATE

        fade_in = np.minimum(1.0, t / 0.3)
        fade_out = np.minimum(
            1.0,
            np.maximum(0.0, (bar_duration - t) / 0.3)
        )

        pad = np.zeros(len(t))

        for frequency in chord:
            pad += np.sin(2 * np.pi * frequency * t)

        pad /= len(chord)

        music[start:end] += (
            0.07 * pad * fade_in * fade_out
        )

    # Melody with rhythmic movement
    melody = [
        523.25, 587.33, 659.25, 587.33,
        523.25, 392.00, 440.00, 523.25,
        587.33, 659.25, 783.99, 659.25,
        587.33, 523.25, 440.00, 392.00
    ]

    note_duration = beat / 2
    steps = int(DURATION / note_duration)

    for i in range(steps):
        if rng.random() < 0.08:
            continue

        # Small variations make repeated phrases less identical
        frequency = melody[i % len(melody)]

        if rng.random() < 0.15:
            frequency *= 1.12246

        add_note(
            i * note_duration,
            note_duration * 0.9,
            frequency,
            volume=0.11
        )

    # Bass notes on selected beats
    bass_notes = [130.81, 110.00, 87.31, 98.00]

    for bar, start_time in enumerate(
        np.arange(0, DURATION, bar_duration)
    ):
        bass = bass_notes[bar % len(bass_notes)]

        for beat_index in [0, 2]:
            add_note(
                start_time + beat_index * beat,
                beat * 0.8,
                bass,
                volume=0.12
            )

    # Soft rhythmic pulse
    for start_time in np.arange(0, DURATION, beat):
        add_note(start_time, 0.12, 90, volume=0.035)

    # Smooth beginning and ending
    fade_length = min(
        int(SAMPLE_RATE),
        total_samples // 2
    )

    music[:fade_length] *= np.linspace(
        0, 1, fade_length
    )

    music[-fade_length:] *= np.linspace(
        1, 0, fade_length
    )

    # Prevent clipping
    peak = np.max(np.abs(music))

    if peak > 0:
        music = music / peak * 0.75

    return music.astype(np.float32)


# ---------------------------------------------------------
# 2. AUDIO FEATURE ANALYSIS
# ---------------------------------------------------------

def analyze_audio(audio, bpm):
    rms = np.sqrt(np.mean(audio ** 2))

    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(
        len(audio),
        d=1 / SAMPLE_RATE
    )

    centroid = (
        np.sum(frequencies * spectrum)
        / (np.sum(spectrum) + 1e-10)
    )

    zero_crossing_rate = np.mean(
        np.diff(np.signbit(audio)).astype(float)
    )

    return {
        "Tempo_BPM": bpm,
        "RMS_Energy": round(float(rms), 4),
        "Spectral_Centroid_Hz": round(float(centroid), 2),
        "Zero_Crossing_Rate": round(
            float(zero_crossing_rate), 4
        )
    }


# ---------------------------------------------------------
# 3. STUDY SUITABILITY EVALUATION
# ---------------------------------------------------------

def evaluate_study_suitability(row):
    score = 0

    # Moderate tempo receives a higher score
    if 75 <= row["Tempo_BPM"] <= 95:
        score += 40
    elif 65 <= row["Tempo_BPM"] < 75 or (
        95 < row["Tempo_BPM"] <= 105
    ):
        score += 25
    else:
        score += 10

    # Lower RMS energy is preferred by this heuristic
    if row["RMS_Energy"] <= 0.20:
        score += 30
    elif row["RMS_Energy"] <= 0.35:
        score += 20
    else:
        score += 10

    # Lower spectral brightness is preferred
    if row["Spectral_Centroid_Hz"] <= 1500:
        score += 30
    elif row["Spectral_Centroid_Hz"] <= 3000:
        score += 20
    else:
        score += 10

    if score >= 80:
        category = "Highly Suitable"
    elif score >= 60:
        category = "Suitable"
    else:
        category = "Less Suitable"

    return pd.Series({
        "Suitability_Score": score,
        "Classification": category
    })


# ---------------------------------------------------------
# 4. MAIN PROGRAM
# ---------------------------------------------------------

def main():
    track_settings = [
        {"name": "LoFi_Focus", "bpm": 82, "seed": 10},
        {"name": "Piano_Study", "bpm": 88, "seed": 20},
        {"name": "Steady_Focus", "bpm": 92, "seed": 30},
        {"name": "Creative_Flow", "bpm": 96, "seed": 40},
        {"name": "Calm_Work", "bpm": 78, "seed": 50}
    ]

    results = []

    print("=" * 55)
    print(" AI MUSIC GENERATOR FOR A STUDENT FOCUS PLAYLIST")
    print("=" * 55)

    # Generate and save tracks
    for track in track_settings:
        name = track["name"]
        bpm = track["bpm"]

        print(f"\nGenerating {name}...")

        audio = generate_music(
            bpm=bpm,
            seed=track["seed"]
        )

        filename = os.path.join(
            OUTPUT_DIR, f"{name}.wav"
        )

        audio_int16 = (
            np.clip(audio, -1, 1) * 32767
        ).astype(np.int16)

        write(filename, SAMPLE_RATE, audio_int16)

        features = analyze_audio(audio, bpm)
        features["Track_Name"] = name
        results.append(features)

        print(f"Saved music: {filename}")

    # Create feature analysis table
    audio_df = pd.DataFrame(results)

    audio_df = audio_df[
        [
            "Track_Name",
            "Tempo_BPM",
            "RMS_Energy",
            "Spectral_Centroid_Hz",
            "Zero_Crossing_Rate"
        ]
    ]

    feature_path = os.path.join(
        OUTPUT_DIR, "audio_feature_analysis.csv"
    )

    audio_df.to_csv(feature_path, index=False)

    print("\nAUDIO FEATURE ANALYSIS")
    print(audio_df.to_string(index=False))

    # Evaluate all tracks
    evaluation_df = audio_df.join(
        audio_df.apply(
            evaluate_study_suitability,
            axis=1
        )
    )

    evaluation_df = evaluation_df.sort_values(
        "Suitability_Score",
        ascending=False
    ).reset_index(drop=True)

    evaluation_path = os.path.join(
        OUTPUT_DIR, "study_suitability_results.csv"
    )

    evaluation_df.to_csv(
        evaluation_path,
        index=False
    )

    print("\nSTUDY SUITABILITY RESULTS")
    print(evaluation_df.to_string(index=False))

    # Create recommended playlist
    focus_playlist = evaluation_df[
        evaluation_df["Classification"].isin(
            ["Highly Suitable", "Suitable"]
        )
    ].copy()

    playlist_path = os.path.join(
        OUTPUT_DIR, "recommended_focus_playlist.csv"
    )

    focus_playlist.to_csv(
        playlist_path,
        index=False
    )

    print("\nRECOMMENDED STUDENT FOCUS PLAYLIST")

    if focus_playlist.empty:
        print("No tracks met the selected score threshold.")
    else:
        for position, (_, row) in enumerate(
            focus_playlist.iterrows(),
            start=1
        ):
            print(
                f"{position}. {row['Track_Name']} | "
                f"Score: {row['Suitability_Score']}/100 | "
                f"{row['Classification']}"
            )

    # Plot comparison chart
    plt.figure(figsize=(10, 5))

    plt.bar(
        evaluation_df["Track_Name"],
        evaluation_df["Suitability_Score"]
    )

    plt.title("Study Suitability of Generated Music")
    plt.xlabel("Music Track")
    plt.ylabel("Suitability Score (out of 100)")
    plt.ylim(0, 100)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    chart_path = os.path.join(
        OUTPUT_DIR, "study_suitability_chart.png"
    )

    plt.savefig(chart_path, dpi=150)
    plt.show()
    plt.close()

    # Project documentation
    readme = """
AI MUSIC GENERATOR FOR A STUDENT FOCUS PLAYLIST

Use case:
Generate instrumental music for students and evaluate
its suitability for studying.

Description:
This program generates synthesized instrumental music,
analyzes audio features, scores tracks using predefined
rules, and creates a recommended playlist.

Libraries:
Python, NumPy, Pandas, Matplotlib, SciPy.

Output files:
- Five WAV music tracks
- audio_feature_analysis.csv
- study_suitability_results.csv
- recommended_focus_playlist.csv
- study_suitability_chart.png
- README.txt

Limitation:
Music is synthesized using mathematical waveforms.
Suitability scores are heuristic estimates, not proof
of improved concentration or learning.
"""

    readme_path = os.path.join(
        OUTPUT_DIR, "README.txt"
    )

    with open(readme_path, "w", encoding="utf-8") as file:
        file.write(readme)

    print("\n" + "=" * 55)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print(f"All output files are in: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
