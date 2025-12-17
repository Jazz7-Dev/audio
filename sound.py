import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from moviepy import VideoFileClip
import urllib.request
import os
from colorama import Fore, Style, init

# Initialize colorama for Windows consoles
init(autoreset=True)

# ----------------------------
# Load any audio or video file
# ----------------------------
def load_audio(file_path, sample_rate=16000):

    ext = os.path.splitext(file_path)[1].lower()

    # If the file is audio (mp3 / wav), load directly
    if ext in [".wav", ".mp3", ".flac", ".ogg"]:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        return audio, sr

    # If it's a video file → extract audio
    temp_audio = "temp_extracted.wav"
    clip = VideoFileClip(file_path)
    clip.audio.write_audiofile(temp_audio)   # FIXED: no verbose/logger args

    audio, sr = librosa.load(temp_audio, sr=sample_rate)
    return audio, sr


# ----------------------------
# Load YAMNet model
# ----------------------------
model_url = "https://tfhub.dev/google/yamnet/1"
yamnet = hub.load(model_url)

# ----------------------------
# Load class labels (FIXED URL)
# ----------------------------
labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
labels_path = "yamnet_class_map.csv"
urllib.request.urlretrieve(labels_url, labels_path)

class_names = []
with open(labels_path, "r") as f:
    for line in f.readlines()[1:]:
        class_names.append(line.strip().split(",")[2])


# ----------------------------
# Sound classification
# ----------------------------
def analyze_sound(file_path):
    audio, sr = load_audio(file_path)

    # Convert stereo to mono if required
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    scores, embeddings, spectrogram = yamnet(audio)
    scores = scores.numpy()

    total = np.sum(scores, axis=0)
    percentages = (total / np.sum(total)) * 100

    sorted_idx = np.argsort(percentages)[::-1]

    print("\n===== SOUND PERCENTAGE REPORT =====\n")

    # Print top N classes in sorted (descending) order and highlight the top one
    top_n = 15
    for rank, i in enumerate(sorted_idx[:top_n], start=1):
        name = f"{class_names[i]:30s}"
        pct = f"{percentages[i]:.2f}%"

        if rank == 1:
            # Highlight the highest-percentage item
            print(Fore.GREEN + Style.BRIGHT + f">> {name} -> {pct}")
        else:
            print(f"   {name} -> {pct}")


def get_percentages(file_path, top_n=15):
    """Return a list of (name, percentage) tuples sorted descending by percentage.

    This is intended for use by a web UI.
    """
    audio, sr = load_audio(file_path)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    scores, embeddings, spectrogram = yamnet(audio)
    scores = scores.numpy()

    total = np.sum(scores, axis=0)
    percentages = (total / np.sum(total)) * 100

    sorted_idx = np.argsort(percentages)[::-1]

    results = []
    for i in sorted_idx[:top_n]:
        results.append({
            "name": class_names[i],
            "percentage": float(percentages[i])
        })

    return results


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    file_path = r"C:\Users\DEVANSH\Desktop\1-977-A-39.wav" # change your file path here
    analyze_sound(file_path)
