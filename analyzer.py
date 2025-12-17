from moviepy.editor import VideoFileClip
import numpy as np
import matplotlib.pyplot as plt
import wave
import contextlib

# Load video and extract audio
clip = VideoFileClip('/mnt/data/water.mp4')
audio = clip.audio

# Write audio to wav
audio_path = '/mnt/data/extracted.wav'
audio.write_audiofile(audio_path)

audio_path
