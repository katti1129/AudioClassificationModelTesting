#スライド窓で評価する版

# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model


@tf.keras.utils.register_keras_serializable()
class ReduceSumLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


# ===== Settings =====
TARGET_FILE = "/Users/katti/Desktop/Lab/AudioClassificationTesting/data/ambulance.wav"

MODEL_PATH = "/Users/katti/Desktop/Lab/AudioClassificationTesting/code/best.keras"

CLASS_NAMES = ["other", "siren"]

SAMPLE_RATE = 16000
WINDOW_DURATION = 1.0
HOP_DURATION = 0.5
WIN_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN, FMAX = 20, 8000
SIREN_THRESHOLD = 0.5
# ====================


def waveform_to_input(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    logmel = logmel.astype(np.float32)
    logmel = np.expand_dims(logmel, axis=-1)
    return logmel


def load_audio(path):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

    if len(y) < WIN_SAMPLES:
        y = np.pad(y, (0, WIN_SAMPLES - len(y)))

    return y, sr


def make_window_batch(y, sr):
    windows = []
    time_ranges = []

    last_start = len(y) - WIN_SAMPLES
    starts = range(0, last_start + 1, HOP_SAMPLES)

    for start in starts:
        end = start + WIN_SAMPLES
        segment = y[start:end]

        if len(segment) < WIN_SAMPLES:
            segment = np.pad(segment, (0, WIN_SAMPLES - len(segment)))

        windows.append(waveform_to_input(segment, sr))
        time_ranges.append((start / sr, end / sr))

    input_data = np.stack(windows, axis=0)
    return input_data, time_ranges


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found -> {MODEL_PATH}")
        return

    if not os.path.exists(TARGET_FILE):
        print(f"Error: audio file not found -> {TARGET_FILE}")
        return

    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    print(f"\nProcessing: {TARGET_FILE}")
    y, sr = load_audio(TARGET_FILE)
    if y is None:
        return

    input_data, time_ranges = make_window_batch(y, sr)

    print(f"Audio length      : {len(y) / sr:.2f} s")
    print(f"Window duration  : {WINDOW_DURATION:.2f} s")
    print(f"Hop duration     : {HOP_DURATION:.2f} s")
    print(f"Number of windows: {len(time_ranges)}")

    print("\nPredicting...")
    preds = model.predict(input_data, verbose=0)
    siren_probs = preds[:, CLASS_NAMES.index("siren")]
    predicted_idx = np.argmax(preds, axis=1)

    print("\n=== Window Results ===")
    for i, ((start, end), prob) in enumerate(zip(time_ranges, siren_probs), start=1):
        label = CLASS_NAMES[predicted_idx[i - 1]]
        print(f"{i:03d}: {start:6.2f}-{end:6.2f} s | siren: {prob*100:6.2f}% | {label}")

    mean_siren = float(np.mean(siren_probs))
    max_siren = float(np.max(siren_probs))
    max_idx = int(np.argmax(siren_probs))
    siren_ratio = float(np.mean(siren_probs >= SIREN_THRESHOLD))

    mean_probs = np.mean(preds, axis=0)
    final_idx_by_mean = int(np.argmax(mean_probs))

    print("\n=== Summary ===")
    print(f"Mean siren probability : {mean_siren*100:.2f}%")
    print(f"Max siren probability  : {max_siren*100:.2f}%")
    print(
        f"Max siren window       : "
        f"{time_ranges[max_idx][0]:.2f}-{time_ranges[max_idx][1]:.2f} s"
    )
    print(f"Siren window ratio     : {siren_ratio*100:.2f}%")

    print("\n=== Mean Probability Decision ===")
    for i, prob in enumerate(mean_probs):
        print(f"{CLASS_NAMES[i]:8s}: {prob*100:.2f}%")
    print(f"\n[Final Decision]: {CLASS_NAMES[final_idx_by_mean]} ({mean_probs[final_idx_by_mean]*100:.2f}%)")


if __name__ == "__main__":
    main()
 