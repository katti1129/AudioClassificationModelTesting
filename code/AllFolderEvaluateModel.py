# -*- coding: utf-8 -*-
import os
import glob
import csv
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# ======================
# ReduceSumLayer
# ======================
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


# ======================
# 設定
# ======================
PARENT_DIR = "/Users/katti/Desktop/Lab/AudioClassificationTesting/data/fold5"  # ← 各クラスフォルダを含む親フォルダ
MODEL_PATH = "/Users/katti/Desktop/Lab/AudioClassificationTesting/code/best.keras"

OUTPUT_CSV = "prediction_results.csv"
WRONG_CSV = "wrong_predictions.csv"

CLASS_NAMES = ['other', 'silence', 'siren']

SAMPLE_RATE = 16000
DURATION = 1.5
WIN_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN, FMAX = 20, 8000


# ======================
# 前処理
# ======================
def preprocess_audio(path):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except:
        print(f"[Error] 読み込み失敗: {path}")
        return None
    
    if len(y) < WIN_SAMPLES:
        y = np.pad(y, (0, WIN_SAMPLES - len(y)))
    else:
        y = y[:WIN_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

    logmel = np.expand_dims(logmel.astype(np.float32), -1)
    return np.expand_dims(logmel, 0)


# ======================
# 推論 & CSV保存
# ======================
def main():

    # === モデルロード ===
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] モデルが見つかりません: {MODEL_PATH}")
        return
    
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model loaded.\n")

    TARGET_FOLDERS = ['other', 'silence', 'siren']

    # === CSV初期化 ===
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "true_label", "pred_label",
                         "prob_other", "prob_silence", "prob_siren",
                         "confidence", "correct"])

    with open(WRONG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "true_label", "pred_label", "confidence"])


    # === 全フォルダ探索 ===
    for true_label in TARGET_FOLDERS:

        target_dir = os.path.join(PARENT_DIR, true_label)
        print(f"=== Scan: {target_dir} ===")

        if not os.path.exists(target_dir):
            print(f"[Warning] フォルダなし → スキップ: {target_dir}")
            continue

        # 音声ファイル探索
        audio_files = []
        for ext in ["wav", "mp3", "flac"]:
            audio_files.extend(
                glob.glob(os.path.join(target_dir, f"**/*.{ext}"), recursive=True)
            )

        if not audio_files:
            print(f"[Warning] ファイルなし: {target_dir}")
            continue

        print(f"Found {len(audio_files)} files.\n")

        # === 推論実行 ===
        for path in sorted(audio_files):

            print(f"-- {path}")

            data = preprocess_audio(path)
            if data is None:
                continue

            preds = model.predict(data, verbose=0)[0]
            pred_idx = np.argmax(preds)
            pred_label = CLASS_NAMES[pred_idx]
            conf = preds[pred_idx] * 100

            # === 正誤判定 ===
            correct_flag = (pred_label == true_label)

            # === 結果をCSVに書き込み ===
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    path,                            # file
                    true_label,                      # true_label
                    pred_label,                      # pred_label
                    preds[0], preds[1], preds[2],    # raw probs
                    conf,                            # confidence
                    "correct" if correct_flag else "wrong"
                ])

            # === 誤分類なら wrong.csv に記録 ===
            if not correct_flag:
                with open(WRONG_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([path, true_label, pred_label, conf])

            print(f"Prediction: {pred_label} ({conf:.2f}%) → {'OK' if correct_flag else 'WRONG'}\n")


if __name__ == "__main__":
    main()
