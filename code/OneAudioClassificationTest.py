# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
# ================= 設定 =================



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



# ================= 設定 =================
# ★ テストしたいファイルのパス (適宜書き換えてください)
TARGET_FILE = "/Users/katti/Desktop/Lab/AudioClassificationTesting/data/Ambulance-Siren01-2.wav"

# ★ 学習済みモデルのパス
MODEL_PATH = "/Users/katti/Desktop/Lab/AudioClassificationTesting/code/best.keras"  # 実際のパスに合わせてください

# クラス名 (学習時の順番)
CLASS_NAMES = ['other', 'silence', 'siren']

# 学習時と同じパラメータ (TrainCRNN_9Fold.pyより)
SAMPLE_RATE = 16000
DURATION = 1.5
WIN_SAMPLES = int(SAMPLE_RATE * DURATION) # 24000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN, FMAX = 20, 8000
# ========================================

def preprocess_audio(path):
    """学習時と全く同じ前処理を行う関数"""
    # 1. 読み込み
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # 2. 長さ調整 (1.5秒に合わせる)
    if len(y) < WIN_SAMPLES:
        # 短い場合はパディング (0埋め)
        y = np.pad(y, (0, WIN_SAMPLES - len(y)))
    else:
        # 長い場合は「中央」を切り出すか、「先頭」を使う
        # ここでは学習時と同様に先頭を使います
        y = y[:WIN_SAMPLES]

    # 3. メルスペクトログラム変換
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    
    # 4. Log変換 (dB)
    logmel = librosa.power_to_db(mel, ref=np.max)
    
    # 5. 正規化 (重要: これがないと精度が出ない)
    # 平均0, 分散1に揃える
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    
    # 6. 形状調整 (Height, Width) -> (Height, Width, Channel)
    # shape: (128, 47) -> (128, 47, 1)
    logmel = logmel.astype(np.float32)
    logmel = np.expand_dims(logmel, axis=-1)
    
    # 7. Batch次元追加 -> (1, 128, 47, 1)
    input_data = np.expand_dims(logmel, axis=0)
    
    return input_data

def main():
    # 1. モデルロード
    if not os.path.exists(MODEL_PATH):
        print(f"Error: モデルが見つかりません -> {MODEL_PATH}")
        return
    
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    # 2. ファイル存在確認
    if not os.path.exists(TARGET_FILE):
        print(f"Error: 音声ファイルが見つかりません -> {TARGET_FILE}")
        # テスト用にダミー生成などの処理を入れても良いですが、まずはパス確認を
        return

    # 3. 前処理
    print(f"\nProcessing: {TARGET_FILE}")
    input_data = preprocess_audio(TARGET_FILE)
    if input_data is None:
        return

    # 4. 推論
    print("Predicting...")
    # shape確認
    # print(f"Input shape: {input_data.shape}")

    preds = model.predict(input_data, verbose=0)[0]

    # 5. 結果表示
    print("\n=== Result ===")
    for i, prob in enumerate(preds):
        print(f"{CLASS_NAMES[i]:8s}: {prob*100:.2f}%")

    predicted_idx = np.argmax(preds)
    confidence = preds[predicted_idx] * 100
    print(f"\n[Final Decision]: {CLASS_NAMES[predicted_idx]} ({confidence:.2f}%)")

    # 注意喚起
    if CLASS_NAMES[predicted_idx] == "siren" and confidence < 99.0:
        print("※ Siren判定ですが、確信度がやや低いです。ノイズの影響の可能性があります。")

if __name__ == "__main__":
    main()