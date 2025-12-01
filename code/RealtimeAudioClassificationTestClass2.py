import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import time
from queue import Queue
import threading
import sys

# ==========================================
# 1. カスタムレイヤー定義
# ==========================================
@tf.keras.utils.register_keras_serializable()
class ReduceSumLayer(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

# ==========================================
# 2. 設定・パラメータ
# ==========================================
MODEL_PATH = "/Users/katti/Desktop/Lab/AudioClassificationTesting/code/best.keras"

# ★修正: 学習時のアルファベット順に合わせる (これがズレると誤検知の原因になります)
CLASS_NAMES = ['other', 'silence', 'siren']

CHANNELS = 1
RATE = 16000
CHUNK = 2048

# モデル入力長 (1.5秒)
BUFFER_SECONDS = 1.5
BUFFER_SAMPLES = int(RATE * BUFFER_SECONDS)

# 特徴量パラメータ
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# ==========================================
# 3. グローバル変数
# ==========================================
audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
data_queue = Queue()
buffer_lock = threading.Lock()
is_running = True
model = None

# ==========================================
# 4. 音声入力コールバック
# ==========================================
def audio_callback(indata, frames, time_info, status):
    if status:
        # エラー表示は改行してしまうため、デバッグ時以外はコメントアウトしても良い
        pass 
    data_queue.put(indata.copy())

# ==========================================
# 5. バッファ更新スレッド
# ==========================================
def buffer_update_thread():
    global audio_buffer, is_running
    print("[Info] Buffer update thread started.")
    
    while is_running:
        try:
            data_chunk = data_queue.get(timeout=1.0)
            data_chunk = data_chunk[:, 0]
            chunk_len = len(data_chunk)

            with buffer_lock:
                audio_buffer = np.roll(audio_buffer, -chunk_len)
                audio_buffer[-chunk_len:] = data_chunk
        except:
            continue

# ==========================================
# 6. 前処理関数
# ==========================================
def preprocess_for_model(audio_segment_float):
    melspec = librosa.feature.melspectrogram(
        y=audio_segment_float,
        sr=RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=20,
        fmax=8000,
        power=2.0
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    mean = np.mean(log_melspec)
    std = np.std(log_melspec)
    log_melspec = (log_melspec - mean) / (std + 1e-6)

    TARGET_WIDTH = 47
    current_width = log_melspec.shape[1]

    if current_width > TARGET_WIDTH:
        log_melspec = log_melspec[:, :TARGET_WIDTH]
    elif current_width < TARGET_WIDTH:
        pad_width = TARGET_WIDTH - current_width
        log_melspec = np.pad(log_melspec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    log_melspec = log_melspec.astype(np.float32)
    log_melspec = np.expand_dims(log_melspec, axis=-1)
    log_melspec = np.expand_dims(log_melspec, axis=0)

    return log_melspec

# ==========================================
# 7. 推論スレッド
# ==========================================
def inference_thread():
    global is_running, audio_buffer, model
    print("[Info] Inference thread started.")

    # クラスのインデックスを取得（表示用）
    idx_other   = CLASS_NAMES.index('other')
    idx_silence = CLASS_NAMES.index('silence')
    idx_siren   = CLASS_NAMES.index('siren')

    # 平滑化
    SMOOTHING_WINDOW = 5
    prediction_history = []
    
    # ヒステリシス
    SIREN_THRESHOLD = 0.97
    SIREN_ON_COUNT = 3
    SIREN_OFF_COUNT = 3
    
    siren_on_counter = 0
    siren_off_counter = 0
    siren_active = False

    while is_running:
        with buffer_lock:
            current_buffer = np.array(audio_buffer, dtype=np.float32)

        start_time = time.time()
        rms = np.sqrt(np.mean(current_buffer**2))

        try:
            processed_data = preprocess_for_model(current_buffer)
            raw_pred = model.predict(processed_data, verbose=0)[0]
        except Exception as e:
            continue

        prediction_history.append(raw_pred)
        if len(prediction_history) > SMOOTHING_WINDOW:
            prediction_history.pop(0)
        smoothed_preds = np.mean(prediction_history, axis=0)

        # 判定ロジック
        siren_prob = smoothed_preds[idx_siren]

        if siren_prob >= SIREN_THRESHOLD:
            siren_on_counter += 1
            siren_off_counter = 0
            if siren_on_counter >= SIREN_ON_COUNT:
                siren_active = True
        else:
            siren_off_counter += 1
            siren_on_counter = 0
            if siren_off_counter >= SIREN_OFF_COUNT:
                siren_active = False

        if siren_active:
            final_class = "siren"
            confidence = siren_prob * 100
        else:
            pred_id = np.argmax(smoothed_preds)
            final_class = CLASS_NAMES[pred_id]
            confidence = smoothed_preds[pred_id] * 100

        latency = time.time() - start_time

        # ★修正: \r を使って行頭に戻り、その場で更新する
        # 文字列末尾にスペースを入れて、前の長い文字が残らないようにする
        print(
            f"\rSiren: {smoothed_preds[idx_siren]*100:5.1f}% | "
            f"Other: {smoothed_preds[idx_other]*100:5.1f}% | "
            f"Silence: {smoothed_preds[idx_silence]*100:5.1f}% "
            f"| Vol: {rms:.3f} "
            f"| 判定: {final_class:7s} ({confidence:5.1f}%) "
            f"| 遅延: {latency:.3f}s    ", # ←末尾に空白パディング
            end="",
            flush=True
        )

        time.sleep(0.1)

# ==========================================
# 8. メイン関数
# ==========================================
def main():
    global model, is_running

    print("モデルを読み込んでいます...")
    try:
        model = load_model(MODEL_PATH, custom_objects={'ReduceSumLayer': ReduceSumLayer})
        print("モデル読み込み完了。")
    except Exception as e:
        print(f"[Fatal Error] {e}")
        sys.exit(1)

    try:
        with sd.InputStream(
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK,
            dtype='float32',
            callback=audio_callback
        ):
            print("\nリアルタイム分類を開始しました (Ctrl+Cで終了)...")

            t_update = threading.Thread(target=buffer_update_thread)
            t_update.daemon = True
            t_update.start()

            t_infer = threading.Thread(target=inference_thread)
            t_infer.daemon = True
            t_infer.start()

            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n停止します...")
    except Exception as e:
        print(f"\n[Error] {e}")
    finally:
        is_running = False
        time.sleep(0.5)
        print("終了しました。")

if __name__ == '__main__':
    main()