# 事前に訓練された機械学習モデルを使用して，
# マイクから入力されるリアルタイムの音声を分類するためのスクリプト (sounddevice版)

import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import time
from queue import Queue
import threading

# --- 1. 設定・パラメータ ---
MODEL_PATH = 'model/best.keras'

CLASS_NAMES = ['siren', 'other', 'silence']

# オーディオ設定
CHUNK = 1024 * 2
CHANNELS = 1
RATE = 16000

# 推論設定
BUFFER_SECONDS = 1.0
BUFFER_SAMPLES = int(RATE * BUFFER_SECONDS)
INFERENCE_INTERVAL_SECONDS = 1.0  # 1秒ごとに推論を実行

# 特徴量パラメータ
N_MELS = 128
#N_FFT = 2048
N_FFT = 1024
HOP_LENGTH = 512

# 推論平滑化用
is_running = True
prediction_history = []

# ★追加: 音量ゲートのしきい値 (0.01〜0.05くらいで調整)
#SILENCE_THRESHOLD = 0.02



# --- 2. グローバル変数とキュー ---
audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)  # float32で扱う
data_queue = Queue()
run_flag = True
buffer_lock = threading.Lock()



# --- 3. オーディオ録音コールバック ---
def audio_callback(indata, frames, time_info, status):
    """sounddeviceが別スレッドで実行する関数．マイクデータをキューに入れる"""
    if status:
        print(status)
    # indataはnumpy(float32, shape=(frames, channels))
    data_queue.put(indata.copy())


# --- 4. 前処理関数 ---
def preprocess_for_model(audio_segment_float):
    melspec = librosa.feature.melspectrogram(
        y=audio_segment_float,
        sr=RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # shape (128, T)

    # モデル入力に合わせて (128, 126) に切る or padする
    if log_melspec.shape[1] > 126:
        log_melspec = log_melspec[:, :126]
    else:
        pad_width = 126 - log_melspec.shape[1]
        log_melspec = np.pad(log_melspec, ((0, 0), (0, pad_width)), mode='constant')

    # 軸を (128, 126, 1) に
    log_melspec = np.expand_dims(log_melspec, axis=-1)

    #log_melspec = np.repeat(log_melspec, 3, axis=-1)

    # batch次元を追加 → (1, 128, 126, 1)
    log_melspec = np.expand_dims(log_melspec, axis=0)

    #print(f"生成されたメルスペクトログラムの形状: {log_melspec.shape}")
    return log_melspec



# ===========================================
# 推論スレッド（平滑化 + 遅延 + 閾値 + ヒステリシス + ★RMSゲート）
# ===========================================
def inference_thread():
    global is_running, prediction_history, audio_buffer, model

    SMOOTHING_WINDOW = 5

    # しきい値 & ヒステリシス用
    SIREN_THRESHOLD = 0.97      # 97% 未満は siren確定としない
    SIREN_ON_COUNT  = 3         # 3連続でON確定
    SIREN_OFF_COUNT = 3         # 3連続でOFF確定

    # 状態管理
    siren_on_counter  = 0
    siren_off_counter = 0
    siren_active = False  # 現在サイレン状態中か

    prediction_history = []

    while is_running:
        if len(audio_buffer) == BUFFER_SAMPLES:

            # ===== 入力コピー =====
            buffer_copy = np.array(audio_buffer, dtype=np.float32)

            # ===== 遅延計測開始 =====
            start_time = time.time()

            # ===== ★追加: 音量(RMS)チェック =====
            rms = np.sqrt(np.mean(buffer_copy**2))

            if rms < SILENCE_THRESHOLD:
                # 音が小さすぎる場合 -> 強制的に Silence とする
                # CLASS_NAMES = ['siren', 'other', 'silence'] なので [0, 0, 1]
                raw_pred = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                
                # 計算負荷を下げるため，この回は前処理・推論をスキップする
                # print(f"Low Volume: {rms:.4f}") # デバッグ用
            else:
                # 音が十分ある場合 -> 通常通り AI に推論させる
                
                # ===== 前処理 =====
                processed_data = preprocess_for_model(buffer_copy)

                # ===== 生予測 =====
                raw_pred = model.predict(processed_data, verbose=0)[0]


            # ===== 平滑化 (強制Silenceの場合も履歴に含めることで滑らかに移行する) =====
            prediction_history.append(raw_pred)
            if len(prediction_history) > SMOOTHING_WINDOW:
                prediction_history.pop(0)
            smoothed_preds = np.mean(prediction_history, axis=0)

            # ===== ヒステリシス判定 =====
            siren_prob = smoothed_preds[CLASS_NAMES.index("siren")]

            if siren_prob >= SIREN_THRESHOLD:
                # sirenぽい結果が続いている
                siren_on_counter += 1
                siren_off_counter = 0
                if siren_on_counter >= SIREN_ON_COUNT:
                    siren_active = True
            else:
                # sirenらしくない結果が続いている
                siren_off_counter += 1
                siren_on_counter = 0
                if siren_off_counter >= SIREN_OFF_COUNT:
                    siren_active = False

            # ===== 表示用クラス確定 =====
            if siren_active:
                predicted_class = "siren"
                confidence = siren_prob * 100
            else:
                # sirenがアクティブでない場合は他クラス最大
                predicted_class_id = np.argmax(smoothed_preds)
                predicted_class = CLASS_NAMES[predicted_class_id]
                confidence = smoothed_preds[predicted_class_id] * 100

            # ===== 遅延測定完了 =====
            latency = time.time() - start_time

            # ===== 1行上書き表示 =====
            # RMS値も表示しておくと調整しやすいです
            print(
                f"\rsiren: {smoothed_preds[0]*100:.2f}% | "
                f"other: {smoothed_preds[1]*100:.2f}% | "
                f"silence: {smoothed_preds[2]*100:.2f}% "
                f"| Vol: {rms:.3f} "  # ★音量を表示
                f"| 状態: {'ON' if siren_active else 'OFF'} "
                f"| 確定: {predicted_class} ({confidence:.2f}%) "
                f"| 遅延: {latency:.3f} 秒 ",
                end="",
                flush=True
            )

        else:
            time.sleep(0.01)



# --- 6. メイン処理 ---
def main():
    global audio_buffer, run_flag,model

    print("モデルを読み込んでいます...")
    global model
    model = load_model(MODEL_PATH)
    print("モデルの読み込み完了．")

    # sounddevice の入力ストリーム開始
    with sd.InputStream(
        channels=CHANNELS,
        samplerate=RATE,
        blocksize=CHUNK,
        dtype='float32',
        callback=audio_callback
    ):
        print("\nリアルタイム分類を開始しました... (Ctrl+Cで終了)")

        # 推論スレッド開始
        infer_thread = threading.Thread(target=inference_thread)
        infer_thread.daemon = True
        infer_thread.start()

        try:
            while run_flag:
                data_chunk = data_queue.get()
                if data_chunk is None:
                    break

                data_chunk = data_chunk[:, 0]  # mono取り出し

                with buffer_lock:
                    chunk_len = len(data_chunk)
                    audio_buffer = np.roll(audio_buffer, -chunk_len)
                    audio_buffer[-chunk_len:] = data_chunk

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nCtrl+Cを検出しました．プログラムを終了します．")
        finally:
            run_flag = False
            if infer_thread.is_alive():
                infer_thread.join(timeout=1)
            print("終了しました．")

is_running = True
if __name__ == '__main__':
    main()
