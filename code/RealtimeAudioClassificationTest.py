#事前に訓練された機械学習モデル（best_yamnet_rnn_model.keras）を使用して，
# マイクから入力されるリアルタイムの音声を分類するためのスクリプト

import numpy as np
import pyaudio
import librosa
from tensorflow.keras.models import load_model
import time
from queue import Queue
import threading

# --- 1. 設定・パラメータ ---
MODEL_PATH = 'model/best_resnet_crnn_model.keras'

#'model/best_yamnet_rnn_model.keras'  転移学習のモデル
#'model/best_crnn_segmented_model.keras' CNN+RNNのモデル
#'model/best_rnn_segmented_model.keras' MFCC+RNNのモデル

CLASS_NAMES = ['0_EmergencyBell', '1_ambulance', '2_RailroadCrossing', '3_ClockAlarm', '4_PoliceCar', '5_Horn',
               '6_FireEngine', '7_Other']

# オーディオ設定
CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
INPUT_DEVICE_INDEX = 4

# 推論設定
BUFFER_SECONDS = 1.5
BUFFER_SAMPLES = int(RATE * BUFFER_SECONDS)
INFERENCE_INTERVAL_SECONDS = 1.0  # 1秒ごとに推論を実行

# 特徴量パラメータ
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# --- 2. グローバル変数とキュー ---
audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.int16)
data_queue = Queue()
run_flag = True
buffer_lock = threading.Lock()  # ★変更点: バッファアクセス保護用のロックを追加


# --- 3. オーディオ録音コールバック ---
def audio_callback(in_data, frame_count, time_info, status):
    """pyaudioが別スレッドで実行する関数．マイクデータをキューに入れるだけ．"""
    data_queue.put(in_data)
    return (None, pyaudio.paContinue)


# --- 4. 前処理関数 ---
def preprocess_for_model(audio_segment_float):
    """AIモデル入力用のメルスペクトログラムを生成"""
    melspec = librosa.feature.melspectrogram(y=audio_segment_float, sr=RATE, n_mels=N_MELS, n_fft=N_FFT,
                                             hop_length=HOP_LENGTH)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    # ★修正点: (周波数, 時間) の形状を (時間, 周波数) に転置
    log_melspec_T = np.transpose(log_melspec)
    # 軸を追加
    log_melspec_T = np.expand_dims(log_melspec_T, axis=0)
    log_melspec_T = np.expand_dims(log_melspec_T, axis=-1)

    # 修正後の形状を確認
    print(f"生成されたメルスペクトログラムの形状: {log_melspec_T.shape}")

    return log_melspec_T

    #log_melspec = np.expand_dims(log_melspec, axis=0)
    #log_melspec = np.expand_dims(log_melspec, axis=-1)
    #print(f"生成されたメルスペクトログラムの形状: {log_melspec.shape}")


    return log_melspec


# --- ★追加点: 推論スレッド用の関数 ---
def inference_thread(model):
    """AI推論を定期的に実行するスレッド"""
    global audio_buffer, run_flag

    while run_flag:
        # 推論間隔を空ける
        time.sleep(INFERENCE_INTERVAL_SECONDS)

        # 現在のオーディオバッファのコピーを作成
        with buffer_lock:
            buffer_copy = np.copy(audio_buffer)

        # --- ここから推論処理 ---
        # 1. 前処理
        audio_float = buffer_copy.astype(np.float32) / 32768.0
        processed_data = preprocess_for_model(audio_float)

        # 2. モデルで予測
        prediction = model.predict(processed_data, verbose=0)

        # 3. 結果を表示
        predicted_class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # キューに溜まっているデータチャンクの数を表示
        queue_size = data_queue.qsize()

        print(
            f"\r予測クラス: {CLASS_NAMES[predicted_class_id]} | 確信度: {confidence:.2f}% | Queue Size: {queue_size}   ",
            end="")


# --- 5. メイン処理 ---
def main():
    global audio_buffer, run_flag

    # モデルのロード
    print("モデルを読み込んでいます...")
    model = load_model(MODEL_PATH)
    print("モデルの読み込み完了．")

    # PyAudioの初期化とストリームの開始
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    stream.start_stream()
    print("\nリアルタイム分類を開始しました... (Ctrl+Cで終了)")

    # ★変更点: 推論スレッドを開始
    infer_thread = threading.Thread(target=inference_thread, args=(model,))
    infer_thread.daemon = True
    infer_thread.start()

    try:
        while run_flag:
            # キューから最新のオーディオデータを取得
            data_chunk_bytes = data_queue.get()
            if data_chunk_bytes is None:
                break
            data_chunk_int16 = np.frombuffer(data_chunk_bytes, dtype=np.int16)

            # ステレオ(L, R, L, R...)の音声から左チャンネル(L)だけを抽出する
            #data_chunk_int16 = data_chunk_int16[::2]

            # ★変更点: ロックをかけてバッファを安全に更新
            with buffer_lock:
                chunk_len = len(data_chunk_int16)
                audio_buffer = np.roll(audio_buffer, -chunk_len)
                audio_buffer[-chunk_len:] = data_chunk_int16

            # CPU負荷を抑えるための非常に短い待機（任意）
            time.sleep(0.01)


    except KeyboardInterrupt:
        print("\nCtrl+Cを検出しました．プログラムを終了します．")
    finally:
        run_flag = False
        print("ストリームを停止中...")
        if 'infer_thread' in locals() and infer_thread.is_alive():
            infer_thread.join(timeout=1)  # スレッドの終了を待つ
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("終了しました．")


if __name__ == '__main__':
    main()