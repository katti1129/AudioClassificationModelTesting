import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import time
from queue import Queue
import threading

# --- 1. 設定 ---
RATE = 16000
CHUNK = 1024 * 2
CHANNELS = 1
BUFFER_SECONDS = 1.0
BUFFER_SAMPLES = int(RATE * BUFFER_SECONDS)
INFERENCE_INTERVAL_SECONDS = 1.0

# --- 2. グローバル変数 ---
audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
data_queue = Queue()
run_flag = True
buffer_lock = threading.Lock()

# --- 3. モデルの読み込み ---
print("YAMNetモデルを読み込み中...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
import csv
with open(class_map_path) as f:
    reader = csv.reader(f)
    next(reader)
    CLASS_NAMES = [row[2] for row in reader]
print(f"クラス数: {len(CLASS_NAMES)}")
print("YAMNetの読み込み完了．")

# --- 4. オーディオコールバック ---
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    data_queue.put(indata.copy())

# --- 5. 推論スレッド ---
def inference_thread():
    global audio_buffer, run_flag

    while run_flag:
        time.sleep(INFERENCE_INTERVAL_SECONDS)

        with buffer_lock:
            buffer_copy = np.copy(audio_buffer)

        # YAMNetは float32 waveform [-1.0, +1.0]
        waveform = buffer_copy.astype(np.float32)

        # 推論開始時間
        start_time = time.time()
        scores, embeddings, spectrogram = yamnet_model(waveform)
        mean_scores = tf.reduce_mean(scores, axis=0)
        top_class = int(tf.argmax(mean_scores))
        top_score = float(mean_scores[top_class]) * 100
        top_label = CLASS_NAMES[top_class]

        latency = time.time() - start_time
        print(f"\r予測: {top_label} | 確信度: {top_score:.2f}% | 遅延: {latency:.3f} 秒 | Queue Size: {data_queue.qsize()}   ", end="")

# --- 6. メイン処理 ---
def main():
    global audio_buffer, run_flag

    with sd.InputStream(
        channels=CHANNELS,
        samplerate=RATE,
        blocksize=CHUNK,
        dtype='float32',
        callback=audio_callback
    ):
        print("\nリアルタイムYAMNet分類を開始しました... (Ctrl+Cで終了)")

        infer_thread = threading.Thread(target=inference_thread)
        infer_thread.daemon = True
        infer_thread.start()

        try:
            while run_flag:
                data_chunk = data_queue.get()
                if data_chunk is None:
                    break

                data_chunk = data_chunk[:, 0]  # モノラル化

                with buffer_lock:
                    chunk_len = len(data_chunk)
                    audio_buffer = np.roll(audio_buffer, -chunk_len)
                    audio_buffer[-chunk_len:] = data_chunk

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nCtrl+Cで停止しました．")
        finally:
            run_flag = False
            if infer_thread.is_alive():
                infer_thread.join(timeout=1)
            print("終了しました．")


if __name__ == '__main__':
    main()
