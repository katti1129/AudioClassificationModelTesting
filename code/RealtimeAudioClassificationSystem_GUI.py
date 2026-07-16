#推論結果をリアルタイムで表示する音声分類プログラム
#vol.判定ありバージョン
#other/sirenの2クラス分類
#ReSpeaker USB Mic Array v2.0 の DOA(Direction of Arrival)表示対応版
#PySide6 によるリアルタイム通知ウィンドウ表示対応版

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
import usb.core
from tuning import Tuning

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
import signal

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

# 学習時のアルファベット順に合わせる
CLASS_NAMES = ['other', 'siren']

CHANNELS = 1
RATE = 16000
CHUNK = 2048

# モデル入力長 (1.0秒)
BUFFER_SECONDS = 1.0
BUFFER_SAMPLES = int(RATE * BUFFER_SECONDS)

# 特徴量パラメータ
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# RMS 無音判定閾値（環境に応じて調整）
RMS_THRESHOLD = 0.012

# ReSpeaker USB Mic Array v2.0
RESPEAKER_VENDOR_ID = 0x2886
RESPEAKER_PRODUCT_ID = 0x0018
RESPEAKER_AUDIO_NAME = "ReSpeaker"

# ==========================================
# 3. グローバル変数
# ==========================================
audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
data_queue = Queue()
buffer_lock = threading.Lock()
direction_lock = threading.Lock()
gui_lock = threading.Lock()
is_running = True
model = None
mic = None
current_direction = -1

# GUI表示用の共有変数
gui_final_class = "silence"
gui_confidence = 0.0
gui_rms = 0.0
gui_latency = 0.0

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
# 6. ReSpeaker 初期化・音声入力デバイス選択・DOA取得
# ==========================================
def find_respeaker_audio_device():
    """sounddeviceからReSpeakerの入力デバイス番号を取得する。"""
    devices = sd.query_devices()
    input_devices = []
    matching_devices = []

    for index, device in enumerate(devices):
        if int(device["max_input_channels"]) <= 0:
            continue

        device_name = str(device["name"])
        input_devices.append(f"  [{index}] {device_name}")

        if RESPEAKER_AUDIO_NAME.casefold() in device_name.casefold():
            matching_devices.append((index, device))

    if not matching_devices:
        available = "\n".join(input_devices) or "  （入力デバイスなし）"
        raise RuntimeError(
            "ReSpeakerの音声入力デバイスが見つかりません。\n"
            f"利用可能な入力デバイス:\n{available}"
        )

    # 同名候補が複数ある場合は、利用可能な入力チャンネル数が多いものを選ぶ。
    device_index, device_info = max(
        matching_devices,
        key=lambda item: int(item[1]["max_input_channels"]),
    )

    sd.check_input_settings(
        device=device_index,
        channels=CHANNELS,
        dtype="float32",
        samplerate=RATE,
    )
    print(
        f"ReSpeaker Audio Input: [{device_index}] {device_info['name']} "
        f"({RATE} Hz, {CHANNELS} ch)"
    )
    return device_index


def initialize_respeaker():
    global current_direction

    print("ReSpeaker USB Mic Array v2.0 を確認しています...")
    try:
        dev = usb.core.find(idVendor=RESPEAKER_VENDOR_ID, idProduct=RESPEAKER_PRODUCT_ID)

        if dev is None:
            print("ReSpeaker Not Found")
            with direction_lock:
                current_direction = -1
            return None

        respeaker_mic = Tuning(dev)
        print(f"ReSpeaker Version: {respeaker_mic.version}")
        return respeaker_mic

    except Exception as e:
        print(f"[Warning] ReSpeaker initialization failed: {e}")
        with direction_lock:
            current_direction = -1
        return None

def doa_update_thread(respeaker_mic):
    global current_direction, is_running
    print("[Info] DOA update thread started.")

    while is_running:
        try:
            direction = respeaker_mic.direction
            if direction is None:
                direction = -1

            with direction_lock:
                current_direction = int(direction)

        except Exception:
            # DOA取得に失敗してもCRNN推論は止めない
            with direction_lock:
                current_direction = -1

        time.sleep(0.05)  # 約20Hz

def get_current_direction_text():
    with direction_lock:
        direction = current_direction

    if direction < 0:
        return "Unknown(-1)"
    return f"{direction}°"

def get_current_direction_value():
    with direction_lock:
        return current_direction

# ==========================================
# 7. GUI用ヘルパー
# ==========================================
def doa_to_direction_label(direction):
    if direction is None or direction < 0:
        return "Unknown"

    angle = direction % 360

    if angle >= 337.5 or angle < 22.5:
        return "← 左"
    elif angle < 67.5:
        return "↙ 左下"
    elif angle < 112.5:
        return "↓ 下"
    elif angle < 157.5:
        return "↘ 右下"
    elif angle < 202.5:
        return "→ 右"
    elif angle < 247.5:
        return "↗ 右上"
    elif angle < 292.5:
        return "↑ 前"
    else:
        return "↖ 左上"

def update_gui_state(final_class, confidence, rms, latency):
    global gui_final_class, gui_confidence, gui_rms, gui_latency

    with gui_lock:
        gui_final_class = final_class
        gui_confidence = float(confidence)
        gui_rms = float(rms)
        gui_latency = float(latency)

def get_gui_state():
    with gui_lock:
        final_class = gui_final_class
        confidence = gui_confidence
        rms = gui_rms
        latency = gui_latency

    direction = get_current_direction_value()
    return final_class, confidence, rms, latency, direction

# ==========================================
# a. 終了関数
# ==========================================

def shutdown(app=None):
    global is_running
    print("\n停止します...")
    is_running = False

    if app is not None:
        app.quit()

# ==========================================
# 8. GUIクラス
# ==========================================
class NotificationWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ambulance Notification")
        self.setFixedSize(400, 220)
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.container = QFrame(self)
        self.container.setObjectName("notificationContainer")
        self.container.setStyleSheet("""
            QFrame#notificationContainer {
                background-color: rgba(0, 0, 0, 120);
                border-radius: 22px;
            }
            QLabel {
                color: white;
                background: transparent;
            }
        """)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(8)

        self.title_label = QLabel("🚑 Ambulance")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(self.title_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(5)
        layout.addLayout(grid)

        self.direction_title = QLabel("方向")
        self.direction_value = QLabel("Unknown")
        self.confidence_title = QLabel("信頼値")
        self.confidence_value = QLabel("0.0 %")
        self.status_title = QLabel("状態")
        self.status_value = QLabel("待機中")
        self.volume_title = QLabel("音量")
        self.volume_value = QLabel("0.000")
        self.inference_title = QLabel("推論時間")
        self.inference_value = QLabel("0.0 ms")

        title_font = QFont("Arial", 12, QFont.Bold)
        value_font = QFont("Arial", 13, QFont.Bold)

        for label in [
            self.direction_title,
            self.confidence_title,
            self.status_title,
            self.volume_title,
            self.inference_title,
        ]:
            label.setFont(title_font)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        for label in [
            self.direction_value,
            self.confidence_value,
            self.status_value,
            self.volume_value,
            self.inference_value,
        ]:
            label.setFont(value_font)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        grid.addWidget(self.direction_title, 0, 0)
        grid.addWidget(self.direction_value, 0, 1)
        grid.addWidget(self.confidence_title, 1, 0)
        grid.addWidget(self.confidence_value, 1, 1)
        grid.addWidget(self.status_title, 2, 0)
        grid.addWidget(self.status_value, 2, 1)
        grid.addWidget(self.volume_title, 3, 0)
        grid.addWidget(self.volume_value, 3, 1)
        grid.addWidget(self.inference_title, 4, 0)
        grid.addWidget(self.inference_value, 4, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_display)
        self.timer.start(100)

        self.move_to_screen_center()

    def move_to_screen_center(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return

        screen_geometry = screen.availableGeometry()
        x = screen_geometry.x() + (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.y() + (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def refresh_display(self):
        final_class, confidence, rms, latency, direction = get_gui_state()

        direction_label = doa_to_direction_label(direction)
        if direction >= 0:
            direction_label = f"{direction_label}  {direction % 360}°"

        if final_class == "siren":
            status = "サイレンです！注意してください"
        elif final_class == "silence":
            status = "無音"
        else:
            status = "その他"

        self.direction_value.setText(direction_label)
        self.confidence_value.setText(f"{confidence:.1f} %")
        self.status_value.setText(status)
        self.volume_value.setText(f"{rms:.3f}")
        self.inference_value.setText(f"{latency * 1000.0:.1f} ms")

# ==========================================
# 9. 前処理関数
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

    #TARGET_WIDTH = 47 1.5sの場合
    TARGET_WIDTH = 32  # 1.0sの場合
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
# 10. 推論スレッド
# ==========================================
def inference_thread():
    global is_running, audio_buffer, model
    print("[Info] Inference thread started.")

    # クラスのインデックスを取得（表示用）
    idx_other   = CLASS_NAMES.index('other')
    #idx_silence = CLASS_NAMES.index('silence')
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

        # ===== RMS 計算 =====
        rms = np.sqrt(np.mean(current_buffer ** 2))

        # ===== 無音判定：推論しない =====
        if rms < RMS_THRESHOLD:
            final_class = "silence"
            confidence = 100.0
            latency = time.time() - start_time
            doa_text = get_current_direction_text()

            update_gui_state(final_class, confidence, rms, latency)

            print(
                f"\rSiren:   0.0% | "
                f"Other:   0.0% | "
                f"Silence:100.0% "
                f"| DOA:{doa_text} "
                f"| Vol: {rms:.3f}    ",
                end="",
                flush=True
            )
            time.sleep(0.1)
            continue
        # ===== 閾値を超えたらここから推論 =====
        processed_data = preprocess_for_model(current_buffer)
        raw_pred = model.predict(processed_data, verbose=0)[0]

        # --- 平滑化 ---
        prediction_history.append(raw_pred)
        if len(prediction_history) > SMOOTHING_WINDOW:
            prediction_history.pop(0)
        smoothed_preds = np.mean(prediction_history, axis=0)

        # --- クラス確率 ---
        siren_prob = smoothed_preds[idx_siren]

        # --- サイレンのヒステリシス判定 ---
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

        # --- 最終判定 ---
        if siren_active:
            final_class = "siren"
            confidence = siren_prob * 100
        else:
            pred_id = np.argmax(smoothed_preds)
            final_class = CLASS_NAMES[pred_id]
            confidence = smoothed_preds[pred_id] * 100

        latency = time.time() - start_time
        doa_text = get_current_direction_text()

        update_gui_state(final_class, confidence, rms, latency)

        print(
            f"\rSiren: {smoothed_preds[idx_siren]*100:5.1f}% | "
            f"Other: {smoothed_preds[idx_other]*100:5.1f}% "
            f"| DOA:{doa_text} "
            f"| Vol: {rms:.3f} "
            f"| 遅延: {latency:.3f}s    ",
            end="",
            flush=True
        )

        time.sleep(0.1)



# ==========================================
# 11. メイン関数
# ==========================================
def main():
    global model, is_running, mic

    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, lambda sig, frame: shutdown(app))

    print("モデルを読み込んでいます...")
    try:
        model = load_model(MODEL_PATH, custom_objects={'ReduceSumLayer': ReduceSumLayer})
        print("モデル読み込み完了。")
    except Exception as e:
        print(f"[Fatal Error] {e}")
        sys.exit(1)

    mic = initialize_respeaker()
    if mic is None:
        print("[Fatal Error] ReSpeakerのDOA機能を初期化できませんでした。")
        sys.exit(1)

    try:
        respeaker_audio_device = find_respeaker_audio_device()
    except Exception as e:
        print(f"[Fatal Error] ReSpeakerの音声入力を使用できません: {e}")
        sys.exit(1)

    def stop_threads():
        global is_running
        is_running = False

    app.aboutToQuit.connect(stop_threads)

    try:
        with sd.InputStream(
            device=respeaker_audio_device,
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK,
            dtype='float32',
            callback=audio_callback
        ):
            print("\nリアルタイム分類を開始しました (Ctrl+Cまたはウィンドウを閉じると終了)...")

            t_update = threading.Thread(target=buffer_update_thread)
            t_update.daemon = True
            t_update.start()

            if mic is not None:
                t_doa = threading.Thread(target=doa_update_thread, args=(mic,))
                t_doa.daemon = True
                t_doa.start()

            t_infer = threading.Thread(target=inference_thread)
            t_infer.daemon = True
            t_infer.start()

            window = NotificationWindow()
            window.show()

            sigint_timer = QTimer()
            sigint_timer.timeout.connect(lambda: None)
            sigint_timer.start(100)

            exit_code = app.exec()
            is_running = False
            time.sleep(0.5)
            sys.exit(exit_code)

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
