import pyaudio

# --- 設定 ---
FORMAT = pyaudio.paInt16
CHANNELS = 2  # ステレオで試す
RATE = 16000
INPUT_DEVICE_INDEX = 4 # MacBook Proのマイク
CHUNK = 1024

# --- テスト実行 ---
p = pyaudio.PyAudio()

try:
    print(f"デバイス {INPUT_DEVICE_INDEX} を，チャンネル数 {CHANNELS} で開きます...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

    print("\n✅ 成功: マイクのストリームを正常に開けました．")

    stream.stop_stream()
    stream.close()

except OSError as e:
    print(f"\n❌ 失敗: エラーが発生しました．")
    print(f"エラー内容: {e}")

finally:
    p.terminate()