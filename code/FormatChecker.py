"""
チャンネル数: 1, サンプルレート: 16000 Hz -> ✅ 対応しています (Supported)
チャンネル数: 2, サンプルレート: 16000 Hz -> ✅ 対応しています (Supported)
チャンネル数: 1, サンプルレート: 22050 Hz -> ✅ 対応しています (Supported)
チャンネル数: 2, サンプルレート: 22050 Hz -> ✅ 対応しています (Supported)
チャンネル数: 1, サンプルレート: 44100 Hz -> ✅ 対応しています (Supported)
チャンネル数: 2, サンプルレート: 44100 Hz -> ✅ 対応しています (Supported)
チャンネル数: 1, サンプルレート: 48000 Hz -> ✅ 対応しています (Supported)
チャンネル数: 2, サンプルレート: 48000 Hz -> ✅ 対応しています (Supported)
"""



import pyaudio

# --- チェックする設定 ---
INPUT_DEVICE_INDEX = 4
RATES_TO_CHECK = [16000, 22050, 44100, 48000]
CHANNELS_TO_CHECK = [1, 2]
FORMAT = pyaudio.paInt16

# --- チェック実行 ---
p = pyaudio.PyAudio()

print(f"デバイス {INPUT_DEVICE_INDEX} ({p.get_device_info_by_index(INPUT_DEVICE_INDEX)['name']}) の対応フォーマットをチェックします...")
print("-" * 40)

supported_formats = []

for rate in RATES_TO_CHECK:
    for channels in CHANNELS_TO_CHECK:
        try:
            is_supported = p.is_format_supported(
                rate,
                input_device=INPUT_DEVICE_INDEX,
                input_channels=channels,
                input_format=FORMAT)

            if is_supported:
                status = "✅ 対応しています (Supported)"
                supported_formats.append((channels, rate))
            else:
                # is_format_supportedがFalseを返す場合
                status = "❌ 対応していません (Unsupported)"

        except ValueError:
            # is_format_supportedが例外を投げる場合
            status = "❌ 対応していません (Unsupported - ValueError)"

        print(f"チャンネル数: {channels}, サンプルレート: {rate} Hz -> {status}")

print("-" * 40)
if supported_formats:
    print("利用可能な設定の組み合わせが見つかりました．")
else:
    print("利用可能な設定の組み合わせが見つかりませんでした．")

p.terminate()