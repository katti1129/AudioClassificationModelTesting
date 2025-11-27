import sounddevice as sd
import numpy as np

duration = 3  # 秒
print("録音中...")
audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
print("録音完了:", audio.shape)
