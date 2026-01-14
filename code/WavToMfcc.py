import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- 1. 設定 ---
file_path = "./data/Ambulance-Siren01-2.wav"

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13

# --- 2. 音声ファイルの読み込み ---
y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

# --- 3. 各過程の計算とグラフ描画 ---

# グラフ1: 元の音声波形
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('1. Original Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ1: プリエンファシス
y_preemphasized = librosa.effects.preemphasis(y)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(y_preemphasized, sr=sr)
plt.title('2. Waveform after Pre-emphasis')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ2,3: フレーミング＋窓関数
frame_sample = y_preemphasized[10000:10000+N_FFT]
frame_windowed = frame_sample * np.hamming(N_FFT)

plt.figure(figsize=(14, 5))
plt.plot(frame_windowed)
plt.title('3. A Single Frame after Windowing')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ4: パワースペクトル
stft_frame = np.abs(librosa.stft(frame_windowed, n_fft=N_FFT))**2

plt.figure(figsize=(14, 5))
freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
plt.plot(freqs, stft_frame)
plt.title('4. Power Spectrum (of a single frame)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# ★★★ 追加部分：メルフィルタバンク ★★★

mel_filters = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)

plt.figure(figsize=(14, 6))
for i in range(0, N_MELS, 10):  # 多いので10枚ずつ表示
    plt.plot(freqs[:len(mel_filters[i])], mel_filters[i])
plt.title('5. Mel Filter Bank (subset)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ5: メルスペクトログラム
melspec = librosa.feature.melspectrogram(
    y=y_preemphasized,
    sr=sr,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

plt.figure(figsize=(10, 4))
librosa.display.specshow(melspec, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f')
plt.title('6. Mel Spectrogram')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ6: ログメルスペクトログラム
log_melspec = librosa.power_to_db(melspec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_melspec, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('7. Log-Mel Spectrogram')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ7,8: MFCC
mfccs = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    n_mfcc=N_MFCC
)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('8. MFCCs (13 dims)')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()
