# 01_prepare_and_segment_dataset.py (修正版)
#音声認識や音声分類のような機械学習モデルを訓練するためのデータセットを準備するスクリプト

import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# --- 設定 ---
SOURCE_DATA_DIR = Path('./data')
OUTPUT_DIR = Path('./segmented_data_split') # 出力先を新しいフォルダに

SEGMENT_DURATION_SEC = 1.5
HOP_DURATION_SEC = 0.5
SAMPLE_RATE = 16000
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION_SEC)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION_SEC)

# --- 1. 元ファイルのリストを作成 ---
all_files = []
all_labels = []
CLASS_NAMES = sorted([p.name for p in SOURCE_DATA_DIR.iterdir() if p.is_dir()])
for i, class_name in enumerate(CLASS_NAMES):
    class_dir = SOURCE_DATA_DIR / class_name
    for file_path in class_dir.glob('*.wav'):
        all_files.append(file_path)
        all_labels.append(i)

# --- 2. 元ファイルのリストを3つに分割 ---
print("元のファイルを訓練・検証・テスト用に分割します．．．")
train_val_files, test_files, train_val_labels, _ = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)
train_files, val_files, _, _ = train_test_split(
    train_val_files, train_val_labels, test_size=0.2, random_state=42, stratify=train_val_labels
)

# --- 3. 分割後の各グループに対してセグメント化を実行 ---
OUTPUT_DIR.mkdir(exist_ok=True)
print("分割後の各データセットに対してセグメントを作成します．．．")

def create_segments(file_list, dataset_type):
    """指定されたファイルリストからセグメントを作成する関数"""
    output_set_dir = OUTPUT_DIR / dataset_type
    output_set_dir.mkdir(exist_ok=True)

    for wav_file in file_list:
        class_name = wav_file.parent.name
        output_class_dir = output_set_dir / class_name
        output_class_dir.mkdir(exist_ok=True)

        try:
            wav, _ = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
            num_segments = (len(wav) - SEGMENT_SAMPLES) // HOP_SAMPLES + 1
            if num_segments <= 0: continue

            for i in range(num_segments):
                start = i * HOP_SAMPLES
                end = start + SEGMENT_SAMPLES
                segment = wav[start:end]

                output_filename = f"{wav_file.stem}_seg{i}.wav"
                output_path = output_class_dir / output_filename
                sf.write(output_path, segment, SAMPLE_RATE)
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

create_segments(train_files, 'train')
create_segments(val_files, 'validation')
create_segments(test_files, 'test')

print("\nセグメントの作成が完了しました．")