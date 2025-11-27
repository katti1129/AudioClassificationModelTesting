from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# モデル名
model_name = "MIT/ast-finetuned-audioset-10-10-0.945"

# 特徴抽出器とモデルをロード
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    num_labels=len(labels), # 分類したいクラスの数を指定
    label2id={label: i for i, label in enumerate(labels)},
    id2label={i: label for i, label in enumerate(labels)},
    ignore_mismatched_sizes=True, # 事前学習モデルの出力層を新しいクラス数で置き換えるために必要
)