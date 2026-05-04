"""from google.colab import drive
drive.mount('/content/gdrive')

!pip install -q transformers datasets evaluate jiwer librosa soundfile"""

import os
import numpy as np
import pandas as pd
import torch
import librosa
import evaluate
from dataclasses import dataclass
from typing import Any, List, Dict, Union

from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline,
)

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(
        "GPU RAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "GB",
    )

import psutil

print("CPU RAM:", round(psutil.virtual_memory().total / 1e9, 1), "GB")

DRIVE_ROOT = "/content/gdrive/MyDrive"
CLIPS_DIR = os.path.join(DRIVE_ROOT, "clips")
TRAIN_TSV = os.path.join(DRIVE_ROOT, "train")
TEST_TSV = os.path.join(DRIVE_ROOT, "test")
MODEL_NAME = "openai/whisper-small"
SAMPLE_LIMIT = 200


def tsv_to_hf_dataset(tsv_path, clips_dir, limit=None):
    df = pd.read_csv(tsv_path, sep="\t")[["path", "sentence"]]
    df["full_path"] = df["path"].apply(lambda x: os.path.join(clips_dir, x))
    # Dropping rows where audio file doesn't exist
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    ds = Dataset.from_dict(
        {"audio": df["full_path"].tolist(), "sentence": df["sentence"].tolist()}
    )
    # Casting audio column — HF will auto-resample to 16kHz for us
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


train_ds = tsv_to_hf_dataset(f"{TRAIN_TSV}.tsv", CLIPS_DIR, limit=SAMPLE_LIMIT)
test_ds = tsv_to_hf_dataset(f"{TEST_TSV}.tsv", CLIPS_DIR, limit=50)
print(train_ds, test_ds)

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language="Azerbaijani", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="Azerbaijani", task="transcribe"
)

"""Part A: Baseline Inference && WER/CER"""

# Quick baseline and preparation before fine-tuning stage
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

references, hypotheses = [], []
best, worst = [], []

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")

for sample in test_ds:
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]

    # Preprocess audio using the feature_extractor
    input_features = processor.feature_extractor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    # Move input_features to GPU if available
    if torch.cuda.is_available():
        input_features = input_features.to("cuda")

    # Generate prediction
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=processor.get_decoder_prompt_ids(
            language="Azerbaijani", task="transcribe"
        ),
    )

    # Decode prediction
    hyp = (
        processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        .strip()
        .lower()
    )
    ref = sample["sentence"].strip().lower()

    references.append(ref)
    hypotheses.append(hyp)

    sample_wer = wer_metric.compute(predictions=[hyp], references=[ref])
    best.append((sample_wer, ref, hyp))
    worst.append((sample_wer, ref, hyp))
baseline_wer = wer_metric.compute(predictions=hypotheses, references=references)
baseline_cer = cer_metric.compute(predictions=hypotheses, references=references)

print(f"Baseline WER: {baseline_wer*100:.2f}%")
print(f"Baseline CER: {baseline_cer*100:.2f}%")

best.sort(key=lambda x: x[0])
worst.sort(key=lambda x: x[0], reverse=True)

print("\n--- Best 5 samples ---")
for wer, ref, hyp in best[:5]:
    print(f"WER: {wer*100:.2f}% | REF: {ref} | HYP: {hyp}")

print("\n--- Worst 5 samples ---")
for wer, ref, hyp in worst[:5]:  # Corrected 'cer' to 'wer' here
    print(f"WER: {wer*100:.2f}% | REF: {ref} | HYP: {hyp}")
