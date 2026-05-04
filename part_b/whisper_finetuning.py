"""Part B: Prepare Dataset for Fine-Tuning"""
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

train_ds = train_ds.map(prepare_dataset, remove_columns=["audio", "sentence"])
test_ds = test_ds.map(prepare_dataset, remove_columns=["audio", "sentence"])

"""Data Collator"""
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # Replacing padding token id with -100 so loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

"""Computing Metrics; correct signature && real edit-distance WER"""
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    labels_ids = pred.label_ids

    # Replacing -100 padding back to pad token id before decoding
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=labels_str)
    cer = cer_metric.compute(predictions=pred_str, references=labels_str)
    return {"wer": round(wer, 4), "cer": round(cer, 4)}

import gc

# Clear baseline model from GPU before fine-tuning
gc.collect()
torch.cuda.empty_cache()

# Reinitialize model fresh for training
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="Azerbaijani", task="transcribe"
)

# Freeze encoder — only train decoder, cuts memory ~40%
model.freeze_encoder()

print("Memory cleared, model ready for fine-tuning")
print(f"GPU memory free: {torch.cuda.memory_reserved(0)/1e9:.1f} GB reserved")

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/gdrive/MyDrive/whisper-az-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    max_steps=400,
    warmup_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=50, # Corrected from 'logging_step'
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,         # ← fixes the UserWarning you saw
    save_total_limit=2,                  # ← only keep 2 checkpoints, saves disk space
    gradient_checkpointing=True          # trades speed for memory
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model("./whisper-az-finetuned/best")

# Reloading audio-intact test set for inference
test_ds_audio = tsv_to_hf_dataset(f"{TEST_TSV}.tsv", CLIPS_DIR, limit=50)
!mkdir -p /content/gdrive/MyDrive/results

import os
import torch
import matplotlib.pyplot as plt
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor

# ==========================================
# 1. FIXING THE PLOTS
# ==========================================
log_history = trainer.state.log_history
train_steps, train_losses = [], []
eval_steps, eval_wers = [], []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["loss"])
    if "eval_wer" in entry:
        eval_steps.append(entry["step"])
        eval_wers.append(entry["eval_wer"] * 100)

# Debug warning if training logs are missing
if not train_steps:
    print("⚠️ Warning: No training loss found in log_history. Check 'logging_steps' in your TrainingArguments.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
if train_steps:
    ax1.plot(train_steps, train_losses, color="#185FA5")
ax1.set_title("Training Loss")
ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")

if eval_steps:
    ax2.plot(eval_steps, eval_wers, color="#1D9E75", marker="o")
ax2.set_title("Validation WER per evaluation step")
ax2.set_xlabel("Step"); ax2.set_ylabel("WER (%)")

plt.tight_layout()

# Safely create the results directory if it doesn't exist yet
os.makedirs("/content/gdrive/MyDrive/results", exist_ok=True)
plt.savefig("/content/gdrive/MyDrive/results/training_curves.png", dpi=150)
plt.show()

# ==========================================
# 2. FIXING THE OSERROR (Model Loading)
# ==========================================
target_model_path = "/content/gdrive/MyDrive/whisper-az-finetuned/best"

# Check if your hardcoded 'best' path actually exists and has a config
if not os.path.exists(os.path.join(target_model_path, "config.json")):
    print(f"⚠️ Could not find config.json in {target_model_path}.")

    # Fallback to the trainer's internal best checkpoint path
    if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
        print(f"✅ Falling back to actual best checkpoint: {trainer.state.best_model_checkpoint}")
        target_model_path = trainer.state.best_model_checkpoint
    else:
        raise FileNotFoundError(
            f"Cannot find valid model files in {target_model_path} and no best_checkpoint found in trainer. "
            "Did you save the model?"
        )

# Explicitly load the fine-tuned model from the verified local path
ft_model = WhisperForConditionalGeneration.from_pretrained(target_model_path, local_files_only=True)

# Use the globally defined 'processor' for tokenizer and feature_extractor
# as its components are typically not fine-tuned and are compatible.
ft_pipeline = pipeline(
    "automatic-speech-recognition",
    model=ft_model,
    tokenizer=processor.tokenizer,           # Use the original processor's tokenizer
    feature_extractor=processor.feature_extractor, # Use the original processor's feature_extractor
    generate_kwargs={"language": "az", "task": "transcribe"},
    device=0 if torch.cuda.is_available() else -1
)

print("Starting inference on test dataset...")
ft_hyps = [
    ft_pipeline({"array": s["audio"]["array"], "sampling_rate": 16000})["text"].strip().lower()
    for s in test_ds_audio
]

# Calculate final metrics
ft_wer = wer_metric.compute(predictions=ft_hyps, references=references)
ft_cer = cer_metric.compute(predictions=ft_hyps, references=references)

print(f"\n{'Metric':<8} {'Baseline':>12} {'Fine-tuned':>12} {'Δ':>8}")
print("-" * 44)
print(f"{'WER':<8} {baseline_wer*100:>11.2f}% {ft_wer*100:>11.2f}% {(ft_wer-baseline_wer)*100:>+7.2f}%")
print(f"{'CER':<8} {baseline_cer*100:>11.2f}% {ft_cer*100:>11.2f}% {(ft_cer-baseline_cer)*100:>+7.2f}%")



print(f"Starting manual inference using model from: {target_model_path}")

ft_model = WhisperForConditionalGeneration.from_pretrained(target_model_path, local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu")

ft_hyps = []
print("Processing audio files manually...")

for i, s in enumerate(test_ds_audio):
    # 1. Manually extract features
    input_features = processor(s["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(ft_model.device)

    # 2. Manually generate tokens
    with torch.no_grad():
        predicted_ids = ft_model.generate(input_features, language="azerbaijani", task="transcribe")

    # 3. Decode to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    ft_hyps.append(transcription.strip().lower())

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(test_ds_audio)} samples")

ft_wer = wer_metric.compute(predictions=ft_hyps, references=references)
ft_cer = cer_metric.compute(predictions=ft_hyps, references=references)

print(f"\n{'METRIC':<15} {'BASELINE':<15} {'FINE-TUNED':<15} {'IMPROVEMENT'}")
print("-" * 60)
print(f"{'WER':<15} {baseline_wer*100:>13.2f}% {ft_wer*100:>13.2f}% {(baseline_wer - ft_wer)*100:>10.2f}%")
print(f"{'CER':<15} {baseline_cer*100:>13.2f}% {ft_cer*100:>13.2f}% {(baseline_cer - ft_cer)*100:>10.2f}%")