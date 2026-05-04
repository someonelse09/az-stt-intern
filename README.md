# Azerbaijan Speech-to-Text (STT) Internship Task

This repository contains a technical evaluation and fine-tuning of the **Whisper-small** model for the Azerbaijani language using the Mozilla Common Voice 17.0 dataset.

## Project Structure
- **Part A**: Baseline evaluation of the pre-trained model.
- **Part B**: Fine-tuning pipeline using a custom dataset subset.
- **Part C**: Technical report and analysis of results.

## Key Results
| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| WER    | 61.34%   | 65.46%     | -4.12%      |
| CER    | 16.50%   | 19.51%     | -3.01%      |

## Analysis Summary
Despite a functional training pipeline, the model showed a regression in performance. This is attributed to:
1. **Catastrophic Forgetting**: The small fine-tuning set (200 samples) caused the model to lose its general linguistic knowledge.
2. **Overfitting**: Training loss reached near-zero, while validation WER increased after 200 steps.
3. **Data Scarcity**: Azerbaijani is a low-resource language; effective fine-tuning requires significantly more annotated data.

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
