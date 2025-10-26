# ruBERT-ruLaw

This repository contains code for continued pretraining of [DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased) on the [RusLawOD dataset](https://huggingface.co/datasets/irlspbru/RusLawOD) — a large corpus of Russian legal texts (court decisions and normative acts). It also includes code for evaluating the model's performance on masked language modeling (MLM) tasks. The goal of this training is to improve RuBERT’s performance on legal domain tasks such as information extraction and retrieval.

The trained model is available at [TryDotAtwo/ruBERT-ruLaw](https://huggingface.co/TryDotAtwo/ruBERT-ruLaw).

## Installation

The scripts handle dependency installation automatically via pip in the same interpreter. Required packages include:

- `torch` (nightly with CUDA 13.0 support)
- `transformers`
- `datasets`
- `tensorboard`
- `accelerate`

Run the training or evaluation script directly, and it will install missing packages as needed.

## Usage

### Training

The main training script (`ruBERT_RuLaw - 3xH200.py.py`) performs continued pretraining using Masked Language Modeling (MLM) on the RusLawOD dataset.

#### Key Features
- Tokenization with sliding window (max_length=512, stride=128) to handle long legal documents.
- Filters out empty or None texts.
- Splits data into train (90%) and eval (10%).
- Supports test mode (subsamples 3k examples for quick testing).
- Uses BF16 precision for A100/H100/H200 GPUs.
- Logs to TensorBoard for monitoring.

#### Running Training
```bash
accelerate launch ruBERT_RuLaw - 3xH200.py
```

- **Test Mode**: Set `test_mode = True` in the script for a quick run (3 epochs, 3k samples, fewer steps).
- **Full Run**: Defaults to 8 epochs, 40k steps, full dataset.
- **Resuming**: Supports resuming from checkpoint with `resume_from_checkpoint=True`.
- **Output**: Model saved to `~/ruBERT_data/ruBERT-ruLaw`. Logs in `~/ruBERT_data/rulaw_logs`.
- **Monitoring**: Launch TensorBoard with `tensorboard --logdir ~/ruBERT_data/rulaw_logs`.

Hyperparameters (configurable in `TrainingArguments`):
- Batch size: 160 (train/eval)
- Warmup steps: 2000
- Save/eval steps: 2000
- MLM probability: 0.15

### Evaluation

The evaluation script (`Test_MLM.py`) tests MLM accuracy (Top-1 and Top-k, k=5) across multiple models, including the fine-tuned ruBERT-ruLaw, on the [sud-resh-benchmark dataset](https://huggingface.co/datasets/lawful-good-project/sud-resh-benchmark).

#### Key Features
- Sliding window tokenization for long texts.
- Tests MLM probabilities from 0.1 to 0.4 (step 0.05).
- Supports resuming interrupted runs (set `resume = True`).
- Batched inference with FP16 autocast for efficiency.
- Saves results as `mlm_results.pth` (PyTorch) and `mlm_results.json` (for plotting).

#### Running Evaluation
```bash
python Test_MLM.py
```

- **Limit Texts**: Set `limit_texts = N` (e.g., 1000) for partial runs.
- **Models Tested**: Includes ruBERT-ruLaw, base ruBERT, Legal-BERT, BERT variants, and others (list in script).
- **Output**: Metrics printed and saved. Use `mlm_results.json` for graphs (e.g., via Matplotlib or Excel).

## Results

## MLM Accuracy Comparison

|  MLM Probability  |  Metric  |  ruBERT-ruLaw  |  rubert-base-cased  |  legal-bert-base-uncased  |
|-------------------|----------|----------------|---------------------|---------------------------|
|        10%        |  Top-1   |    **0.81**    |        0.73         |           0.45            |
|                   |  Top-5   |    **0.92**    |        0.87         |           0.77            |
|        15%        |  Top-1   |    **0.79**    |        0.68         |           0.45            |
|                   |  Top-5   |    **0.91**    |        0.83         |           0.77            |
|        20%        |  Top-1   |    **0.76**    |        0.54         |           0.45            |
|                   |  Top-5   |    **0.89**    |        0.72         |           0.76            |
|        25%        |  Top-1   |    **0.74**    |        0.18         |           0.44            |
|                   |  Top-5   |    **0.87**    |        0.32         |           0.75            |
|        30%        |  Top-1   |    **0.70**    |        0.06         |           0.44            |
|                   |  Top-5   |    **0.85**    |        0.11         |           0.74            |
|        35%        |  Top-1   |    **0.67**    |        0.06         |           0.43            |
|                   |  Top-5   |    **0.82**    |        0.09         |           0.73            |
|        40%        |  Top-1   |    **0.63**    |        0.06         |           0.42            |
|                   |  Top-5   |    **0.79**    |        0.09         |           0.72            |
Press any key to continue . . .
[Placeholder: Evaluation results will be added here once testing is complete]

### TensorBoard Graphs

View training progress via TensorBoard:

![TensorBoard Screenshot](path/to/screenshot.png) <!-- Placeholder for actual screenshot -->

Launch with: `tensorboard --logdir ~/ruBERT_data/rulaw_logs`

### Article Archive

[Link to related paper or blog post] <!-- Placeholder for archive link -->

For more details, see the [Hugging Face model card](https://huggingface.co/TryDotAtwo/ruBERT-ruLaw).
