# Fine-Tuning Open-Weight LLMs with LoRA for Document Understanding

> **What this is:** I fine-tuned Mistral-7B and LLaMA-2 (7B) using PEFT/LoRA to extract structured information from identity documents (PAN cards, Aadhaar, driving licenses, passports). This started as a weekend experiment and grew into something I'm genuinely proud of.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?logo=pytorch)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why I Built This

I've been interested in document AI for a while now, and most of the existing solutions either use expensive API calls (GPT-4V, Claude) or require massive compute to train from scratch. I wanted to see if I could get decent results fine-tuning a 7B model on a single GPU using LoRA.

Spoiler: it works surprisingly well if you're careful with your data.

## What It Does

Given an image of an identity document (or its OCR text), the model extracts structured fields:

```json
{
  "document_type": "PAN_CARD",
  "name": "RAJESH KUMAR SHARMA",
  "father_name": "SURESH CHANDRA SHARMA", 
  "dob": "15/03/1985",
  "pan_number": "ABCDE1234F",
  "confidence": 0.94
}
```

## Setup & Requirements

I ran everything on a single RTX 3090 (24GB VRAM). If you're working with less VRAM, check the memory optimization section below.

```bash
pip install -r requirements.txt
```

You'll also need a Hugging Face account with access to Mistral-7B (it's gated, you need to request access).

## Quick Start

```bash
# 1. Prepare your data (format described below)
python scripts/prepare_data.py --input data/raw/ --output data/processed/

# 2. Start training
python scripts/train.py --config configs/mistral_lora_r16.yaml

# 3. Run evaluation
python scripts/evaluate.py --model checkpoints/best_model --data data/test/

# 4. Inference on new documents
python scripts/inference.py --model checkpoints/best_model --image path/to/document.jpg
```

## Training Data Format

This was honestly the hardest part. Getting the data format right matters more than any hyperparameter.

Each training example is a JSON object:

```json
{
  "instruction": "Extract all fields from this PAN card document.",
  "input": "INCOME TAX DEPARTMENT\nGOVT OF INDIA\n\nName: RAJESH KUMAR SHARMA\nFather's Name: SURESH CHANDRA SHARMA\nDate of Birth: 15/03/1985\nPermanent Account Number: ABCDE1234F",
  "output": "{\"document_type\": \"PAN_CARD\", \"name\": \"RAJESH KUMAR SHARMA\", \"father_name\": \"SURESH CHANDRA SHARMA\", \"dob\": \"15/03/1985\", \"pan_number\": \"ABCDE1234F\"}"
}
```

I started with ~500 examples and it was... not great. The model would hallucinate fields and mix up formats. Around 2000 examples it started getting reliable. My final dataset has 3,847 examples across 6 document types.

## LoRA Configuration

I experimented quite a bit with LoRA parameters. Here's what I found:

| Parameter | Values Tried | Best | Notes |
|-----------|:---:|:---:|-------|
| `r` (rank) | 8, 16, 32, 64 | **16** | r=64 overfitted badly on my dataset size |
| `alpha` | 16, 32, 64 | **32** | alpha = 2*r worked best for me |
| `dropout` | 0.0, 0.05, 0.1 | **0.05** | 0.0 gave slightly better train loss but worse val |
| `target_modules` | q_proj only, q+k, all linear | **all linear** | targeting all linear layers was worth the ~20% slower training |
| `learning_rate` | 1e-4, 2e-4, 5e-4 | **2e-4** | 5e-4 diverged after ~500 steps |

The config files in `configs/` have all my experimental setups.

## Training Details

```
Base Model:       mistralai/Mistral-7B-Instruct-v0.2
Trainable Params: ~41M (out of 7.2B total)
Training Time:    ~4.5 hours on RTX 3090
Batch Size:       4 (with gradient accumulation = 8)
Max Seq Length:   2048
Optimizer:        AdamW (8-bit)
LR Schedule:      cosine with warmup (100 steps)
Epochs:           3 (early stopped at epoch 3, val loss plateaued)
Precision:        bf16 (fp16 had some NaN issues early on)
```

**Note:** I initially tried fp16 mixed precision but kept getting NaN losses around step 300. Switching to bf16 fixed this completely. If you're on a GPU that doesn't support bf16 (like a 3090 technically... but it worked for me with `torch_dtype=torch.bfloat16`), you might need to use fp16 with a lower learning rate.

## Results

### Quantitative

| Document Type | Accuracy | F1-Score | Latency (ms) |
|---|:---:|:---:|:---:|
| PAN Card | 94.2% | 0.93 | 820 |
| Aadhaar | 91.8% | 0.91 | 890 |
| Driving License | 89.5% | 0.88 | 850 |
| Passport | 92.1% | 0.91 | 870 |
| Voter ID | 88.3% | 0.87 | 840 |
| Bank Statement | 85.7% | 0.84 | 1120 |

### What Surprised Me

The model is really good at PAN cards and passports (probably because these have very consistent formats) but struggles with bank statements. Bank statements have wild variation in layout and the OCR quality is often poor.

I also noticed the model sometimes "corrects" obvious OCR errors on its own, which is cool but also risky. For example, if the OCR reads "RAJESH" as "RAJESH" with a weird character, the model usually fixes it. But I wouldn't trust this behavior in production without a human in the loop.

### What Didn't Work

- **Training on raw images directly:** I initially wanted to do end-to-end vision+language but 7B models just can't handle the sequence length. I ended up using Tesseract OCR as a preprocessing step.
- **r=64 LoRA:** massively overfitted. The model would memorize training examples perfectly but fail on anything slightly different.
- **Training for more than 3 epochs:** validation loss started going up after epoch 3 every single time I tried. Classic overfitting.
- **Using GPT-4 to generate synthetic data without filtering:** the synthetic data had subtle format inconsistencies that confused the model. I had to manually review and clean ~30% of it.

## Evaluation Harness

I built a custom evaluation harness because standard metrics didn't capture what I cared about. The key metrics I track:

1. **Field-Level Exact Match:** Does each extracted field exactly match the ground truth?
2. **Field-Level Fuzzy Match:** Allows for minor variations (extra spaces, case differences)
3. **JSON Structure Validity:** Does the output parse as valid JSON with correct keys?
4. **Hallucination Rate:** How often does the model invent fields that don't exist in the document?
5. **Adversarial Robustness:** Performance on documents with intentional noise (blur, rotation, partial occlusion)

The adversarial evaluation was eye-opening. Performance drops by about 12-15% when documents are slightly degraded. This is still a work in progress.

Run evaluation:
```bash
python scripts/evaluate.py \
    --model checkpoints/best_model \
    --data data/test/ \
    --adversarial  # include noisy/adversarial samples
```

## Experiment Log

I kept a rough log of experiments. Full details in `experiments/experiment_log.md`.

| # | Change | Result | Notes |
|---|--------|--------|-------|
| 1 | Baseline (r=8, lr=2e-4) | 82% F1 | Starting point |
| 2 | r=16, same lr | 88% F1 | Clear improvement |
| 3 | r=32 | 89% F1 | Marginal gain, not worth the extra params |
| 4 | r=64 | 85% F1 | Overfitting, val loss diverged |
| 5 | Back to r=16, target all linear | 91% F1 | Best so far |
| 6 | Added 1000 more training examples | 93% F1 | Data > hyperparams, as usual |
| 7 | bf16 instead of fp16 | 93% F1 | Same accuracy, no NaN issues |
| 8 | Gradient accumulation=8 | 93.5% F1 | Tiny improvement, kept it |
| 9 | Added data augmentation (OCR noise) | 94.2% F1 | Helped with adversarial robustness |
| 10 | Tried QLoRA (4-bit) | 90% F1 | Faster training but ~4% accuracy drop |

## Memory Optimization

If you're running on less than 24GB VRAM:

1. **Use QLoRA (4-bit quantization):** ~8GB VRAM. Slight accuracy drop (~3-4%) but trains fine.
   ```bash
   python scripts/train.py --config configs/mistral_qlora_r16.yaml
   ```

2. **Gradient checkpointing:** Enabled by default in my config. Costs ~20% training time but saves significant memory.

3. **Reduce batch size to 2** with gradient accumulation of 16.

## Project Structure

```
LLM-Fine-Tuning-Document-Understanding/
├── configs/
│   ├── mistral_lora_r16.yaml      # Best config
│   ├── mistral_qlora_r16.yaml     # 4-bit version
│   └── llama2_lora_r16.yaml       # LLaMA-2 variant
├── scripts/
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation harness
│   ├── inference.py               # Single document inference
│   └── prepare_data.py            # Data preprocessing
├── data/
│   ├── raw/                       # Raw OCR outputs + labels
│   └── processed/                 # Cleaned training data
├── evaluations/
│   └── results/                   # Evaluation results JSON
├── experiments/
│   └── experiment_log.md          # Full experiment tracking
├── notebooks/
│   └── analysis.ipynb             # Result visualization
├── requirements.txt
├── .gitignore
└── README.md
```

## TODO

- [ ] Try preference alignment (DPO) to improve output format consistency
- [ ] Add support for tabular document types (bank statements, invoices)
- [ ] Benchmark against GPT-4V on the same test set
- [ ] Add ONNX export for faster inference
- [ ] Write a proper blog post about the data curation process (it was the hardest part)

## Known Issues

- Model sometimes outputs fields in a different order than the training format (JSON is still valid, just annoying)
- Performance on handwritten documents is poor (all training data was printed text)
- Inference latency is ~800ms per document, too slow for real-time use cases without batching

## References

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT docs](https://huggingface.co/docs/peft)
- [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)

## Author

**Tushar Kumar** — [GitHub](https://github.com/sparrow000iv) | [LinkedIn](https://www.linkedin.com/in/tushar-kumar-737a6b303/)

*This project was built as part of my exploration into applied LLM research. Happy to discuss the approach or results if you're working on similar problems.*
