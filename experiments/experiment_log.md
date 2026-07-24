# Experiment Log

Keeping track of everything I tried so I don't repeat mistakes.

## Baseline (Experiment #1)
**Date:** Initial run
**Config:** r=8, alpha=16, lr=2e-4, 2 epochs
**Result:** 82% F1 on validation set
**Notes:** Starting point. Model works but hallucinates a lot of fields. JSON output is messy.

## Experiment #2: Increase LoRA rank
**Change:** r=8 → r=16, alpha=32
**Result:** 88% F1 (+6%)
**Notes:** Clear improvement. The model seems to learn more nuanced patterns with higher rank. Worth the extra ~20M params.

## Experiment #3: Even higher rank
**Change:** r=16 → r=32, alpha=64
**Result:** 89% F1 (+1%)
**Notes:** Marginal gain. Not worth the extra training time and memory. Reverted to r=16.

## Experiment #4: Going too high
**Change:** r=16 → r=64
**Result:** 85% F1 (-4% from r=16)
**Notes:** Classic overfitting. Training loss kept going down but validation loss diverged after ~400 steps. The model was memorizing training examples. Lesson learned: more capacity ≠ better.

## Experiment #5: Target all linear layers
**Change:** Target only q_proj, v_proj → all linear layers (q, k, v, o, gate, up, down)
**Result:** 91% F1 (+3% from #2)
**Notes:** This was the single biggest improvement. Training is ~20% slower but the quality gain is worth it. Trainable params went from ~20M to ~41M.

## Experiment #6: More data
**Change:** 2000 → 3000 training examples (added 1000 more)
**Result:** 93% F1 (+2%)
**Notes:** As expected, data > hyperparameters. The new examples covered more edge cases (blurry documents, unusual formats). The model got noticeably better at handling noisy inputs.

## Experiment #7: bf16 vs fp16
**Change:** Switched from fp16 to bf16 mixed precision
**Result:** Same accuracy (93% F1) but NO NaN losses
**Notes:** This was more of a stability fix than a performance improvement. With fp16 I was getting NaN losses randomly around step 300-500. Switching to bf16 completely eliminated this issue. If your GPU supports bf16, use it.

## Experiment #8: Gradient accumulation
**Change:** batch_size=2, grad_accum=16 → batch_size=4, grad_accum=8 (same effective batch size)
**Result:** 93.5% F1 (tiny improvement, probably noise)
**Notes:** Honestly this was more about convenience. Larger per-device batch size means fewer optimizer steps, which is slightly faster. The accuracy difference is within noise margin.

## Experiment #9: Data augmentation (OCR noise)
**Change:** Added synthetic OCR errors to 30% of training data
**Result:** 94.2% F1 (+0.7%), but adversarial robustness improved by ~8%
**Notes:** This was inspired by the fact that real documents have OCR errors. I randomly inserted character substitutions, deletions, and extra whitespace into the input text. The model became much more robust to noisy inputs without sacrificing clean-input accuracy.

## Experiment #10: QLoRA (4-bit quantization)
**Change:** 16-bit → 4-bit quantization with QLoRA
**Result:** 90% F1 (-4% from full LoRA)
**Notes:** Training is ~40% faster and uses ~8GB VRAM instead of ~18GB. But there's a clear accuracy drop. If you're VRAM-constrained this is still a good option, but expect ~4% lower accuracy. I'd use this for experimentation and switch to full LoRA for final training.

## Things I Tried That Didn't Work

- **Training for 5+ epochs:** Validation loss starts going up after epoch 3 every time. Early stopping at epoch 3 is optimal.
- **Using the base Mistral model (not Instruct):** Worse at structured output. The Instruct variant is clearly better for JSON generation.
- **Learning rate 5e-4:** Diverged after ~500 steps. Too aggressive.
- **Learning rate 5e-5:** Converged too slowly, didn't reach good performance in 3 epochs.
- **Targeting only q_proj and v_proj:** Significantly worse than targeting all linear layers.
- **Using GPT-4 to generate synthetic data without filtering:** The synthetic data had subtle inconsistencies that confused the model. Had to manually review and clean ~30% of it.
- **Training on raw images (vision+language):** Sequence length becomes way too long. Ended up using Tesseract OCR as preprocessing instead.

## Current Best Config

r=16, alpha=32, all linear layers, lr=2e-4, bf16, 3 epochs, 3847 training examples with OCR noise augmentation.

**Final result:** 94.2% F1 on test set, 93.5% on validation set.
