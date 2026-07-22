# Cancer Detection with Attention-Based Multiple Instance Learning

<p align="center">
  <img width="1099" alt="Model attention map on the left versus the ground-truth tumor location on the right, for the same pathology slide" src="https://github.com/user-attachments/assets/12ce17bb-448e-42a4-80b5-9916e7fb79f6" />
</p>

<p align="center">
  <em>Same slide, twice. Left: where the model focused. Right: where the tumor actually is (red). The model was only ever told "this slide has cancer," never which parts. It found the region on its own.</em>
</p>

## Overview

A pathology slide is about 100,000 x 100,000 pixels, far too big to hand to a neural network. So it gets cut into thousands of small patches. The catch is that a pathologist only labels the whole slide, not the individual patches. You get one "cancer / no cancer" answer for a bag of thousands of tiles, with no record of which tiles caused it.

That constraint is exactly what Multiple Instance Learning solves. If a slide is labeled cancer, at least one patch is tumor. If it's labeled clean, none are. The model learns which patches matter using only the slide-level label. This is weak supervision, and it's how real pathology models are trained, since annotating every patch by hand doesn't scale.

## Results

I trained an attention-based MIL model on a balanced 60-slide subset of CAMELYON16 (breast cancer lymph node slides). It reached 86.7% accuracy on 15 held-out slides it never saw during training.

The stronger result is what the attention weights revealed. The model assigns every patch a weight for how much it drove the final prediction. Mapped back onto the slide, those weights land on the real tumor:

- Mean attention on true tumor patches: **0.62**
- Mean attention on healthy patches: **0.11** (a 6x concentration)
- On the slide above, tumor made up only 0.5% of patches (43 of 7,893). Of the model's 50 highest-attention patches, **19 were real tumor** — roughly 60x better than chance.

The model learned to localize the tumor as a byproduct of learning to classify slides, with no patch-level supervision at any point. That is the core capability attention-based MIL is prized for in computational pathology.

## The build

I reached the cancer model through a staged curriculum, adding one hard technique per stage and validating it on progressively harder data before moving on.

| Stage | Dataset | New technique | Result |
|-------|---------|---------------|--------|
| 1-2 | FashionMNIST | Neural nets, then CNNs | 91.4% |
| 3 | CIFAR-10 | Color images, harder classes | 69.9% |
| 4 | CIFAR-10 | Transfer learning with ResNet18 | 89.4% |
| 5 | PathMNIST | Real medical tissue images | 86.4% |
| 6 | TissueMNIST | MIL pipeline built from scratch | pipeline validated |
| 7 | CAMELYON16 | Attention-based MIL on real slides | 86.7% + localization |

Stage 6 is worth calling out. The first MIL model sat at 50% because it was trained on synthetically constructed bags with an arbitrary "cancer" signal, so there was no real pattern to learn. Diagnosing that as a data problem rather than a code bug is what drove the move to real CAMELYON16 data in Stage 7.

## Files

```
train.py                 Stages 1-4: FashionMNIST, CIFAR-10, transfer learning
medical.py               Stage 5: PathMNIST
cancer.py                Stage 6: MIL pipeline built from scratch (synthetic bags)
camelyon.py              Stage 7: attention-based MIL on real CAMELYON16
visualize_attention.py   attention analysis and the localization figure
PROJECT.md               full technical write-up
```

## Scope

The CAMELYON16 result uses 60 of roughly 400 slides and a fixed-seed 75/25 split, run on free-tier GPU compute. Full-dataset versions of this method reach around 90% AUC; this is a focused demonstration of the same approach at smaller scale.

## Stack

Python, PyTorch, torchmil, MedMNIST
