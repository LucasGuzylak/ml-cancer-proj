import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from torchmil.models import ABMIL


FEAT_DIR = "./data/patches_512/features/features_resnet50_bt"
PLABEL_DIR = "./data/patches_512/patch_labels"
COORD_DIR = "./data/patches_512/coords"
MODEL_PATH = "camelyon_abmil.pth"


def load_model(device):
    model = ABMIL(in_shape=(2048,)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def slide_attention(model, slide, device):
    feats = np.load(os.path.join(FEAT_DIR, slide))
    plabels = np.load(os.path.join(PLABEL_DIR, slide))
    coords = np.load(os.path.join(COORD_DIR, slide))

    X = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        Y_pred, att = model(X, return_att=True)

    att = att.squeeze(0).cpu().numpy()
    att_norm = (att - att.min()) / (att.max() - att.min() + 1e-8)
    prob = torch.sigmoid(Y_pred.squeeze()).item()
    return att, att_norm, plabels, coords, prob


def plot_slide(model, slide, device, out_path="attention_vs_truth.png"):
    att, att_norm, plabels, coords, prob = slide_attention(model, slide, device)
    tumor_mask = plabels == 1

    print(f"Slide: {slide}")
    print(f"Slide prediction: {prob:.3f} (>0.5 = tumor)")
    print(f"Mean attention on TUMOR patches:  {att_norm[tumor_mask].mean():.3f}")
    print(f"Mean attention on NORMAL patches: {att_norm[~tumor_mask].mean():.3f}")
    top50 = np.argsort(att)[-50:]
    print(f"Real tumor patches among top-50 attended: {int(plabels[top50].sum())}/50")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=att_norm, cmap="inferno", s=8)
    axes[0].set_title("Model attention (brighter = more suspicious)")
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0], fraction=0.046)

    axes[1].scatter(coords[~tumor_mask, 0], coords[~tumor_mask, 1], c="lightgray", s=8)
    axes[1].scatter(coords[tumor_mask, 0], coords[tumor_mask, 1], c="red", s=20)
    axes[1].set_title("Ground truth (red = real tumor)")
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path}")


def summarize_all_tumor_slides(model, device):
    tumor_files = sorted(f for f in os.listdir(FEAT_DIR) if "tumor" in f)

    tumor_means = []
    normal_means = []
    hit_fractions = []
    correct_preds = 0

    for slide in tumor_files:
        att, att_norm, plabels, coords, prob = slide_attention(model, slide, device)
        if plabels.sum() == 0:
            continue
        tumor_mask = plabels == 1
        tumor_means.append(att_norm[tumor_mask].mean())
        normal_means.append(att_norm[~tumor_mask].mean())
        top50 = np.argsort(att)[-50:]
        hit_fractions.append(plabels[top50].sum() / min(50, int(plabels.sum())))
        correct_preds += int(prob > 0.5)

    n = len(tumor_means)
    print(f"\nAcross {n} tumor slides:")
    print(f"  Correctly predicted as tumor: {correct_preds}/{n}")
    print(f"  Mean attention on tumor patches:  {np.mean(tumor_means):.3f}")
    print(f"  Mean attention on normal patches: {np.mean(normal_means):.3f}")
    print(f"  Mean tumor-recall in top-50 attended: {np.mean(hit_fractions):.3f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    plot_slide(model, "tumor_001.npy", device)
    summarize_all_tumor_slides(model, device)