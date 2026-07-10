import os
import shutil
import tarfile
import torch
from torch.utils.data import DataLoader, Subset
from torchmil.datasets import CAMELYON16MILDataset
from torchmil.models import ABMIL
from torchmil.data import collate_fn


def download_archives(dest="./download"):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="torchmil/Camelyon16_MIL",
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=[
            "dataset/patches_512/features/features_resnet50_bt.tar.gz",
            "dataset/patches_512/labels.tar.gz",
            "dataset/splits.csv",
        ],
    )
    print("Download complete.")


def extract_balanced_subset(src="./download", dst="./data", per_class=30):
    feat_dir = os.path.join(dst, "patches_512/features/features_resnet50_bt")
    lbl_dir = os.path.join(dst, "patches_512/labels")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    feat_archive = os.path.join(
        src, "dataset/patches_512/features/features_resnet50_bt.tar.gz"
    )
    with tarfile.open(feat_archive) as tar:
        npy = [m for m in tar.getmembers() if m.name.endswith(".npy")]
        tumor = [m for m in npy if "tumor" in os.path.basename(m.name)]
        normal = [m for m in npy if "normal" in os.path.basename(m.name)]
        print(f"In archive -> tumor: {len(tumor)}, normal: {len(normal)}")

        subset = tumor[:per_class] + normal[:per_class]
        for m in subset:
            m.name = os.path.basename(m.name)
            tar.extract(m, feat_dir)
    print(f"Extracted {len(subset)} feature files (balanced).")

    lbl_archive = os.path.join(src, "dataset/patches_512/labels.tar.gz")
    with tarfile.open(lbl_archive) as tar:
        for m in [m for m in tar.getmembers() if m.name.endswith(".npy")]:
            m.name = os.path.basename(m.name)
            tar.extract(m, lbl_dir)
    print("Extracted labels.")

    shutil.copy(os.path.join(src, "dataset/splits.csv"), os.path.join(dst, "splits.csv"))
    print("Copied splits.csv.")

    shutil.rmtree(src, ignore_errors=True)
    print("Deleted archive, reclaimed space.")


def run(root="./data", epochs=20, lr=1e-4, seed=42, train_frac=0.75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    fullset = CAMELYON16MILDataset(
        root=root, features="resnet50_bt", partition="train", bag_keys=["X", "Y"]
    )
    n = len(fullset)
    print("Total slides:", n)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    split = int(train_frac * n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_sub = Subset(fullset, train_idx)
    test_sub = Subset(fullset, test_idx)
    print(f"Train: {len(train_sub)} | Test: {len(test_sub)}")

    trainloader = DataLoader(train_sub, batch_size=1, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(test_sub, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = ABMIL(in_shape=(2048,)).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in trainloader:
            X = batch["X"].to(device)
            Y = batch["Y"].float().to(device)
            mask = batch["mask"].to(device) if "mask" in batch else None

            optimizer.zero_grad()
            Y_pred = model(X, mask)
            loss = loss_fn(Y_pred.squeeze(), Y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.3f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in testloader:
            X = batch["X"].to(device)
            Y = batch["Y"].float().to(device)
            mask = batch["mask"].to(device) if "mask" in batch else None

            Y_pred = model(X, mask)
            pred = (torch.sigmoid(Y_pred.squeeze()) > 0.5).float()
            correct += (pred == Y.squeeze()).sum().item()
            total += Y.numel()

    print(f"\nHeld-out test slides: {total}")
    print(f"Test Accuracy: {100 * correct / total:.1f}%")

    torch.save(model.state_dict(), "camelyon_abmil.pth")
    print("Model saved to camelyon_abmil.pth")
    return model


if __name__ == "__main__":
    run()