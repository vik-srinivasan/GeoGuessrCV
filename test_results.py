import os
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────
PERSIST_ROOT   = "/mnt/data"                 # same root you used to save splits & model
TEST_DS_PATH   = os.path.join(PERSIST_ROOT, "test_ds")
MODEL_DIR      = os.path.join(PERSIST_ROOT, "vit_ckpt")  # where trained model was saved
BATCH_SIZE     = 32
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV     = os.path.join(PERSIST_ROOT, "vit_test_predictions.csv")

# ─── 1. Reload the saved test-split (before transforms) ────────────────────
# This dataset has columns: "image" (PIL image), "label" (integer), and label names are in its ClassLabel feature.
test_raw = load_from_disk(TEST_DS_PATH)

# Extract the id2label mapping (list of country names, indexed by integer)
id2label = test_raw.features["label"].names

# For convenience, also pull out file paths and true labels into Python lists:
#   test_raw["image"] is a PIL Image object, so its path is in image.filename
all_paths  = [example["image"].filename for example in test_raw]
all_labels = [id2label[example["label"]] for example in test_raw]

# ─── 2. Load processor & model (fine-tuned) ─────────────────────────────────
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model     = ViTForImageClassification.from_pretrained(
    MODEL_DIR,
    ignore_mismatched_sizes=True
).to(DEVICE)
model.eval()

# ─── 3. Build a Dataset+DataLoader that yields pixel‐values + labels ─────────
# We can reuse the same transform_fn you used in training:
def preprocess(examples):
    # `examples["image"]` is still a PIL.Image
    out = processor(examples["image"], return_tensors="pt")
    out["labels"] = examples["label"]
    return out

# Attach the transform so that each batch has {"pixel_values": Tensor, "labels": Tensor}
test_ds = test_raw.with_transform(preprocess)

# Create a PyTorch DataLoader for batching
def collate_fn(batch):
    # HuggingFace's default_data_collator expects pixel_values & labels
    pixel_vals = torch.stack([ex["pixel_values"].squeeze(0) for ex in batch])
    labels     = torch.tensor([ex["labels"].item() for ex in batch])
    return {"pixel_values": pixel_vals, "labels": labels}

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ─── 4. Iterate and collect logits, then compute top-1/top-5 ──────────────────
all_pred_top1 = []
all_pred_top5 = []

with torch.no_grad():
    for batch in test_loader:
        pixels = batch["pixel_values"].to(DEVICE)       # shape (B, 3, 224, 224)
        logits = model(pixels).logits                    # shape (B, num_classes)
        probs  = torch.softmax(logits, dim=-1)

        # top-1 indices
        top1_ids = torch.argmax(probs, dim=-1).cpu().tolist()
        all_pred_top1.extend([id2label[i] for i in top1_ids])

        # top-5 indices (sorted by descending probability)
        top5_ids = torch.topk(probs, k=5, dim=-1).indices.cpu().tolist()
        for idx_list in top5_ids:
            all_pred_top5.append([id2label[i] for i in idx_list])

# Sanity check lengths
assert len(all_paths) == len(all_labels) == len(all_pred_top1) == len(all_pred_top5)

# ─── 5. Build a DataFrame and save as CSV ────────────────────────────────────
df_out = pd.DataFrame({
    "image_path": all_paths,
    "true_label": all_labels,
    "pred_top1":  all_pred_top1,
    "pred_top5":  all_pred_top5
})
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"Saved per‐image predictions to {OUTPUT_CSV}")
