
import argparse
import os
import time
import torch
import evaluate
from collections import Counter
from datasets import load_dataset, Value
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    default_data_collator,
    TrainingArguments,
    Trainer,
    logging as hf_logging
)

def ts() -> str:
    """Return a timestamp string [HH:MM:SS]."""
    return time.strftime("[%H:%M:%S]")

def load_filter_split(img_root: str, min_imgs: int = 5, seed: int = 42):
    """
    • Load imagefolder from img_root.
    • Drop any class with fewer than min_imgs examples.
    • Re-encode labels to contiguous ids.
    • Return stratified 80/10/10 splits: train, val, test, and class names.
    """
    print(f"{ts()} Loading images from {img_root} …")
    ds = load_dataset("imagefolder", data_dir=img_root, split="train")
    print(f"{ts()} Raw dataset: {len(ds):,} images, "
          f"{ds.features['label'].num_classes} classes")

    # 1. Drop rare classes
    cnt = Counter(ds["label"])
    keep_ids = {lbl for lbl, n in cnt.items() if n >= min_imgs}
    drop_ids = {lbl for lbl, n in cnt.items() if n < min_imgs}
    if drop_ids:
        n_drop = sum(cnt[i] for i in drop_ids)
        print(f"{ts()} Dropping {len(drop_ids)} rare classes (<{min_imgs} imgs) "
              f"totalling {n_drop:,} images")
    ds = ds.filter(lambda ex: ex["label"] in keep_ids)

    # 2. Re-encode labels for contiguous ids
    int2str = ds.features["label"].int2str
    ds = ds.map(lambda ex: {"label": int2str(ex["label"])})
    ds = ds.cast_column("label", Value("string")).class_encode_column("label")
    classes = ds.features["label"].names
    print(f"{ts()} Filtered dataset: {len(ds):,} images, {len(classes)} classes")

    # 3. Stratified 80/20 train/hold-out split
    try:
        split = ds.train_test_split(test_size=0.2, seed=seed,
                                    stratify_by_column="label")
    except ValueError as e:
        print(f"{ts()} ⚠️ Stratified split failed ({e}); using random split.")
        split = ds.train_test_split(test_size=0.2, seed=seed, shuffle=True)

    train = split["train"]       # 80%
    hold  = split["test"]        # 20%

    # 4. Hold-out → 50/50 val/test (random)
    split2 = hold.train_test_split(test_size=0.5, seed=seed, shuffle=True)
    val, test = split2["train"], split2["test"]

    return train, val, test, classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Folder containing compressed_dataset/")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--freeze", action="store_true",
                        help="Freeze backbone; train classifier only")
    parser.add_argument("--log-every", type=int, default=25,
                        help="Number of batches between logging to console")
    parser.add_argument("--min-images", type=int, default=5,
                        help="Drop classes with fewer than this many images")
    args = parser.parse_args()

    # 1. Load, filter, split data and save to disk
    IMG_ROOT = os.path.join(args.root, "compressed_dataset")
    train_ds, val_ds, test_ds, classes = load_filter_split(
        IMG_ROOT, min_imgs=args.min_images
    )

    train_ds.save_to_disk("train_ds")
    val_ds.save_to_disk("val_ds")
    test_ds.save_to_disk("test_ds")
    print("Path to train dataset: " + train_ds.data_files["image"][0])

    # 2. Initialize ViT processor and model
    proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(classes),
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)},
        ignore_mismatched_sizes=True
    )

    # 3. Optionally freeze backbone (only classifier head trains)
    if args.freeze:
        print(f"{ts()} Freezing backbone — only classifier will train")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

    # 4. Define transformation: preprocess images and attach labels
    def transform_fn(examples):
        outputs = proc(examples["image"], return_tensors="pt")
        outputs["labels"] = examples["label"]
        return outputs

    train_ds.set_transform(transform_fn)
    val_ds.set_transform(transform_fn)
    test_ds.set_transform(transform_fn)

    # 5. Set up training arguments
    hf_logging.set_verbosity_info()
    targs = TrainingArguments(
        output_dir="vit_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.log_every,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False
    )

    # 6. Load accuracy metric and define compute_metrics
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        top1 = acc_metric.compute(
            predictions=logits.argmax(axis=-1), references=labels
        )["accuracy"]
        # Compute top-5 accuracy
        top5 = (
            logits.argsort(axis=-1)[:, -5:] == labels[:, None]
        ).any(axis=-1).mean()
        return {"accuracy": top1, "top5_accuracy": top5}

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    # 8. Train and evaluate
    print(f"{ts()} Starting ViT training for {args.epochs} epochs …")
    trainer.train()
    trainer.save_model("vit_ckpt")

    print(f"{ts()} Training complete — evaluating on test set …")
    test_metrics = trainer.predict(test_ds).metrics
    print(f"{ts()} Test metrics → {test_metrics}")

if __name__ == "__main__":
    main()

"""
Run with:
python3 train_geo_vit.py \
  --root /path/to/dataset/ (whatever path geo50k dataset is in)\
  --epochs 10 (however many we want)\
  --log-every 1000 (how often to log to console)\
"""

"""
Needs: !pip install -q kaggle transformers datasets torch torchvision timm accelerate evaluate "fsspec>=2024.3.0"
If getting a ** error, try: !pip install -q --upgrade "fsspec>=2024.3.0" "datasets>=2.0.0"
"""