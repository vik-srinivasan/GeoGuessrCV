"""
Fine-tune Vision Transformer (ViT) on GeoGuessr images
(one sub-folder per country).

• Uses google/vit-base-patch16-224 pre-trained checkpoint
• Drop countries with fewer than --min-images pictures
  (default 5, change as needed).
• 80 / 10 / 10 splits:
      – train vs hold-out **stratified**
      – hold-out → val / test **random**.
• Supports layer-wise learning rate decay, mixed precision training,
  and other ViT-specific hyperparameters.
"""

import argparse, os, time, torch, evaluate
from collections import Counter
from datasets import load_dataset, Value
from transformers import (
    AutoImageProcessor, ViTForImageClassification, default_data_collator,
    TrainingArguments, Trainer, logging as hf_logging
)

def ts():  # timestamp
    return time.strftime("[%H:%M:%S]")

# ───────────────────────── data loader ──────────────────────────
def load_filter_split(img_root: str, min_imgs: int, seed: int = 42):
    print(f"{ts()} Loading images from {img_root} …")
    ds = load_dataset("imagefolder", data_dir=img_root, split="train")
    print(f"{ts()} Raw dataset: {len(ds):,} images, "
          f"{ds.features['label'].num_classes} classes")

    # 1. drop rare countries
    cnt = Counter(ds["label"])
    keep_ids = {lbl for lbl, n in cnt.items() if n >= min_imgs}
    drop_ids = {lbl for lbl, n in cnt.items() if n < min_imgs}
    if drop_ids:
        n_drop = sum(cnt[i] for i in drop_ids)
        print(f"{ts()} Dropping {len(drop_ids)} rare classes (<{min_imgs} imgs) "
              f"totalling {n_drop:,} images")
    ds = ds.filter(lambda ex: ex["label"] in keep_ids)

    # 2. rebuild ClassLabel for contiguous ids
    int2str = ds.features["label"].int2str
    ds = ds.map(lambda ex: {"label": int2str(ex["label"])})
    ds = ds.cast_column("label", Value("string"))
    ds = ds.class_encode_column("label")
    classes = ds.features["label"].names

    print(f"{ts()} Filtered dataset: {len(ds):,} images, "
          f"{len(classes)} classes after filtering")

    # 3. train / hold-out (stratified)
    try:
        split = ds.train_test_split(test_size=0.2, seed=seed,
                                    stratify_by_column="label")
    except ValueError as e:
        print(f"{ts()} ⚠️ Stratified split failed ({e}); using random split.")
        split = ds.train_test_split(test_size=0.2, seed=seed, shuffle=True)

    train = split["train"]
    hold  = split["test"]   # 20 %

    # 4. hold-out → val / test (random 50 / 50)
    split2 = hold.train_test_split(test_size=0.5, seed=seed, shuffle=True)
    val, test = split2["train"], split2["test"]

    return train, val, test, classes
# ─────────────────────────────────────────────────────────────────

# ViT-specific learning rate scheduler with layer-wise decay
def get_layer_wise_lr_decay_optimizer(model, weight_decay, lr, decay_rate):
    """
    Apply layer-wise learning rate decay for ViT model.
    Lower layers get lower learning rates.
    """
    param_groups = []
    
    # Layer-wise learning rate decay
    # Embedding layer (lowest LR)
    embedding_params = list(model.vit.embeddings.parameters())
    param_groups.append({
        "params": embedding_params,
        "lr": lr * decay_rate**2,
        "weight_decay": weight_decay
    })
    
    # Encoder layers (gradually increasing LR)
    num_layers = len(model.vit.encoder.layer)
    for i, layer in enumerate(model.vit.encoder.layer):
        layer_lr = lr * decay_rate**(1 - i/num_layers)
        param_groups.append({
            "params": layer.parameters(),
            "lr": layer_lr,
            "weight_decay": weight_decay
        })
    
    # Classifier head (highest LR)
    classifier_params = list(model.classifier.parameters())
    param_groups.append({
        "params": classifier_params,
        "lr": lr,
        "weight_decay": weight_decay
    })
    
    # Create optimizer
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Folder containing compressed_dataset/")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--freeze", action="store_true",
                    help="Train classifier head only")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--min-images", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for training (16-64 recommended)")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Base learning rate (1e-5 to 5e-5 recommended)")
    ap.add_argument("--weight-decay", type=float, default=0.05,
                    help="Weight decay (0.01 to 0.1 recommended)")
    ap.add_argument("--layer-wise-decay", type=float, default=0.75,
                    help="Layer-wise learning rate decay (0.65-0.8 recommended)")
    ap.add_argument("--no-layer-wise-lr", action="store_true",
                    help="Disable layer-wise learning rate decay")
    args = ap.parse_args()

    IMG_ROOT = os.path.join(args.root, "compressed_dataset")
    train, val, test, classes = load_filter_split(
        IMG_ROOT, min_imgs=args.min_images)

    # Use ViT base model with 16x16 patch size
    model_name = "google/vit-base-patch16-224"
    print(f"{ts()} Loading ViT model: {model_name}")
    
    # Image processor for ViT
    proc = AutoImageProcessor.from_pretrained(model_name)
    
    # ViT model with classification head
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(classes),
        id2label=dict(enumerate(classes)),
        label2id={c: i for i, c in enumerate(classes)},
        ignore_mismatched_sizes=True,
    )

    if args.freeze:
        print(f"{ts()} Freezing backbone — only classifier will train")
        for p in model.vit.parameters(): p.requires_grad = False
        for p in model.classifier.parameters(): p.requires_grad = True

    def tfm(ex):
        batch = proc(ex["image"], return_tensors="pt")
        batch["labels"] = ex["label"]
        return batch
    
    for ds in (train, val, test):
        ds.set_transform(tfm)

    hf_logging.set_verbosity_info()
    
    # Set up training arguments with ViT-specific settings
    targs = TrainingArguments(
        output_dir="vit_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.log_every,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        fp16=True,  # Mixed precision training
        remove_unused_columns=False,
        save_total_limit=2,  # Only keep the 2 most recent checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Set up metrics
    acc = evaluate.load("accuracy")
    def metrics(p):
        preds, refs = p.predictions, p.label_ids
        return {
            "accuracy": acc.compute(predictions=preds.argmax(1), references=refs)["accuracy"],
            "top5_accuracy": (preds.argsort(axis=-1)[:, -5:] == refs[:, None]).any(-1).mean(),
        }

    print(f"{ts()} Starting training for {args.epochs} epoch(s)…")
    
    # Initialize trainer with or without custom optimizer for layer-wise LR decay
    if args.no_layer_wise_lr:
        trainer = Trainer(
            model,
            targs,
            train_dataset=train,
            eval_dataset=val,
            data_collator=default_data_collator,
            compute_metrics=metrics,
        )
    else:
        print(f"{ts()} Using layer-wise learning rate decay with factor {args.layer_wise_decay}")
        optimizer = get_layer_wise_lr_decay_optimizer(
            model, 
            weight_decay=args.weight_decay, 
            lr=args.lr, 
            decay_rate=args.layer_wise_decay
        )
        
        trainer = Trainer(
            model,
            targs,
            train_dataset=train,
            eval_dataset=val,
            data_collator=default_data_collator,
            compute_metrics=metrics,
            optimizers=(optimizer, None),  # Custom optimizer, default scheduler
        )
    
    # Train the model
    trainer.train()
    trainer.save_model("vit_ckpt/final")

    print(f"{ts()} Training done — evaluating on test set")
    print(trainer.predict(test).metrics)

if __name__ == "__main__":
    main() 