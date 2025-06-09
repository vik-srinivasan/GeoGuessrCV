#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoGuessrCV: Country Classification with Vision Transformer

This script demonstrates how to train a Vision Transformer (ViT) model
to predict a country from an image, as a more powerful alternative to ResNet.
"""

import os
import time
import random
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

from datasets import load_dataset, Value
from transformers import (
    AutoImageProcessor, 
    ViTForImageClassification, 
    default_data_collator,
    TrainingArguments, 
    Trainer, 
    logging as hf_logging
)
import evaluate

def ts():  # timestamp
    return time.strftime("[%H:%M:%S]")

# ───────────────────────── data loader ──────────────────────────
def load_filter_split(img_root: str, min_imgs: int = 10, seed: int = 42):
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

def plot_class_distribution(train_dataset, classes, top_n=20):
    """Plot distribution of top N most frequent classes"""
    class_counts = Counter(train_dataset["label"])
    class_df = pd.DataFrame({
        'country': [classes[i] for i in range(len(classes))],
        'count': [class_counts.get(i, 0) for i in range(len(classes))]
    }).sort_values('count', ascending=False)
    
    # Plot top N countries by count
    plt.figure(figsize=(14, 6))
    top_countries = class_df.head(top_n)
    plt.bar(top_countries['country'], top_countries['count'])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Countries by Number of Images')
    plt.tight_layout()
    plt.savefig('top_countries_distribution.png')
    plt.close()
    
    print(f"Total number of countries: {len(classes)}")
    print(f"Distribution plot saved to 'top_countries_distribution.png'")

def plot_samples(dataset, classes, n=5, filename='sample_images.png'):
    """Plot sample images with their country labels"""
    fig, axes = plt.subplots(1, n, figsize=(15, 4))
    indices = random.sample(range(len(dataset)), n)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        label = sample["label"]
        country = classes[label]
        
        axes[i].imshow(image)
        axes[i].set_title(f"{country}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Sample images saved to '{filename}'")

def plot_confusion_matrix(labels, predictions, classes, top_n=15, filename='confusion_matrix.png'):
    """Plot confusion matrix for most confused country pairs"""
    # Get predicted class indices
    pred_labels = predictions.argmax(axis=1)
    
    # Full confusion matrix would be too large, so let's focus on the most confused pairs
    cm = confusion_matrix(labels, pred_labels)
    
    # Find the most confused pairs (excluding the diagonal)
    np.fill_diagonal(cm, 0)  # Zero out the diagonal to focus on errors
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:  # If there's confusion
                confused_pairs.append((i, j, cm[i, j]))
    
    # Sort by confusion count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Get the top confused pairs
    top_confused = confused_pairs[:top_n]
    print("Top confused country pairs:")
    for true_idx, pred_idx, count in top_confused:
        print(f"True: {classes[true_idx]}, Predicted: {classes[pred_idx]}, Count: {count}")
    
    # Create a smaller confusion matrix for visualization
    # Get unique indices involved in top confused pairs
    unique_indices = set()
    for true_idx, pred_idx, _ in top_confused:
        unique_indices.add(true_idx)
        unique_indices.add(pred_idx)
    unique_indices = sorted(list(unique_indices))
    
    # Create a smaller confusion matrix
    small_cm = confusion_matrix(
        [labels[i] for i in range(len(labels)) if labels[i] in unique_indices],
        [pred_labels[i] for i in range(len(pred_labels)) if labels[i] in unique_indices],
        labels=unique_indices
    )
    
    # Get class names for these indices
    small_classes = [classes[i] for i in unique_indices]
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(small_cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=small_classes, yticklabels=small_classes)
    plt.title("Confusion Matrix for Most Confused Countries")
    plt.ylabel("True Country")
    plt.xlabel("Predicted Country")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to '{filename}'")

def compare_with_resnet(test_results, filename='model_comparison.txt'):
    """Compare ViT results with ResNet baseline"""
    # ResNet results from the paper
    resnet_accuracy = 0.413
    resnet_top5 = 0.592
    
    vit_accuracy = test_results.metrics['test_accuracy']
    vit_top5 = test_results.metrics['test_top5_accuracy']
    
    comparison = "Model Comparison:\n"
    comparison += f"{'Model':<20} {'Accuracy':<10} {'Top-5 Accuracy':<15}\n"
    comparison += f"{'ResNet-50':<20} {resnet_accuracy:.4f}    {resnet_top5:.4f}\n"
    comparison += f"{'ViT-Base-Patch16':<20} {vit_accuracy:.4f}    {vit_top5:.4f}\n"
    comparison += f"Improvement: {(vit_accuracy - resnet_accuracy) * 100:.2f}% for top-1, {(vit_top5 - resnet_top5) * 100:.2f}% for top-5}"
    
    print(comparison)
    
    with open(filename, 'w') as f:
        f.write(comparison)
    print(f"Comparison saved to '{filename}'")

def main():
    print("GeoGuessrCV: Country Classification with Vision Transformer")
    
    # Set paths
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        PROJECT_DIR = "/content/drive/MyDrive/cs231n_project"  # adjust to your folder
    except ImportError:
        # Not in Colab, set your local project directory
        PROJECT_DIR = "."
    
    DATA_DIR = os.path.join(PROJECT_DIR, "geo50k")
    IMG_ROOT = os.path.join(DATA_DIR, "compressed_dataset")
    
    # Create output directory
    os.makedirs("vit_output", exist_ok=True)
    
    # Load and split dataset
    train, val, test, classes = load_filter_split(IMG_ROOT, min_imgs=10)
    
    # Plot class distribution
    plot_class_distribution(train, classes)
    
    # Plot sample images
    plot_samples(train, classes, filename='vit_output/sample_images.png')
    
    # Set up ViT model
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
    
    # Define the transform function
    def tfm(ex):
        batch = proc(ex["image"], return_tensors="pt")
        batch["labels"] = ex["label"]
        return batch
    
    # Apply transform to datasets
    for ds in (train, val, test):
        ds.set_transform(tfm)
    
    # Hyperparameters
    base_lr = 1e-5
    weight_decay = 0.05
    layer_wise_decay_rate = 0.75
    batch_size = 32
    num_epochs = 3
    
    # Create optimizer with layer-wise LR decay
    print(f"{ts()} Using layer-wise learning rate decay with factor {layer_wise_decay_rate}")
    optimizer = get_layer_wise_lr_decay_optimizer(
        model, 
        weight_decay=weight_decay, 
        lr=base_lr, 
        decay_rate=layer_wise_decay_rate
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="vit_output/checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=base_lr,
        weight_decay=weight_decay,
        fp16=True,  # Mixed precision training
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # Set up metrics
    acc = evaluate.load("accuracy")
    def compute_metrics(p):
        preds, refs = p.predictions, p.label_ids
        return {
            "accuracy": acc.compute(predictions=preds.argmax(1), references=refs)["accuracy"],
            "top5_accuracy": (preds.argsort(axis=-1)[:, -5:] == refs[:, None]).any(-1).mean(),
        }
    
    # Initialize trainer
    hf_logging.set_verbosity_info()
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )
    
    # Train the model
    print(f"{ts()} Starting training for {num_epochs} epoch(s)...")
    trainer.train()
    
    # Save the model
    model_save_path = os.path.join("vit_output", "final_model")
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Evaluate on test set
    print(f"{ts()} Evaluating on test set...")
    test_results = trainer.predict(test)
    print(test_results.metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_results.label_ids, test_results.predictions, 
                         classes, filename='vit_output/confusion_matrix.png')
    
    # Compare with ResNet baseline
    compare_with_resnet(test_results, filename='vit_output/model_comparison.txt')
    
    print("Complete! Check the 'vit_output' directory for results.")

if __name__ == "__main__":
    main() 