#!/usr/bin/env python3
"""
AWS-optimized Vision Transformer training for GeoGuessr country classification
Dataset: https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k

Usage:
    python train_geo_vit_aws.py --data_dir /path/to/dataset --output_dir ./vit_aws_output
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from collections import Counter
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoImageProcessor, 
    ViTForImageClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import wandb
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeoGuessrDatasetLoader:
    """Loads and preprocesses the Kaggle GeoGuessr dataset"""
    
    def __init__(self, data_dir, min_samples_per_country=10, max_samples_per_country=None):
        self.data_dir = Path(data_dir)
        self.min_samples_per_country = min_samples_per_country
        self.max_samples_per_country = max_samples_per_country
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
    def load_dataset(self):
        """Load images and labels from folder structure"""
        logger.info(f"Loading dataset from {self.data_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        # Find all country folders
        country_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        logger.info(f"Found {len(country_folders)} country folders")
        
        images = []
        labels = []
        country_counts = Counter()
        
        # Process each country folder
        for country_folder in tqdm(country_folders, desc="Loading countries"):
            country_name = country_folder.name
            image_files = []
            
            # Find all image files in the country folder
            for ext in self.valid_extensions:
                image_files.extend(country_folder.glob(f"*{ext}"))
                image_files.extend(country_folder.glob(f"*{ext.upper()}"))
            
            if len(image_files) < self.min_samples_per_country:
                logger.warning(f"Skipping {country_name}: only {len(image_files)} images (min: {self.min_samples_per_country})")
                continue
            
            # Limit samples per country if specified
            if self.max_samples_per_country and len(image_files) > self.max_samples_per_country:
                image_files = image_files[:self.max_samples_per_country]
            
            # Load images
            for img_path in image_files:
                try:
                    # Verify image can be opened
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    images.append(str(img_path))
                    labels.append(country_name)
                    country_counts[country_name] += 1
                    
                except Exception as e:
                    logger.warning(f"Skipping corrupted image {img_path}: {e}")
        
        logger.info(f"Loaded {len(images)} images from {len(country_counts)} countries")
        
        # Create label mappings
        unique_countries = sorted(list(set(labels)))
        label_to_id = {country: idx for idx, country in enumerate(unique_countries)}
        id_to_label = {idx: country for country, idx in label_to_id.items()}
        
        # Convert labels to integers
        label_ids = [label_to_id[label] for label in labels]
        
        # Log dataset statistics
        self.log_dataset_stats(country_counts, unique_countries)
        
        return images, label_ids, label_to_id, id_to_label
    
    def log_dataset_stats(self, country_counts, countries):
        """Log dataset statistics"""
        total_images = sum(country_counts.values())
        logger.info(f"Dataset Statistics:")
        logger.info(f"  Total images: {total_images}")
        logger.info(f"  Total countries: {len(countries)}")
        logger.info(f"  Average images per country: {total_images / len(countries):.1f}")
        logger.info(f"  Min images per country: {min(country_counts.values())}")
        logger.info(f"  Max images per country: {max(country_counts.values())}")

class GeoGuessrTrainer:
    """AWS-optimized trainer for GeoGuessr ViT model"""
    
    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.setup_output_dir()
        
    def setup_device(self):
        """Setup optimal device for training"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon)")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
    
    def setup_output_dir(self):
        """Setup output directory"""
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
    def create_datasets(self, images, labels, label_to_id, id_to_label):
        """Create train/val/test datasets with stratified split"""
        logger.info("Creating datasets with stratified split...")
        
        # Stratified split: 80% train, 10% val, 10% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=0.1, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp  # 0.111 * 0.9 ≈ 0.1
        )
        
        logger.info(f"Dataset splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict({"image": X_train, "label": y_train})
        val_dataset = Dataset.from_dict({"image": X_val, "label": y_val})
        test_dataset = Dataset.from_dict({"image": X_test, "label": y_test})
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        # Save label mappings
        label_info = {
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "num_classes": len(label_to_id)
        }
        
        with open(self.output_dir / "label_mappings.json", "w") as f:
            json.dump(label_info, f, indent=2)
        
        return dataset_dict, label_info
    
    def setup_model_and_processor(self, num_classes, id_to_label):
        """Setup ViT model and image processor"""
        logger.info("Setting up ViT model and processor...")
        
        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224",
            do_rescale=True,
            do_normalize=True
        )
        
        # Load model
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            id2label=id_to_label,
            label2id={v: k for k, v in id_to_label.items()},
            ignore_mismatched_sizes=True,
        )
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model, self.processor
    
    def transform_dataset(self, dataset):
        """Transform dataset for training"""
        def transform(examples):
            # Load and process images
            images = []
            for img_path in examples["image"]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Error loading image {img_path}: {e}")
                    # Create a black placeholder image
                    img = Image.new("RGB", (224, 224), color="black")
                    images.append(img)
            
            # Process with ViT processor
            inputs = self.processor(images, return_tensors="pt")
            inputs["labels"] = examples["label"]
            return inputs
        
        dataset.set_transform(transform)
        return dataset
    
    def create_trainer(self, train_dataset, val_dataset):
        """Create HuggingFace trainer with AWS optimizations"""
        
        # Training arguments optimized for AWS GPU
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            
            # Learning rate and optimization
            learning_rate=self.args.learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            lr_scheduler_type="cosine",
            
            # Evaluation and logging
            eval_strategy="steps",
            eval_steps=500,
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,
            
            # Performance optimizations
            dataloader_num_workers=4,
            fp16=torch.cuda.is_available(),  # Mixed precision for GPU
            gradient_checkpointing=True,
            remove_unused_columns=False,
            
            # Early stopping and best model
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            
            # Reproducibility
            seed=42,
            data_seed=42,
            
            # Reporting
            report_to="wandb" if self.args.use_wandb else "none",
            run_name=f"geoguessr_vit_{int(time.time())}" if self.args.use_wandb else None,
        )
        
        # Metrics computation
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            # Calculate accuracy
            accuracy = (predictions == labels).mean()
            
            # Calculate top-5 accuracy
            top5_predictions = np.argsort(eval_pred[0], axis=1)[:, -5:]
            top5_accuracy = np.mean([label in top5_pred for label, top5_pred in zip(labels, top5_predictions)])
            
            return {
                "accuracy": accuracy,
                "top5_accuracy": top5_accuracy
            }
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if self.args.early_stopping else [],
        )
        
        return trainer
    
    def train(self):
        """Main training pipeline"""
        logger.info("Starting GeoGuessr ViT training on AWS...")
        
        # Initialize wandb if requested
        if self.args.use_wandb:
            wandb.init(
                project="geoguessr-vit-aws",
                config=vars(self.args),
                name=f"vit_training_{int(time.time())}"
            )
        
        try:
            # Load dataset
            loader = GeoGuessrDatasetLoader(
                self.args.data_dir,
                min_samples_per_country=self.args.min_samples_per_country,
                max_samples_per_country=self.args.max_samples_per_country
            )
            images, labels, label_to_id, id_to_label = loader.load_dataset()
            
            # Create datasets
            datasets, label_info = self.create_datasets(images, labels, label_to_id, id_to_label)
            
            # Setup model
            model, processor = self.setup_model_and_processor(label_info["num_classes"], id_to_label)
            
            # Transform datasets
            train_dataset = self.transform_dataset(datasets["train"])
            val_dataset = self.transform_dataset(datasets["validation"])
            test_dataset = self.transform_dataset(datasets["test"])
            
            # Create trainer
            trainer = self.create_trainer(train_dataset, val_dataset)
            
            # Train model
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save final model
            trainer.save_model(str(self.output_dir / "final_model"))
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate(eval_dataset=test_dataset)
            
            # Save results
            results = {
                "train_results": train_result.metrics,
                "test_results": test_results,
                "model_config": {
                    "num_classes": label_info["num_classes"],
                    "model_name": "google/vit-base-patch16-224",
                    "dataset_size": len(images)
                }
            }
            
            with open(self.output_dir / "training_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Training completed! Results saved to {self.output_dir}")
            logger.info(f"Test accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
            logger.info(f"Test top-5 accuracy: {test_results.get('eval_top5_accuracy', 'N/A'):.4f}")
            
            if self.args.use_wandb:
                wandb.log({"final_test_accuracy": test_results.get('eval_accuracy', 0)})
                wandb.finish()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.args.use_wandb:
                wandb.finish()
            raise

def main():
    parser = argparse.ArgumentParser(description="Train ViT on GeoGuessr dataset (AWS optimized)")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./vit_aws_output", help="Output directory")
    parser.add_argument("--min_samples_per_country", type=int, default=10, help="Minimum samples per country")
    parser.add_argument("--max_samples_per_country", type=int, default=None, help="Maximum samples per country")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    
    # Monitoring
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    trainer = GeoGuessrTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main() 