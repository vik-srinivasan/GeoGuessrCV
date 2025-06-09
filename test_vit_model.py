#!/usr/bin/env python
"""
Test script for Vision Transformer GeoGuessr model.
This script can run in different modes:
1. Synthetic data test (no dataset required)
2. Small dataset test (if you have some sample images)
3. Full dataset test (if you have the complete geo50k dataset)
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import Dataset
import random
import time

def ts():
    return time.strftime("[%H:%M:%S]")

def create_synthetic_dataset(num_samples=100, num_classes=10):
    """Create a synthetic dataset for testing"""
    print(f"{ts()} Creating synthetic dataset with {num_samples} samples and {num_classes} classes")
    
    # Create random images (224x224 RGB)
    images = []
    labels = []
    
    for i in range(num_samples):
        # Create a random image with some pattern based on class
        class_id = i % num_classes
        
        # Create image with class-specific color pattern
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some class-specific pattern
        img_array[:, :, class_id % 3] = (img_array[:, :, class_id % 3] * 0.7 + class_id * 25) % 255
        
        images.append(Image.fromarray(img_array))
        labels.append(class_id)
    
    # Create class names
    class_names = [f"TestCountry_{i}" for i in range(num_classes)]
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict({
        "image": images,
        "label": labels
    })
    
    return dataset, class_names

def test_model_creation(num_classes):
    """Test if we can create and initialize the ViT model"""
    print(f"{ts()} Testing ViT model creation for {num_classes} classes...")
    
    try:
        # Load processor
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        print(f"{ts()} ✓ Image processor loaded successfully")
        
        # Create model
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            id2label={i: f"TestCountry_{i}" for i in range(num_classes)},
            label2id={f"TestCountry_{i}": i for i in range(num_classes)},
            ignore_mismatched_sizes=True,
        )
        print(f"{ts()} ✓ ViT model created successfully")
        print(f"{ts()} Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model, processor
        
    except Exception as e:
        print(f"{ts()} ✗ Error creating model: {e}")
        return None, None

def test_forward_pass(model, processor, test_image):
    """Test a forward pass through the model"""
    print(f"{ts()} Testing forward pass...")
    
    try:
        # Process image
        inputs = processor(images=test_image, return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        print(f"{ts()} ✓ Forward pass successful")
        print(f"{ts()} Output shape: {logits.shape}")
        print(f"{ts()} Max probability: {probabilities.max().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"{ts()} ✗ Forward pass failed: {e}")
        return False

def test_training_step(model, processor, dataset, class_names):
    """Test a few training steps"""
    print(f"{ts()} Testing training setup...")
    
    try:
        # Transform function
        def transform(examples):
            inputs = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
            inputs["labels"] = examples["label"]
            return inputs
        
        # Apply transform
        dataset.set_transform(transform)
        
        # Split dataset
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        print(f"{ts()} ✓ Dataset preparation successful")
        print(f"{ts()} Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        
        # Training arguments for quick test
        training_args = TrainingArguments(
            output_dir="./test_vit_output",
            num_train_epochs=1,  # Just 1 epoch for testing
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_steps=5,
            eval_steps=20,
            save_steps=50,
            learning_rate=5e-5,
            weight_decay=0.01,
            remove_unused_columns=False,
            load_best_model_at_end=False,  # Skip for quick test
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor,
        )
        
        print(f"{ts()} ✓ Trainer created successfully")
        
        # Try a few training steps
        print(f"{ts()} Running a few training steps...")
        trainer.train()
        
        print(f"{ts()} ✓ Training test completed successfully")
        
        # Quick evaluation
        eval_result = trainer.evaluate()
        print(f"{ts()} ✓ Evaluation completed")
        print(f"{ts()} Eval loss: {eval_result.get('eval_loss', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"{ts()} ✗ Training test failed: {e}")
        return False

def test_prediction_functionality(model, processor, test_image, class_names):
    """Test the prediction functionality"""
    print(f"{ts()} Testing prediction functionality...")
    
    try:
        model.eval()
        
        # Ensure model and inputs are on the same device
        device = next(model.parameters()).device
        inputs = processor(images=test_image, return_tensors="pt")
        
        # Move inputs to same device as model
        if hasattr(inputs, 'pixel_values'):
            inputs.pixel_values = inputs.pixel_values.to(device)
        for key in inputs:
            if hasattr(inputs[key], 'to'):
                inputs[key] = inputs[key].to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        top_5_probs, top_5_indices = torch.topk(probabilities, min(5, len(class_names)))
        
        print(f"{ts()} ✓ Prediction successful")
        print(f"{ts()} Top predictions:")
        for i, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices)):
            class_name = class_names[idx.item()] if idx.item() < len(class_names) else f"Class_{idx.item()}"
            print(f"{ts()}   {i+1}. {class_name}: {prob.item():.4f}")
            
        return True
        
    except Exception as e:
        print(f"{ts()} ✗ Prediction test failed: {e}")
        return False

def main():
    print("="*60)
    print("Vision Transformer GeoGuessr Model Test Suite")
    print("="*60)
    
    # Configuration
    NUM_CLASSES = 10
    NUM_SAMPLES = 50  # Small for quick testing
    
    # Test 1: Create synthetic dataset
    print(f"\n{ts()} TEST 1: Creating synthetic dataset")
    dataset, class_names = create_synthetic_dataset(NUM_SAMPLES, NUM_CLASSES)
    test_image = dataset[0]["image"]  # Use first image for testing
    
    # Test 2: Model creation
    print(f"\n{ts()} TEST 2: Model creation")
    model, processor = test_model_creation(NUM_CLASSES)
    if model is None:
        print(f"{ts()} ✗ Cannot proceed without model")
        return
    
    # Test 3: Forward pass
    print(f"\n{ts()} TEST 3: Forward pass")
    forward_success = test_forward_pass(model, processor, test_image)
    
    # Test 4: Training setup and short training
    print(f"\n{ts()} TEST 4: Training test")
    training_success = test_training_step(model, processor, dataset, class_names)
    
    # Test 5: Prediction functionality
    print(f"\n{ts()} TEST 5: Prediction functionality")
    prediction_success = test_prediction_functionality(model, processor, test_image, class_names)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Model Creation:    {'✓ PASS' if model is not None else '✗ FAIL'}")
    print(f"Forward Pass:      {'✓ PASS' if forward_success else '✗ FAIL'}")
    print(f"Training Setup:    {'✓ PASS' if training_success else '✗ FAIL'}")
    print(f"Prediction:        {'✓ PASS' if prediction_success else '✗ FAIL'}")
    
    all_tests_passed = all([
        model is not None,
        forward_success,
        training_success,
        prediction_success
    ])
    
    if all_tests_passed:
        print(f"\n{ts()} 🎉 ALL TESTS PASSED! Your ViT model is working correctly.")
        print(f"{ts()} You can now proceed to train on real data.")
    else:
        print(f"\n{ts()} ⚠️ Some tests failed. Please check the error messages above.")
    
    # Cleanup
    try:
        import shutil
        if os.path.exists("./test_vit_output"):
            shutil.rmtree("./test_vit_output")
        print(f"{ts()} Cleaned up test files")
    except:
        pass

if __name__ == "__main__":
    main() 