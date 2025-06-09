#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Make predictions using the trained Vision Transformer model.

Usage:
  python predict_country.py --image path/to/image.jpg --model path/to/model
"""

import argparse
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

def predict_country(image_path, model_path, top_k=5):
    """
    Predict the country of an image using a trained ViT model.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the saved model
        top_k: Number of top predictions to return
    
    Returns:
        List of (country, probability) tuples
    """
    # Load the model and processor
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    
    # Get top k predictions
    values, indices = torch.topk(probs, top_k)
    
    # Map indices to country names
    id2label = model.config.id2label
    predictions = [(id2label[idx.item()], val.item()) for idx, val in zip(indices, values)]
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Predict country from image using ViT")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--top-k", type=int, default=5, help="Show top K predictions")
    args = parser.parse_args()
    
    predictions = predict_country(args.image, args.model, args.top_k)
    
    print(f"\nTop {args.top_k} predictions for {args.image}:")
    print("-" * 50)
    for i, (country, probability) in enumerate(predictions):
        print(f"{i+1}. {country:<20} {probability:.4f} ({probability*100:.1f}%)")

if __name__ == "__main__":
    main() 