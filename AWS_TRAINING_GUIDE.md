# AWS Training Guide for GeoGuessr Vision Transformer

This guide will walk you through setting up and training your Vision Transformer model on AWS using GPU instances for the Kaggle GeoGuessr dataset.

## 🚀 Quick Start

1. **Launch AWS EC2 Instance**
2. **Upload and run setup script**
3. **Download dataset**
4. **Start training**

## 📋 Prerequisites

- AWS account with credits
- Kaggle account and API key
- Basic familiarity with AWS EC2

## 🖥️ Step 1: Launch AWS EC2 Instance

### Recommended Instance Types

**For Development/Testing:**
- `g4dn.xlarge` - 1x NVIDIA T4, 4 vCPUs, 16GB RAM (~$0.526/hour)
- Best for initial testing and smaller datasets

**For Full Training:**
- `g4dn.2xlarge` - 1x NVIDIA T4, 8 vCPUs, 32GB RAM (~$0.752/hour)
- `g4dn.4xlarge` - 1x NVIDIA T4, 16 vCPUs, 64GB RAM (~$1.204/hour)

**For High Performance:**
- `p3.2xlarge` - 1x NVIDIA V100, 8 vCPUs, 61GB RAM (~$3.06/hour)
- `p3.8xlarge` - 4x NVIDIA V100, 32 vCPUs, 244GB RAM (~$12.24/hour)

### Launch Configuration

1. **Go to AWS EC2 Console**
2. **Launch Instance:**
   - **AMI:** Ubuntu Server 20.04 LTS (HVM), SSD Volume Type
   - **Instance Type:** Choose from above recommendations
   - **Storage:** 100GB+ (dataset is ~10GB, need space for models/checkpoints)
   - **Security Group:** SSH (port 22) from your IP

3. **Launch and connect via SSH**

## 🔧 Step 2: Setup Environment

### Option A: Automated Setup (Recommended)

```bash
# Upload your files to the instance
scp -i your-key.pem train_geo_vit_aws.py aws_setup.sh ubuntu@your-instance-ip:~/
scp -i your-key.pem requirements_aws.txt ubuntu@your-instance-ip:~/

# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Run setup script
chmod +x aws_setup.sh
./aws_setup.sh
```

### Option B: Manual Setup

```bash
# Update system
sudo apt-get update -y && sudo apt-get upgrade -y

# Install Python 3.9
sudo apt-get install -y python3.9 python3.9-pip python3.9-venv python3.9-dev

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install -r requirements_aws.txt
```

## 📦 Step 3: Download Dataset

### Setup Kaggle API

1. **Get Kaggle API credentials:**
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Click "Create New API Token"
   - Download `kaggle.json`

2. **Upload kaggle.json to instance:**
   ```bash
   # On your local machine
   scp -i your-key.pem kaggle.json ubuntu@your-instance-ip:~/.kaggle/
   
   # On the instance
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download dataset:**
   ```bash
   mkdir -p data && cd data
   kaggle datasets download -d ubitquitin/geolocation-geoguessr-images-50k
   unzip geolocation-geoguessr-images-50k.zip
   cd ..
   ```

## 🏃‍♂️ Step 4: Start Training

### Basic Training

```bash
# Activate environment
source venv/bin/activate

# Start training
python3 train_geo_vit_aws.py \
    --data_dir ./data \
    --output_dir ./vit_aws_output \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --early_stopping
```

### Advanced Training with Monitoring

```bash
# Login to Weights & Biases (optional but recommended)
wandb login

# Training with W&B monitoring
python3 train_geo_vit_aws.py \
    --data_dir ./data \
    --output_dir ./vit_aws_output \
    --epochs 15 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --early_stopping \
    --use_wandb
```

### Using the Launcher Script

```bash
# Basic usage
./launch_training.sh

# With custom parameters
./launch_training.sh --epochs 15 --batch_size 32 --use_wandb
```

## ⚙️ Training Configuration

### Batch Size Guidelines

| Instance Type | VRAM | Recommended Batch Size |
|---------------|------|----------------------|
| g4dn.xlarge   | 16GB | 16-24               |
| g4dn.2xlarge  | 16GB | 16-32               |
| p3.2xlarge    | 16GB | 24-48               |
| p3.8xlarge    | 64GB | 64-128              |

### Hyperparameter Recommendations

```bash
# Conservative (safer, slower)
--epochs 10 --batch_size 16 --learning_rate 1e-5

# Balanced (recommended)
--epochs 15 --batch_size 32 --learning_rate 2e-5

# Aggressive (faster, higher risk)
--epochs 20 --batch_size 48 --learning_rate 5e-5
```

## 📊 Monitoring Training

### Option 1: Weights & Biases (Recommended)

```bash
# Setup W&B account at https://wandb.ai/
wandb login

# Train with W&B
python3 train_geo_vit_aws.py --use_wandb ...
```

View real-time metrics at: https://wandb.ai/

### Option 2: Local Logs

```bash
# Monitor training logs
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## 💾 Model Outputs

After training, you'll find:

```
vit_aws_output/
├── final_model/           # Final trained model
├── checkpoints/           # Training checkpoints
├── training_results.json  # Performance metrics
├── label_mappings.json    # Country label mappings
└── logs/                  # Training logs
```

## 🔄 Resuming Training

```bash
# Resume from checkpoint
python3 train_geo_vit_aws.py \
    --data_dir ./data \
    --output_dir ./vit_aws_output \
    --resume_from_checkpoint ./vit_aws_output/checkpoints/checkpoint-1000
```

## 💰 Cost Optimization

### Instance Management

```bash
# Stop instance when not training (saves money)
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance when needed
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

### Training Efficiency

1. **Use mixed precision training** (enabled by default on GPU)
2. **Enable gradient checkpointing** (reduces memory usage)
3. **Use early stopping** (prevents overfitting and saves time)
4. **Monitor training closely** (stop if not improving)

### Expected Costs

| Instance Type | Training Time | Estimated Cost |
|---------------|---------------|----------------|
| g4dn.xlarge   | 8-12 hours    | $4-6          |
| g4dn.2xlarge  | 6-8 hours     | $5-6          |
| p3.2xlarge    | 3-4 hours     | $9-12         |

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
--batch_size 8

# Enable gradient accumulation
--gradient_accumulation_steps 2
```

**Slow Training:**
```bash
# Check GPU utilization
nvidia-smi

# Increase batch size if GPU not fully utilized
--batch_size 32
```

**Dataset Issues:**
```bash
# Verify dataset structure
ls -la data/
# Should show country folders like: Argentina/, Brazil/, etc.
```

## 📈 Expected Results

Based on the ResNet baseline (41.3% top-1, 59.2% top-5), expect:

- **ViT Performance:** 45-55% top-1 accuracy, 65-75% top-5 accuracy
- **Training Time:** 4-12 hours depending on instance
- **Dataset:** ~50K images across 50+ countries

## 🔍 Model Evaluation

After training, test your model:

```bash
# Use the prediction script
python3 predict_country.py \
    --model_path ./vit_aws_output/final_model \
    --image_path test_image.jpg \
    --top_k 5
```

## 📋 Next Steps

1. **Download trained model** to local machine
2. **Deploy model** for inference
3. **Fine-tune** on additional data
4. **Compare** with ResNet baseline

## 💡 Pro Tips

1. **Use tmux/screen** to keep training running if connection drops
2. **Set up CloudWatch** for instance monitoring
3. **Create AMI** after setup for future use
4. **Use Spot Instances** for 70% cost savings (with risk of termination)
5. **Monitor AWS billing** regularly

---

🎯 **Ready to train?** Follow the steps above and you'll have a state-of-the-art Vision Transformer trained on the GeoGuessr dataset! 