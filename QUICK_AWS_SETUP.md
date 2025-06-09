# Quick AWS Setup for GeoGuessr ViT Training

## 🚀 TL;DR - Get Training in 15 Minutes

### 1. Launch AWS Instance
- **Instance Type:** `g4dn.xlarge` or `g4dn.2xlarge` 
- **AMI:** Ubuntu 20.04 LTS
- **Storage:** 100GB
- **Security:** SSH access

### 2. Upload Files
```bash
scp -i your-key.pem train_geo_vit_aws.py aws_setup.sh requirements_aws.txt ubuntu@your-ip:~/
```

### 3. Setup Environment
```bash
ssh -i your-key.pem ubuntu@your-ip
chmod +x aws_setup.sh
./aws_setup.sh
```

### 4. Get Kaggle API Key
- Go to https://www.kaggle.com/settings
- Download `kaggle.json`
- Upload: `scp -i your-key.pem kaggle.json ubuntu@your-ip:~/.kaggle/`

### 5. Start Training
```bash
source venv/bin/activate
./launch_training.sh --use_wandb
```

## 💰 Expected Costs
- **g4dn.xlarge:** ~$4-6 for full training
- **g4dn.2xlarge:** ~$5-6 for full training  

## 📊 Expected Results
- **Accuracy:** 45-55% (vs 41% ResNet baseline)
- **Time:** 4-8 hours depending on instance
- **Dataset:** 50K images, 50+ countries

## 🔗 Useful Commands
```bash
# Monitor training
tail -f training.log

# Check GPU usage  
nvidia-smi

# Resume training
python3 train_geo_vit_aws.py --resume_from_checkpoint ./vit_aws_output/checkpoints/checkpoint-X

# Download results
scp -r -i your-key.pem ubuntu@your-ip:~/vit_aws_output ./
```

**Full guide:** See `AWS_TRAINING_GUIDE.md` for detailed instructions. 