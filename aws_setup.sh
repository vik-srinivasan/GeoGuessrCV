#!/bin/bash
# AWS Setup Script for GeoGuessr ViT Training
# This script helps you set up an AWS EC2 GPU instance for training

echo "=== GeoGuessr ViT AWS Setup ==="
echo "This script will help you set up AWS for training"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on AWS
check_aws_environment() {
    print_status "Checking if running on AWS EC2..."
    if [ -f /sys/hypervisor/uuid ] && [ `head -c 3 /sys/hypervisor/uuid` == "ec2" ]; then
        print_success "Running on AWS EC2"
        return 0
    else
        print_warning "Not running on AWS EC2"
        return 1
    fi
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    sudo apt-get update -y
    sudo apt-get upgrade -y
    print_success "System updated"
}

# Install NVIDIA drivers for GPU instances
install_nvidia_drivers() {
    print_status "Installing NVIDIA drivers..."
    
    # Check if NVIDIA drivers are already installed
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA drivers already installed"
        nvidia-smi
        return 0
    fi
    
    # Install NVIDIA drivers
    sudo apt-get install -y nvidia-driver-470
    
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    
    print_success "NVIDIA drivers installed. Please reboot the instance."
}

# Install Python and pip
install_python() {
    print_status "Installing Python 3.9..."
    sudo apt-get install -y python3.9 python3.9-pip python3.9-venv python3.9-dev
    
    # Set python3.9 as default python3
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
    
    # Install pip
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    
    print_success "Python 3.9 installed"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    
    # Install other dependencies
    pip install transformers datasets accelerate
    pip install pillow numpy matplotlib seaborn scikit-learn tqdm
    pip install wandb kaggle
    
    print_success "Dependencies installed"
}

# Setup Kaggle
setup_kaggle() {
    print_status "Setting up Kaggle..."
    
    echo "Please place your kaggle.json file in ~/.kaggle/"
    echo "You can download it from: https://www.kaggle.com/settings"
    echo ""
    read -p "Press Enter when you've placed the kaggle.json file..."
    
    # Create kaggle directory if it doesn't exist
    mkdir -p ~/.kaggle
    
    # Set permissions
    chmod 600 ~/.kaggle/kaggle.json
    
    print_success "Kaggle configured"
}

# Download dataset
download_dataset() {
    print_status "Downloading GeoGuessr dataset..."
    
    # Create data directory
    mkdir -p data
    cd data
    
    # Download dataset
    kaggle datasets download -d ubitquitin/geolocation-geoguessr-images-50k
    
    # Extract dataset
    unzip geolocation-geoguessr-images-50k.zip
    
    print_success "Dataset downloaded and extracted"
    cd ..
}

# Create training script launcher
create_launcher() {
    print_status "Creating training launcher script..."
    
    cat > launch_training.sh << 'EOF'
#!/bin/bash
# Training launcher script

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Default training parameters
DATA_DIR="./data"
OUTPUT_DIR="./vit_aws_output"
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=2e-5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Starting training with parameters:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Run training
python3 train_geo_vit_aws.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --early_stopping \
    $USE_WANDB

echo "Training completed!"
EOF

    chmod +x launch_training.sh
    print_success "Training launcher created"
}

# Main setup function
main() {
    print_status "Starting AWS setup for GeoGuessr ViT training..."
    echo ""
    
    # Check if on AWS
    if check_aws_environment; then
        # Update system
        update_system
        
        # Install NVIDIA drivers for GPU instances
        read -p "Install NVIDIA drivers? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_nvidia_drivers
        fi
        
        # Install Python
        install_python
        
        # Install dependencies
        install_dependencies
        
        # Setup Kaggle
        read -p "Setup Kaggle for dataset download? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            setup_kaggle
            download_dataset
        fi
        
        # Create launcher
        create_launcher
        
        print_success "AWS setup completed!"
        echo ""
        echo "Next steps:"
        echo "1. If you installed NVIDIA drivers, reboot the instance"
        echo "2. Activate the virtual environment: source venv/bin/activate"
        echo "3. Start training: ./launch_training.sh"
        echo ""
        
    else
        print_error "This script is designed for AWS EC2 instances"
        echo "If you want to run locally, use the manual installation steps."
    fi
}

# Run main function
main "$@" 