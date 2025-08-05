#!/bin/bash

# EmotionalAI System Dependencies Installation Script
# For Ubuntu/Debian Linux servers

set -e

echo "🔧 Installing EmotionalAI system dependencies..."

# Update package list
echo "📦 Updating package list..."
sudo apt-get update

# Install essential packages
echo "📦 Installing essential packages..."
sudo apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker $USER
    echo "✅ Docker installed successfully"
else
    echo "✅ Docker already installed"
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed successfully"
else
    echo "✅ Docker Compose already installed"
fi

# Install Ollama (for LLM models)
echo "🤖 Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed successfully"
else
    echo "✅ Ollama already installed"
fi

# Install multimedia dependencies
echo "🎵 Installing multimedia dependencies..."
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    libssl-dev \
    libffi-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly

# Install Python development dependencies
echo "🐍 Installing Python development dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libpython3-dev

# Install monitoring tools
echo "📊 Installing monitoring tools..."
sudo apt-get install -y \
    htop \
    iotop \
    nethogs \
    nginx \
    certbot \
    python3-certbot-nginx

# Install additional utilities
echo "🛠️ Installing additional utilities..."
sudo apt-get install -y \
    tree \
    jq \
    htop \
    tmux \
    vim \
    nano \
    unzip \
    zip

# Create system directories
echo "📁 Creating system directories..."
sudo mkdir -p /opt/emotionalai/{models,data,logs,cache}
sudo chown -R $USER:$USER /opt/emotionalai

# Set up firewall (if ufw is available)
if command -v ufw &> /dev/null; then
    echo "🔥 Configuring firewall..."
    sudo ufw allow 22/tcp    # SSH
    sudo ufw allow 80/tcp    # HTTP
    sudo ufw allow 443/tcp   # HTTPS
    sudo ufw allow 8000/tcp  # EmotionalAI API
    sudo ufw allow 8001/tcp  # WebSocket
    sudo ufw allow 8002/tcp  # LLM Router
    sudo ufw allow 3000/tcp  # Grafana
    sudo ufw allow 9090/tcp  # Prometheus
    echo "✅ Firewall configured"
fi

# Install Node.js (for frontend if needed)
echo "🟢 Installing Node.js..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    echo "✅ Node.js installed successfully"
else
    echo "✅ Node.js already installed"
fi

# Install PM2 (process manager)
echo "⚡ Installing PM2..."
if ! command -v pm2 &> /dev/null; then
    sudo npm install -g pm2
    echo "✅ PM2 installed successfully"
else
    echo "✅ PM2 already installed"
fi

# Set up log rotation
echo "📋 Setting up log rotation..."
sudo tee /etc/logrotate.d/emotionalai > /dev/null << EOF
/opt/emotionalai/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}
EOF

echo "✅ All dependencies installed successfully!"
echo ""
echo "🎉 System is ready for EmotionalAI deployment!"
echo ""
echo "📋 Next steps:"
echo "1. Clone the EmotionalAI repository"
echo "2. Run: ./deployment/deploy.sh production"
echo "3. Access the system at http://localhost:8000"
echo ""
echo "⚠️ Please restart your shell or run 'newgrp docker' to use Docker without sudo" 