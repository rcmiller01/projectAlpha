#!/bin/bash

# EmotionalAI UCS M3 Cluster Deployment Script
# This script deploys the EmotionalAI system across your UCS M3 servers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="emotional_ai_cluster"
NETWORK_SUBNET="172.20.0.0/16"
MONGODB_PASSWORD="emotional_ai_password"
GRAFANA_PASSWORD="emotional_ai_admin"

# Server configurations (adjust IPs to match your network)
M3_PRIMARY_IP="192.168.1.10"
M3_GPU_IP="192.168.1.11"

echo -e "${BLUE}🚀 EmotionalAI UCS M3 Cluster Deployment${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check NVIDIA Docker (for GPU server)
    if ! command -v nvidia-docker &> /dev/null; then
        print_warning "NVIDIA Docker not found. GPU features may not work properly."
    fi
    
    # Check if ports are available
    local ports=(80 443 8000 8001 27017 6379 9090 3000)
    for port in "${ports[@]}"; do
        if netstat -tuln | grep -q ":$port "; then
            print_warning "Port $port is already in use. Make sure it's not needed by another service."
        fi
    done
    
    print_status "Prerequisites check completed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p config/mongodb
    mkdir -p config/nginx/ssl
    mkdir -p config/prometheus
    mkdir -p config/grafana/dashboards
    mkdir -p config/grafana/datasources
    mkdir -p logs
    mkdir -p data/mongodb
    mkdir -p data/redis
    mkdir -p data/prometheus
    mkdir -p data/grafana
    
    print_status "Directories created"
}

# Create MongoDB initialization script
create_mongodb_init() {
    print_status "Creating MongoDB initialization script..."
    
    cat > config/mongodb/init.js << EOF
// MongoDB initialization script for EmotionalAI
db = db.getSiblingDB('emotional_ai');

// Create collections with proper indexes
db.createCollection('memories');
db.createCollection('threads');
db.createCollection('relationships');
db.createCollection('biometrics');
db.createCollection('avatars');
db.createCollection('sessions');
db.createCollection('analytics');

// Create indexes for better performance
db.memories.createIndex({"user_id": 1});
db.memories.createIndex({"tags": 1});
db.memories.createIndex({"memory_type": 1});
db.memories.createIndex({"created_at": -1});
db.memories.createIndex({"emotional_tags": 1});

db.threads.createIndex({"user_id": 1});
db.threads.createIndex({"thread_id": 1});
db.threads.createIndex({"created_at": -1});
db.threads.createIndex({"tags": 1});

db.relationships.createIndex({"user_id": 1});
db.relationships.createIndex({"persona": 1});
db.relationships.createIndex({"created_at": -1});

db.biometrics.createIndex({"user_id": 1});
db.biometrics.createIndex({"timestamp": -1});

db.sessions.createIndex({"user_id": 1});
db.sessions.createIndex({"session_id": 1});
db.sessions.createIndex({"created_at": -1});

print("MongoDB initialization completed");
EOF
    
    print_status "MongoDB initialization script created"
}

# Create Prometheus configuration
create_prometheus_config() {
    print_status "Creating Prometheus configuration..."
    
    cat > config/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'emotional_ai_primary'
    static_configs:
      - targets: ['m3-primary:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'emotional_ai_gpu'
    static_configs:
      - targets: ['m3-gpu:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb:27017']
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: '/nginx_status'
    scrape_interval: 30s
EOF
    
    print_status "Prometheus configuration created"
}

# Create Grafana datasource configuration
create_grafana_datasource() {
    print_status "Creating Grafana datasource configuration..."
    
    mkdir -p config/grafana/datasources
    
    cat > config/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    print_status "Grafana datasource configuration created"
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    cat > .env << EOF
# EmotionalAI Cluster Configuration
CLUSTER_NAME=$CLUSTER_NAME
NETWORK_SUBNET=$NETWORK_SUBNET

# MongoDB Configuration
MONGODB_ROOT_USERNAME=admin
MONGODB_ROOT_PASSWORD=$MONGODB_PASSWORD
MONGODB_DATABASE=emotional_ai

# Server Configuration
M3_PRIMARY_IP=$M3_PRIMARY_IP
M3_GPU_IP=$M3_GPU_IP
M3_PRIMARY_PORT=8000
M3_GPU_PORT=8001

# Redis Configuration
REDIS_PASSWORD=emotional_ai_redis_password

# Grafana Configuration
GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASSWORD

# Security (for local deployment, no authentication needed)
AUTH_ENABLED=false
SSL_ENABLED=false

# GPU Configuration
GPU_ENABLED=true
NVIDIA_VISIBLE_DEVICES=all
EOF
    
    print_status "Environment configuration created"
}

# Deploy the cluster
deploy_cluster() {
    print_status "Deploying EmotionalAI cluster..."
    
    # Stop any existing containers
    docker-compose -f docker-compose.cluster.yml down --remove-orphans
    
    # Build and start services
    docker-compose -f docker-compose.cluster.yml up -d --build
    
    print_status "Cluster deployment initiated"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo -n "Checking services (attempt $attempt/$max_attempts)... "
        
        # Check MongoDB
        if docker-compose -f docker-compose.cluster.yml exec -T mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
            echo -e "${GREEN}MongoDB ✓${NC}"
        else
            echo -e "${YELLOW}MongoDB ⚠${NC}"
        fi
        
        # Check Primary Server
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}Primary Server ✓${NC}"
        else
            echo -e "${YELLOW}Primary Server ⚠${NC}"
        fi
        
        # Check GPU Server
        if curl -f http://localhost:8001/health > /dev/null 2>&1; then
            echo -e "${GREEN}GPU Server ✓${NC}"
        else
            echo -e "${YELLOW}GPU Server ⚠${NC}"
        fi
        
        # Check Nginx
        if curl -f http://localhost/health > /dev/null 2>&1; then
            echo -e "${GREEN}Nginx ✓${NC}"
            print_status "All services are ready!"
            return 0
        fi
        
        echo -e "${YELLOW}Nginx ⚠${NC}"
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Services failed to start within the expected time"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Display cluster information
display_cluster_info() {
    print_status "Cluster deployment completed successfully!"
    echo ""
    echo -e "${BLUE}🌐 Cluster Access Information:${NC}"
    echo "=================================="
    echo -e "Frontend Application: ${GREEN}http://localhost${NC}"
    echo -e "Primary Server API:   ${GREEN}http://localhost:8000${NC}"
    echo -e "GPU Server API:       ${GREEN}http://localhost:8001${NC}"
    echo -e "MongoDB:              ${GREEN}mongodb://localhost:27017${NC}"
    echo -e "Redis:                ${GREEN}redis://localhost:6379${NC}"
    echo ""
    echo -e "${BLUE}📊 Monitoring:${NC}"
    echo "================"
    echo -e "Prometheus:           ${GREEN}http://localhost:9090${NC}"
    echo -e "Grafana:              ${GREEN}http://localhost:3000${NC}"
    echo -e "Grafana Admin:        ${GREEN}admin / $GRAFANA_PASSWORD${NC}"
    echo ""
    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo "========================"
    echo -e "View logs:            ${YELLOW}docker-compose -f docker-compose.cluster.yml logs -f${NC}"
    echo -e "Stop cluster:         ${YELLOW}docker-compose -f docker-compose.cluster.yml down${NC}"
    echo -e "Restart cluster:      ${YELLOW}docker-compose -f docker-compose.cluster.yml restart${NC}"
    echo -e "Scale services:       ${YELLOW}docker-compose -f docker-compose.cluster.yml up -d --scale m3-primary=2${NC}"
    echo ""
    echo -e "${BLUE}📝 Next Steps:${NC}"
    echo "============="
    echo "1. Access the application at http://localhost"
    echo "2. Configure Grafana dashboards for monitoring"
    echo "3. Set up SSL certificates if needed"
    echo "4. Configure backup strategies for MongoDB"
    echo "5. Monitor cluster health and performance"
    echo ""
}

# Main deployment process
main() {
    echo -e "${BLUE}Starting EmotionalAI UCS M3 cluster deployment...${NC}"
    echo ""
    
    check_prerequisites
    create_directories
    create_mongodb_init
    create_prometheus_config
    create_grafana_datasource
    create_env_file
    deploy_cluster
    
    if wait_for_services; then
        display_cluster_info
    else
        print_error "Deployment failed. Check logs with: docker-compose -f docker-compose.cluster.yml logs"
        exit 1
    fi
}

# Run main function
main "$@" 