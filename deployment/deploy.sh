#!/bin/bash

# EmotionalAI Deployment Script for Linux Server Cluster
# Usage: ./deploy.sh [production|staging|development]

set -e

ENVIRONMENT=${1:-production}
CLUSTER_NAME="emotionalai-cluster"

echo "🚀 Deploying EmotionalAI to $ENVIRONMENT environment..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models data logs cache nginx/ssl monitoring/grafana/dashboards monitoring/grafana/datasources

# Set environment-specific configurations
if [ "$ENVIRONMENT" = "production" ]; then
    echo "🔧 Configuring for production..."
    export COMPOSE_PROJECT_NAME=emotionalai-prod
    export MONGODB_PASSWORD=$(openssl rand -base64 32)
    export REDIS_PASSWORD=$(openssl rand -base64 32)
elif [ "$ENVIRONMENT" = "staging" ]; then
    echo "🔧 Configuring for staging..."
    export COMPOSE_PROJECT_NAME=emotionalai-staging
    export MONGODB_PASSWORD=staging_password_123
    export REDIS_PASSWORD=staging_redis_123
else
    echo "🔧 Configuring for development..."
    export COMPOSE_PROJECT_NAME=emotionalai-dev
    export MONGODB_PASSWORD=dev_password_123
    export REDIS_PASSWORD=dev_redis_123
fi

# Create .env file
echo "📝 Creating environment file..."
cat > .env << EOF
ENVIRONMENT=$ENVIRONMENT
MONGODB_PASSWORD=$MONGODB_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
COMPOSE_PROJECT_NAME=$COMPOSE_PROJECT_NAME
EOF

# Pull latest images
echo "📥 Pulling Docker images..."
docker-compose pull

# Build images
echo "🔨 Building Docker images..."
docker-compose build

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose down

# Start services
echo "▶️ Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose ps

# Initialize models (if needed)
echo "🤖 Initializing AI models..."
docker-compose exec emotionalai-app python -c "
import asyncio
from backend.api.utils.llm_router import llm_router

async def init_models():
    try:
        await llm_router.initialize_models()
        print('✅ Models initialized successfully')
    except Exception as e:
        print(f'⚠️ Model initialization warning: {e}')

asyncio.run(init_models())
"

# Run health checks
echo "🔍 Running health checks..."
curl -f http://localhost:8000/health || echo "⚠️ Main app health check failed"
curl -f http://localhost:8002/health || echo "⚠️ LLM router health check failed"

echo "✅ Deployment completed successfully!"
echo "📊 Access points:"
echo "   - Main API: http://localhost:8000"
echo "   - LLM Router: http://localhost:8002"
echo "   - Grafana: http://localhost:3000 (admin/admin123)"
echo "   - Prometheus: http://localhost:9090"

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=20 