#!/bin/bash
# Container Hardening Test Script

echo "🐳 ProjectAlpha Container Hardening Test"
echo "========================================"

# Check if Docker Compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed or not in PATH"
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p logs data/redis data/mongodb

# Build and start containers
echo "🔨 Building containers..."
docker-compose build

echo "🚀 Starting containers..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Test health endpoints
echo "🏥 Testing health endpoints..."

# Test backend health
echo "Testing backend health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend health check passed"
    curl -s http://localhost:8000/health | jq '.'
else
    echo "❌ Backend health check failed"
fi

# Check container security
echo "🔒 Checking container security..."

# Check if containers are running as non-root
backend_user=$(docker-compose exec -T backend id -u)
if [ "$backend_user" != "0" ]; then
    echo "✅ Backend container running as non-root user (UID: $backend_user)"
else
    echo "❌ Backend container running as root"
fi

# Check read-only filesystem
if docker-compose exec -T backend touch /test_file 2>/dev/null; then
    echo "❌ Root filesystem is writable"
else
    echo "✅ Root filesystem is read-only"
fi

# Check if tmp is writable
if docker-compose exec -T backend touch /app/tmp/test_file 2>/dev/null; then
    echo "✅ Tmp directory is writable"
    docker-compose exec -T backend rm -f /app/tmp/test_file
else
    echo "❌ Tmp directory is not writable"
fi

echo "🎉 Container hardening test complete!"
echo "Use 'docker-compose logs' to view container logs"
echo "Use 'docker-compose down' to stop containers"
