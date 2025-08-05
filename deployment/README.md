# EmotionalAI Deployment Guide

## 🚀 Linux Server Cluster Deployment

This guide covers deploying the EmotionalAI system on a local Linux server cluster using Docker and Docker Compose.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ or Debian 11+
- **RAM**: Minimum 16GB (32GB+ recommended for LLM models)
- **Storage**: Minimum 100GB SSD (500GB+ recommended)
- **CPU**: 8+ cores (16+ cores recommended)
- **Network**: Stable internet connection for model downloads

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Git

---

## 🛠️ Installation

### 1. Install System Dependencies

```bash
# Run the automated installation script
chmod +x deployment/install-dependencies.sh
./deployment/install-dependencies.sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/rcmiller01/test2.git
cd test2
```

### 3. Configure Environment

```bash
# Copy and edit environment configuration
cp .env.example .env
nano .env
```

### 4. Deploy the System

```bash
# Deploy to production
chmod +x deployment/deploy.sh
./deployment/deploy.sh production

# Or deploy to staging
./deployment/deploy.sh staging

# Or deploy to development
./deployment/deploy.sh development
```

---

## 🐳 Docker Services

The system consists of the following Docker services:

### Core Services
- **emotionalai-app**: Main FastAPI application
- **llm-router**: LLM model routing service
- **mongodb**: Database for user data and memories
- **redis**: Caching and session storage

### LLM Models (Ollama-based)
- **mythomax**: MythoMax model for general conversation
- **openchat**: OpenChat model for technical discussions
- **qwen2**: Qwen2.5 model for sophisticated responses
- **kimik2**: KimiK2 model for creative content

### Monitoring & Infrastructure
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards
- **nginx**: Reverse proxy and load balancing

---

## 📊 Model Management

### Option 1: Docker-based Models (Recommended)

The models are automatically downloaded and managed by Docker containers using Ollama:

```bash
# Models are automatically pulled during deployment
# You can manually pull specific models:
docker-compose exec mythomax ollama pull mythomax
docker-compose exec openchat ollama pull openchat
docker-compose exec qwen2 ollama pull qwen2.5:7b
docker-compose exec kimik2 ollama pull kimik2
```

### Option 2: Local Model Files

If you prefer to manage models locally:

1. Create a `models/` directory in your project
2. Download models to appropriate subdirectories:
   ```
   models/
   ├── mythomax/
   ├── openchat/
   ├── qwen2/
   └── kimik2/
   ```
3. Update the Docker Compose configuration to mount local model directories

### Model Storage Locations

- **Docker Volumes**: `/var/lib/docker/volumes/emotionalai_*_data`
- **Local Mounts**: `./models/` (if using local files)
- **Cache**: `./cache/` for temporary model files

---

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Database
MONGODB_URL=mongodb://mongodb:27017/emotionalai
MONGODB_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://redis:6379
REDIS_PASSWORD=your_redis_password

# LLM Endpoints
MYTHOMAX_ENDPOINT=http://mythomax:11434
OPENCHAT_ENDPOINT=http://openchat:11434
QWEN2_ENDPOINT=http://qwen2:11434
KIMIK2_ENDPOINT=http://kimik2:11434

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Service Configuration

Each service can be configured independently:

```bash
# Edit service-specific configurations
nano docker-compose.yml
nano nginx/nginx.conf
nano monitoring/prometheus.yml
```

---

## 📈 Monitoring & Logs

### Access Points

- **Main API**: http://localhost:8000
- **LLM Router**: http://localhost:8002
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Nginx**: http://localhost:80

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f emotionalai-app
docker-compose logs -f llm-router
docker-compose logs -f mongodb

# Real-time logs
docker-compose logs -f --tail=100
```

### Health Checks

```bash
# Check service status
docker-compose ps

# Health check endpoints
curl http://localhost:8000/health
curl http://localhost:8002/health
```

---

## 🔄 Maintenance

### Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart services
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup

```bash
# Backup database
docker-compose exec mongodb mongodump --out /backup/$(date +%Y%m%d)

# Backup models (if using local files)
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml nginx/ monitoring/
```

### Scaling

```bash
# Scale specific services
docker-compose up -d --scale emotionalai-app=3
docker-compose up -d --scale llm-router=2
```

---

## 🚨 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check what's using the ports
   sudo netstat -tulpn | grep :8000
   sudo netstat -tulpn | grep :27017
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats
   free -h
   ```

3. **Model Download Failures**
   ```bash
   # Check model status
   docker-compose exec mythomax ollama list
   docker-compose logs mythomax
   ```

4. **Database Connection Issues**
   ```bash
   # Check MongoDB status
   docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"
   ```

### Performance Optimization

1. **GPU Support** (if available)
   ```bash
   # Uncomment GPU lines in requirements.txt
   # Add GPU runtime to docker-compose.yml
   ```

2. **Resource Limits**
   ```bash
   # Set memory limits in docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 8G
   ```

3. **Caching**
   ```bash
   # Enable Redis caching
   # Configure model caching in llm_router.py
   ```

---

## 🔒 Security

### Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API (restrict to internal network)
```

### SSL/TLS Setup

```bash
# Generate SSL certificates
sudo certbot --nginx -d your-domain.com

# Or use self-signed certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/emotionalai.key \
  -out nginx/ssl/emotionalai.crt
```

### Access Control

```bash
# Set up authentication
# Configure API keys in .env
# Set up user management
```

---

## 📞 Support

For deployment issues:

1. Check the logs: `docker-compose logs`
2. Verify system requirements
3. Check network connectivity
4. Review configuration files
5. Consult the troubleshooting section above

**Status**: �� Production Ready 