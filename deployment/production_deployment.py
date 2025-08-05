#!/usr/bin/env python3
"""
Production Deployment Script for Emotionally Intelligent AI Companion System
Handles comprehensive deployment preparation, configuration, and launch
"""

import os
import sys
import json
import subprocess
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import docker
import psutil
import requests
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Comprehensive production deployment orchestrator"""
    
    def __init__(self, config_path: str = "deployment/production_config.yml"):
        self.config_path = config_path
        self.config = self.load_deployment_config()
        self.docker_client = docker.from_env()
        self.deployment_root = Path(__file__).parent.parent
        self.backup_dir = self.deployment_root / "backups"
        
    def load_deployment_config(self) -> Dict[str, Any]:
        """Load production deployment configuration"""
        try:
            config_file = self.deployment_root / self.config_path
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self.create_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.create_default_config()
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default production configuration"""
        return {
            "deployment": {
                "environment": "production",
                "domain": "ai-companion.example.com",
                "ssl_enabled": True,
                "backup_enabled": True,
                "monitoring_enabled": True
            },
            "services": {
                "frontend": {
                    "port": 3000,
                    "replicas": 2,
                    "health_check": "/health"
                },
                "backend": {
                    "port": 8000,
                    "replicas": 3,
                    "health_check": "/api/health"
                },
                "mongodb": {
                    "port": 27017,
                    "replicas": 1,
                    "data_volume": "/data/mongodb"
                },
                "nginx": {
                    "port": 80,
                    "ssl_port": 443,
                    "replicas": 1
                }
            },
            "security": {
                "ssl_certificate_path": "/etc/ssl/certs/ai-companion.crt",
                "ssl_key_path": "/etc/ssl/private/ai-companion.key",
                "firewall_enabled": True,
                "rate_limiting": True,
                "cors_origins": ["https://ai-companion.example.com"]
            },
            "monitoring": {
                "prometheus": True,
                "grafana": True,
                "log_aggregation": True,
                "alerts": True
            },
            "backup": {
                "schedule": "0 2 * * *",  # Daily at 2 AM
                "retention_days": 30,
                "s3_bucket": "ai-companion-backups",
                "encryption": True
            }
        }
    
    def check_system_requirements(self) -> bool:
        """Verify system meets production requirements"""
        logger.info("🔍 Checking system requirements...")
        
        requirements_met = True
        
        # Check available memory (minimum 8GB)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            logger.error(f"❌ Insufficient memory: {memory_gb:.1f}GB (minimum 8GB required)")
            requirements_met = False
        else:
            logger.info(f"✅ Memory: {memory_gb:.1f}GB")
        
        # Check available disk space (minimum 50GB)
        disk_usage = psutil.disk_usage('/')
        disk_gb = disk_usage.free / (1024**3)
        if disk_gb < 50:
            logger.error(f"❌ Insufficient disk space: {disk_gb:.1f}GB (minimum 50GB required)")
            requirements_met = False
        else:
            logger.info(f"✅ Disk space: {disk_gb:.1f}GB available")
        
        # Check Docker availability
        try:
            docker_version = self.docker_client.version()
            logger.info(f"✅ Docker: {docker_version['Version']}")
        except Exception as e:
            logger.error(f"❌ Docker not available: {e}")
            requirements_met = False
        
        # Check Python version (minimum 3.8)
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"❌ Python version too old: {python_version} (minimum 3.8 required)")
            requirements_met = False
        else:
            logger.info(f"✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        return requirements_met
    
    def create_production_env_file(self):
        """Generate production environment variables"""
        logger.info("📝 Creating production environment file...")
        
        env_vars = {
            # Core Configuration
            "NODE_ENV": "production",
            "PYTHON_ENV": "production",
            "DEBUG": "false",
            
            # Database
            "MONGODB_URI": "mongodb://mongodb:27017/ai_companion_prod",
            "MONGODB_DATABASE": "ai_companion_prod",
            
            # API Configuration
            "API_BASE_URL": f"https://{self.config['deployment']['domain']}/api",
            "FRONTEND_URL": f"https://{self.config['deployment']['domain']}",
            
            # Security
            "JWT_SECRET": self.generate_secure_key(),
            "ENCRYPTION_KEY": self.generate_secure_key(),
            "SESSION_SECRET": self.generate_secure_key(),
            
            # External APIs (placeholder - to be configured)
            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
            "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
            "ELEVENLABS_API_KEY": "${ELEVENLABS_API_KEY}",
            "TWILIO_ACCOUNT_SID": "${TWILIO_ACCOUNT_SID}",
            "TWILIO_AUTH_TOKEN": "${TWILIO_AUTH_TOKEN}",
            
            # Performance
            "MAX_WORKERS": str(psutil.cpu_count()),
            "REDIS_URL": "redis://redis:6379/0",
            
            # Monitoring
            "PROMETHEUS_ENABLED": "true",
            "LOG_LEVEL": "info",
            
            # Features
            "CREATIVE_DISCOVERY_ENABLED": "true",
            "VOICE_SYNTHESIS_ENABLED": "true",
            "SMS_INTEGRATION_ENABLED": "true",
            "BIOMETRIC_INTEGRATION_ENABLED": "true"
        }
        
        env_file_path = self.deployment_root / ".env.production"
        with open(env_file_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"✅ Environment file created: {env_file_path}")
        
        # Create template for secrets
        secrets_template_path = self.deployment_root / ".env.secrets.template"
        with open(secrets_template_path, 'w') as f:
            f.write("# Production Secrets - Replace with actual values\n")
            f.write("OPENAI_API_KEY=your_openai_key_here\n")
            f.write("ANTHROPIC_API_KEY=your_anthropic_key_here\n")
            f.write("ELEVENLABS_API_KEY=your_elevenlabs_key_here\n")
            f.write("TWILIO_ACCOUNT_SID=your_twilio_sid_here\n")
            f.write("TWILIO_AUTH_TOKEN=your_twilio_token_here\n")
        
        logger.info(f"📋 Secrets template created: {secrets_template_path}")
    
    def generate_secure_key(self, length: int = 64) -> str:
        """Generate cryptographically secure random key"""
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def create_docker_compose_production(self):
        """Generate production-ready docker-compose.yml"""
        logger.info("🐳 Creating production Docker Compose configuration...")
        
        compose_config = {
            "version": "3.8",
            "services": {
                "nginx": {
                    "image": "nginx:alpine",
                    "container_name": "ai_companion_nginx",
                    "ports": [
                        f"{self.config['services']['nginx']['port']}:80",
                        f"{self.config['services']['nginx']['ssl_port']}:443"
                    ],
                    "volumes": [
                        "./nginx/nginx.conf:/etc/nginx/nginx.conf:ro",
                        "./nginx/ssl:/etc/ssl:ro",
                        "./frontend/dist:/usr/share/nginx/html:ro"
                    ],
                    "depends_on": ["frontend", "backend"],
                    "restart": "unless-stopped",
                    "networks": ["ai_companion_network"]
                },
                "frontend": {
                    "build": {
                        "context": "./frontend",
                        "dockerfile": "Dockerfile.production"
                    },
                    "container_name": "ai_companion_frontend",
                    "expose": [str(self.config['services']['frontend']['port'])],
                    "environment": [
                        "NODE_ENV=production"
                    ],
                    "restart": "unless-stopped",
                    "networks": ["ai_companion_network"],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", f"http://localhost:{self.config['services']['frontend']['port']}/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                },
                "backend": {
                    "build": {
                        "context": "./backend",
                        "dockerfile": "Dockerfile.production"
                    },
                    "container_name": "ai_companion_backend",
                    "expose": [str(self.config['services']['backend']['port'])],
                    "env_file": [".env.production", ".env.secrets"],
                    "volumes": [
                        "./data/uploads:/app/uploads",
                        "./data/generated_content:/app/generated_content",
                        "./logs:/app/logs"
                    ],
                    "depends_on": ["mongodb", "redis"],
                    "restart": "unless-stopped",
                    "networks": ["ai_companion_network"],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", f"http://localhost:{self.config['services']['backend']['port']}/api/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                },
                "mongodb": {
                    "image": "mongo:6.0",
                    "container_name": "ai_companion_mongodb",
                    "expose": [str(self.config['services']['mongodb']['port'])],
                    "environment": [
                        "MONGO_INITDB_ROOT_USERNAME=admin",
                        "MONGO_INITDB_ROOT_PASSWORD=${MONGODB_ROOT_PASSWORD}",
                        "MONGO_INITDB_DATABASE=ai_companion_prod"
                    ],
                    "volumes": [
                        "./data/mongodb:/data/db",
                        "./mongodb/init:/docker-entrypoint-initdb.d"
                    ],
                    "restart": "unless-stopped",
                    "networks": ["ai_companion_network"],
                    "command": ["mongod", "--auth", "--bind_ip_all"]
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "container_name": "ai_companion_redis",
                    "expose": ["6379"],
                    "volumes": [
                        "./data/redis:/data"
                    ],
                    "restart": "unless-stopped",
                    "networks": ["ai_companion_network"],
                    "command": ["redis-server", "--appendonly", "yes"]
                }
            },
            "networks": {
                "ai_companion_network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "mongodb_data": None,
                "redis_data": None,
                "uploads_data": None
            }
        }
        
        # Add monitoring services if enabled
        if self.config['monitoring']['prometheus']:
            compose_config['services']['prometheus'] = {
                "image": "prom/prometheus:latest",
                "container_name": "ai_companion_prometheus",
                "ports": ["9090:9090"],
                "volumes": [
                    "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro"
                ],
                "restart": "unless-stopped",
                "networks": ["ai_companion_network"]
            }
        
        if self.config['monitoring']['grafana']:
            compose_config['services']['grafana'] = {
                "image": "grafana/grafana:latest",
                "container_name": "ai_companion_grafana",
                "ports": ["3001:3000"],
                "environment": [
                    "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}"
                ],
                "volumes": [
                    "./monitoring/grafana:/var/lib/grafana"
                ],
                "restart": "unless-stopped",
                "networks": ["ai_companion_network"]
            }
        
        # Write production docker-compose file
        compose_file_path = self.deployment_root / "docker-compose.production.yml"
        with open(compose_file_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Production Docker Compose created: {compose_file_path}")
    
    def create_nginx_config(self):
        """Generate production Nginx configuration"""
        logger.info("🌐 Creating Nginx configuration...")
        
        nginx_dir = self.deployment_root / "nginx"
        nginx_dir.mkdir(exist_ok=True)
        
        nginx_config = f"""
# Production Nginx Configuration for AI Companion System
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {{
    worker_connections 1024;
    use epoll;
    multi_accept on;
}}

http {{
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/json
        application/xml+rss;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;

    # Upstream backends
    upstream frontend {{
        server frontend:{self.config['services']['frontend']['port']} max_fails=3 fail_timeout=30s;
    }}
    
    upstream backend {{
        server backend:{self.config['services']['backend']['port']} max_fails=3 fail_timeout=30s;
    }}

    # HTTP to HTTPS redirect
    server {{
        listen 80;
        server_name {self.config['deployment']['domain']};
        return 301 https://$server_name$request_uri;
    }}

    # Main HTTPS server
    server {{
        listen 443 ssl http2;
        server_name {self.config['deployment']['domain']};

        # SSL Configuration
        ssl_certificate {self.config['security']['ssl_certificate_path']};
        ssl_certificate_key {self.config['security']['ssl_key_path']};
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 1d;
        ssl_session_tickets off;

        # Security headers
        add_header Strict-Transport-Security "max-age=63072000" always;
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Referrer-Policy "strict-origin-when-cross-origin";

        # API routes
        location /api/ {{
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }}

        # WebSocket routes
        location /ws/ {{
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # File uploads
        location /api/upload/ {{
            limit_req zone=upload burst=5 nodelay;
            client_max_body_size 100M;
            proxy_pass http://backend;
            proxy_request_buffering off;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # Frontend application
        location / {{
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # Static assets caching
        location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {{
            expires 1y;
            add_header Cache-Control "public, immutable";
        }}

        # Health check endpoint
        location /health {{
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }}
    }}
}}
"""
        
        nginx_config_path = nginx_dir / "nginx.conf"
        with open(nginx_config_path, 'w') as f:
            f.write(nginx_config.strip())
        
        logger.info(f"✅ Nginx configuration created: {nginx_config_path}")
    
    def create_production_dockerfiles(self):
        """Create optimized production Dockerfiles"""
        logger.info("🐳 Creating production Dockerfiles...")
        
        # Frontend production Dockerfile
        frontend_dockerfile = """
# Frontend Production Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
"""
        
        frontend_docker_path = self.deployment_root / "frontend" / "Dockerfile.production"
        frontend_docker_path.parent.mkdir(exist_ok=True)
        with open(frontend_docker_path, 'w') as f:
            f.write(frontend_dockerfile.strip())
        
        # Backend production Dockerfile
        backend_dockerfile = """
# Backend Production Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    ffmpeg \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "main:app"]
"""
        
        backend_docker_path = self.deployment_root / "backend" / "Dockerfile.production"
        backend_docker_path.parent.mkdir(exist_ok=True)
        with open(backend_docker_path, 'w') as f:
            f.write(backend_dockerfile.strip())
        
        logger.info("✅ Production Dockerfiles created")
    
    def setup_monitoring(self):
        """Configure monitoring and observability"""
        logger.info("📊 Setting up monitoring and observability...")
        
        monitoring_dir = self.deployment_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [],
            "scrape_configs": [
                {
                    "job_name": "ai-companion-backend",
                    "static_configs": [
                        {"targets": ["backend:8000"]}
                    ],
                    "metrics_path": "/api/metrics"
                },
                {
                    "job_name": "ai-companion-frontend",
                    "static_configs": [
                        {"targets": ["frontend:3000"]}
                    ]
                },
                {
                    "job_name": "mongodb",
                    "static_configs": [
                        {"targets": ["mongodb:27017"]}
                    ]
                },
                {
                    "job_name": "redis",
                    "static_configs": [
                        {"targets": ["redis:6379"]}
                    ]
                }
            ]
        }
        
        prometheus_config_path = monitoring_dir / "prometheus.yml"
        with open(prometheus_config_path, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        logger.info("✅ Monitoring configuration created")
    
    def create_backup_scripts(self):
        """Create automated backup scripts"""
        logger.info("💾 Creating backup scripts...")
        
        backup_script = f"""#!/bin/bash
# Automated Backup Script for AI Companion System

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ai_companion_backup_$DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup MongoDB
echo "Backing up MongoDB..."
docker exec ai_companion_mongodb mongodump --out "/tmp/mongodb_backup_$DATE"
docker cp ai_companion_mongodb:/tmp/mongodb_backup_$DATE "$BACKUP_DIR/$BACKUP_NAME/mongodb"

# Backup application data
echo "Backing up application data..."
cp -r ./data "$BACKUP_DIR/$BACKUP_NAME/"
cp -r ./config "$BACKUP_DIR/$BACKUP_NAME/"

# Backup logs
echo "Backing up logs..."
cp -r ./logs "$BACKUP_DIR/$BACKUP_NAME/"

# Create compressed archive
echo "Creating compressed archive..."
cd "$BACKUP_DIR"
tar -czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"

# Upload to S3 if configured
if [ ! -z "${{S3_BUCKET}}" ]; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_NAME.tar.gz" "s3://${{S3_BUCKET}}/backups/"
fi

# Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -name "ai_companion_backup_*.tar.gz" -mtime +{self.config['backup']['retention_days']} -delete

echo "Backup completed: $BACKUP_NAME.tar.gz"
"""
        
        backup_script_path = self.deployment_root / "scripts" / "backup.sh"
        backup_script_path.parent.mkdir(exist_ok=True)
        with open(backup_script_path, 'w') as f:
            f.write(backup_script.strip())
        
        backup_script_path.chmod(0o755)
        
        logger.info("✅ Backup scripts created")
    
    def create_deployment_scripts(self):
        """Create deployment and management scripts"""
        logger.info("📜 Creating deployment scripts...")
        
        scripts_dir = self.deployment_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Deploy script
        deploy_script = """#!/bin/bash
# AI Companion Production Deployment Script

set -e

echo "🚀 Starting AI Companion deployment..."

# Check if secrets file exists
if [ ! -f ".env.secrets" ]; then
    echo "❌ Error: .env.secrets file not found!"
    echo "Please copy .env.secrets.template to .env.secrets and fill in your API keys"
    exit 1
fi

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose -f docker-compose.production.yml pull

# Build custom images
echo "🔨 Building application images..."
docker-compose -f docker-compose.production.yml build

# Start services
echo "▶️ Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run health checks
echo "🏥 Running health checks..."
for service in frontend backend mongodb redis; do
    if docker-compose -f docker-compose.production.yml ps $service | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
        exit 1
    fi
done

echo "🎉 Deployment completed successfully!"
echo "🌐 Application available at: https://$(grep FRONTEND_URL .env.production | cut -d'=' -f2)"
"""
        
        deploy_script_path = scripts_dir / "deploy.sh"
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script.strip())
        deploy_script_path.chmod(0o755)
        
        # Update script
        update_script = """#!/bin/bash
# AI Companion System Update Script

set -e

echo "🔄 Updating AI Companion system..."

# Create backup before update
echo "💾 Creating backup..."
./scripts/backup.sh

# Pull latest changes
echo "📥 Pulling latest code..."
git pull origin main

# Rebuild and restart services
echo "🔨 Rebuilding services..."
docker-compose -f docker-compose.production.yml build --no-cache

echo "🔄 Restarting services..."
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d

echo "✅ Update completed successfully!"
"""
        
        update_script_path = scripts_dir / "update.sh"
        with open(update_script_path, 'w') as f:
            f.write(update_script.strip())
        update_script_path.chmod(0o755)
        
        # Status script
        status_script = """#!/bin/bash
# AI Companion System Status Check

echo "📊 AI Companion System Status"
echo "=============================="

# Service status
echo "🐳 Docker Services:"
docker-compose -f docker-compose.production.yml ps

echo ""
echo "💾 System Resources:"
echo "Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% used"

echo ""
echo "🌐 Health Checks:"
curl -s -o /dev/null -w "Frontend: %{http_code}\\n" http://localhost/health
curl -s -o /dev/null -w "Backend: %{http_code}\\n" http://localhost/api/health

echo ""
echo "📈 Recent Logs:"
docker-compose -f docker-compose.production.yml logs --tail=10 backend
"""
        
        status_script_path = scripts_dir / "status.sh"
        with open(status_script_path, 'w') as f:
            f.write(status_script.strip())
        status_script_path.chmod(0o755)
        
        logger.info("✅ Deployment scripts created")
    
    def create_security_configurations(self):
        """Set up security configurations and firewall rules"""
        logger.info("🔒 Setting up security configurations...")
        
        security_dir = self.deployment_root / "security"
        security_dir.mkdir(exist_ok=True)
        
        # Firewall rules script
        firewall_script = """#!/bin/bash
# Production Firewall Configuration

# Reset firewall
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing

# SSH access (adjust port as needed)
ufw allow 22/tcp

# HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Monitoring (restrict to specific IPs)
# ufw allow from YOUR_MONITORING_IP to any port 9090
# ufw allow from YOUR_MONITORING_IP to any port 3001

# Enable firewall
ufw --force enable

echo "🔒 Firewall configured successfully"
"""
        
        firewall_script_path = security_dir / "setup_firewall.sh"
        with open(firewall_script_path, 'w') as f:
            f.write(firewall_script.strip())
        firewall_script_path.chmod(0o755)
        
        # SSL certificate generation script
        ssl_script = """#!/bin/bash
# SSL Certificate Setup Script

DOMAIN="${1:-ai-companion.example.com}"
SSL_DIR="./nginx/ssl"

mkdir -p "$SSL_DIR"

# Generate self-signed certificate for testing
# Replace with Let's Encrypt or proper CA certificate for production
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\
    -keyout "$SSL_DIR/server.key" \\
    -out "$SSL_DIR/server.crt" \\
    -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"

echo "🔐 SSL certificate generated for $DOMAIN"
echo "⚠️  This is a self-signed certificate. Use Let's Encrypt for production."
"""
        
        ssl_script_path = security_dir / "setup_ssl.sh"
        with open(ssl_script_path, 'w') as f:
            f.write(ssl_script.strip())
        ssl_script_path.chmod(0o755)
        
        logger.info("✅ Security configurations created")
    
    def run_full_deployment(self):
        """Execute complete production deployment"""
        logger.info("🚀 Starting full production deployment...")
        
        try:
            # System checks
            if not self.check_system_requirements():
                logger.error("❌ System requirements not met. Aborting deployment.")
                return False
            
            # Create configuration files
            self.create_production_env_file()
            self.create_docker_compose_production()
            self.create_nginx_config()
            self.create_production_dockerfiles()
            
            # Setup monitoring and backup
            self.setup_monitoring()
            self.create_backup_scripts()
            
            # Create deployment scripts
            self.create_deployment_scripts()
            self.create_security_configurations()
            
            # Create data directories
            data_dirs = ["data/mongodb", "data/redis", "data/uploads", "data/generated_content", "logs", "backups"]
            for dir_path in data_dirs:
                (self.deployment_root / dir_path).mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ Production deployment preparation completed successfully!")
            logger.info("📋 Next steps:")
            logger.info("1. Copy .env.secrets.template to .env.secrets and fill in your API keys")
            logger.info("2. Configure SSL certificates: ./security/setup_ssl.sh your-domain.com")
            logger.info("3. Run deployment: ./scripts/deploy.sh")
            logger.info("4. Check status: ./scripts/status.sh")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment preparation failed: {e}")
            return False

def main():
    """Main deployment function"""
    deployer = ProductionDeployer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "check":
            deployer.check_system_requirements()
        elif command == "config":
            deployer.create_production_env_file()
        elif command == "docker":
            deployer.create_docker_compose_production()
        elif command == "deploy":
            deployer.run_full_deployment()
        else:
            print("Usage: python production_deployment.py [check|config|docker|deploy]")
    else:
        deployer.run_full_deployment()

if __name__ == "__main__":
    main()
