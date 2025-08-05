# 🚀 Production Deployment Guide
*Complete Production Deployment for Emotionally Intelligent AI Companion System*

## 📋 **Pre-Deployment Checklist**

### **✅ System Requirements**
- [ ] **Server Specifications**: Minimum 8GB RAM, 50GB disk space, 4 CPU cores
- [ ] **Operating System**: Ubuntu 20.04+ or CentOS 8+ (Linux recommended)
- [ ] **Docker**: Version 20.10+
- [ ] **Docker Compose**: Version 2.0+
- [ ] **Python**: Version 3.8+ (for deployment scripts)
- [ ] **Domain & SSL**: Valid domain name and SSL certificate
- [ ] **External APIs**: API keys for OpenAI, Anthropic, ElevenLabs, Twilio

### **✅ Infrastructure Preparation**
- [ ] **Firewall Configuration**: Ports 80, 443 open for HTTP/HTTPS
- [ ] **DNS Configuration**: Domain pointing to server IP
- [ ] **SSL Certificate**: Valid certificate installed
- [ ] **Backup Storage**: S3 bucket or backup storage configured
- [ ] **Monitoring**: Prometheus/Grafana endpoints accessible

---

## 🛠️ **Deployment Process**

### **Step 1: System Preparation**
```bash
# 1. Clone the repository
git clone https://github.com/your-org/ai-companion-system.git
cd ai-companion-system

# 2. Run system requirements check
python deployment/production_deployment.py check

# 3. Install required packages
pip install -r requirements.production.txt
```

### **Step 2: Configuration Setup**
```bash
# 1. Generate production configuration
python deployment/production_deployment.py config

# 2. Configure API keys
cp .env.secrets.template .env.secrets
# Edit .env.secrets with your actual API keys:
# - OPENAI_API_KEY=your_openai_key
# - ANTHROPIC_API_KEY=your_anthropic_key
# - ELEVENLABS_API_KEY=your_elevenlabs_key
# - TWILIO_ACCOUNT_SID=your_twilio_sid
# - TWILIO_AUTH_TOKEN=your_twilio_token

# 3. Setup SSL certificates
./security/setup_ssl.sh your-domain.com
```

### **Step 3: Security Configuration**
```bash
# 1. Configure firewall
sudo ./security/setup_firewall.sh

# 2. Set file permissions
chmod 600 .env.secrets
chmod 755 scripts/*.sh
```

### **Step 4: Production Deployment**
```bash
# 1. Run full deployment preparation
python deployment/production_deployment.py deploy

# 2. Deploy the application
./scripts/deploy.sh

# 3. Verify deployment
./scripts/status.sh
```

---

## 🔧 **Configuration Details**

### **Environment Variables (.env.production)**
```bash
# Core Configuration
NODE_ENV=production
PYTHON_ENV=production
DEBUG=false

# Database
MONGODB_URI=mongodb://mongodb:27017/ai_companion_prod
REDIS_URL=redis://redis:6379/0

# API Configuration
API_BASE_URL=https://your-domain.com/api
FRONTEND_URL=https://your-domain.com

# Security (auto-generated)
JWT_SECRET=[auto-generated-64-char-key]
ENCRYPTION_KEY=[auto-generated-64-char-key]
SESSION_SECRET=[auto-generated-64-char-key]

# Performance
MAX_WORKERS=4
```

### **Required Secrets (.env.secrets)**
```bash
# External API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Communication Services
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token

# Database Security
MONGODB_ROOT_PASSWORD=your_secure_mongodb_password

# Monitoring
GRAFANA_ADMIN_PASSWORD=your_grafana_admin_password

# Backup Storage (optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET=your-backup-bucket-name
```

### **Service Configuration**
```yaml
# Services Overview
frontend:
  - Port: 3000 (internal)
  - Technology: SvelteKit + Nginx
  - Features: Responsive UI, PWA support

backend:
  - Port: 8000 (internal)
  - Technology: FastAPI + Gunicorn
  - Workers: 4 (configurable)
  - Features: 140+ API endpoints

database:
  - MongoDB: Port 27017
  - Redis: Port 6379
  - Persistence: Docker volumes

proxy:
  - Nginx: Ports 80/443
  - SSL termination
  - Load balancing
  - Rate limiting
```

---

## 📊 **Health Monitoring**

### **Health Check Endpoints**
```bash
# System Health
curl https://your-domain.com/health
# Expected: 200 OK "healthy"

# Backend API Health
curl https://your-domain.com/api/health
# Expected: 200 OK with system status JSON

# Service Status
./scripts/status.sh
# Shows all service statuses and resource usage
```

### **Monitoring Dashboards**
- **Prometheus**: `https://your-domain.com:9090`
- **Grafana**: `https://your-domain.com:3001`
- **Application Logs**: `docker-compose logs -f backend`

### **Key Metrics to Monitor**
- **Response Time**: API endpoint performance
- **Memory Usage**: Backend service memory consumption
- **Database Connections**: MongoDB connection pool
- **Creative Model Usage**: AI model invocation frequency
- **User Sessions**: Active user connections
- **Error Rate**: 4xx/5xx HTTP response rates

---

## 🔄 **Maintenance Operations**

### **Regular Updates**
```bash
# Update system (includes backup)
./scripts/update.sh

# Manual backup
./scripts/backup.sh

# View recent logs
docker-compose -f docker-compose.production.yml logs --tail=100 backend
```

### **Scaling Operations**
```bash
# Scale backend workers
docker-compose -f docker-compose.production.yml up -d --scale backend=5

# Scale frontend instances
docker-compose -f docker-compose.production.yml up -d --scale frontend=3
```

### **Troubleshooting**
```bash
# Check service status
docker-compose -f docker-compose.production.yml ps

# Restart specific service
docker-compose -f docker-compose.production.yml restart backend

# View detailed logs
docker-compose -f docker-compose.production.yml logs backend

# Database maintenance
docker exec -it ai_companion_mongodb mongo
```

---

## 🔒 **Security Considerations**

### **Production Security Features**
- **SSL/TLS**: Full HTTPS encryption with modern ciphers
- **Rate Limiting**: API and upload rate limits
- **Firewall**: UFW configured for minimal attack surface
- **Container Security**: Non-root container users
- **Data Encryption**: Database and backup encryption
- **Secret Management**: Separate secrets file with restricted permissions

### **Security Best Practices**
1. **Regular Updates**: Keep all components updated
2. **Access Control**: Use SSH keys, disable password auth
3. **Monitoring**: Set up intrusion detection
4. **Backups**: Automated daily backups with encryption
5. **Logs**: Centralized logging with rotation
6. **Network**: VPC/private networks in cloud environments

### **Compliance Considerations**
- **GDPR**: User data encryption and deletion capabilities
- **CCPA**: Data export and privacy controls
- **SOC 2**: Security monitoring and access controls
- **HIPAA**: Healthcare data encryption (if applicable)

---

## 📈 **Performance Optimization**

### **Production Performance Features**
- **Load Balancing**: Nginx upstream configuration
- **Caching**: Redis for session and content caching
- **CDN Ready**: Static asset optimization
- **Database Indexing**: Optimized MongoDB indexes
- **Connection Pooling**: Efficient database connections

### **Recommended Performance Tuning**
```bash
# MongoDB optimization
docker exec ai_companion_mongodb mongo --eval "db.runCommand({profile: 2})"

# Redis memory optimization
docker exec ai_companion_redis redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Nginx worker optimization (adjust based on CPU cores)
# Edit nginx/nginx.conf: worker_processes auto;
```

---

## 🆘 **Emergency Procedures**

### **Service Recovery**
```bash
# Emergency restart all services
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d

# Restore from backup
./scripts/restore.sh backup_filename.tar.gz

# Emergency maintenance mode
# Edit nginx.conf to return 503 for all requests
```

### **Data Recovery**
```bash
# MongoDB data recovery
docker exec ai_companion_mongodb mongorestore /backup/mongodb/

# Application data recovery
cp -r /backup/data/* ./data/
```

---

## 📞 **Support & Documentation**

### **System Architecture Documentation**
- **API Documentation**: Available at `/api/docs` (Swagger UI)
- **Component Integration**: See `COMPONENT_INTEGRATION_PLAN.md`
- **Database Schema**: See `docs/database_schema.md`
- **Security Architecture**: See `docs/security_architecture.md`

### **Operational Runbooks**
- **Incident Response**: `docs/incident_response.md`
- **Backup/Restore**: `docs/backup_procedures.md`
- **Scaling Guide**: `docs/scaling_guide.md`
- **API Key Rotation**: `docs/api_key_rotation.md`

### **Technical Support**
- **Production Issues**: Follow incident response procedures
- **Performance Issues**: Check monitoring dashboards first
- **Security Issues**: Immediate escalation required
- **Feature Requests**: Development team via standard channels

---

## ✅ **Deployment Verification**

### **Post-Deployment Checklist**
- [ ] **All services running**: Check `./scripts/status.sh`
- [ ] **Health endpoints responding**: Test `/health` and `/api/health`
- [ ] **SSL certificate valid**: Verify HTTPS connection
- [ ] **API functionality**: Test core endpoints
- [ ] **Database connectivity**: Verify MongoDB/Redis connections
- [ ] **External integrations**: Test OpenAI, Anthropic, Twilio APIs
- [ ] **Creative features**: Test AI model installations
- [ ] **Monitoring active**: Verify Prometheus/Grafana
- [ ] **Backups configured**: Test backup script execution
- [ ] **Documentation updated**: Verify all docs are current

### **Success Criteria**
✅ **System Status**: All services green  
✅ **Response Time**: < 200ms for API endpoints  
✅ **Uptime**: 99.9% availability target  
✅ **Security**: All security scans pass  
✅ **Performance**: Load testing successful  
✅ **Monitoring**: All metrics collecting  
✅ **Backup**: Automated backups working  

**🎉 Deployment Complete!**

Your emotionally intelligent AI companion system is now running in production with:
- 6 major enhancement modules (100% complete)
- 140+ API endpoints
- Creative discovery and dynamic AI model integration
- Commercial-grade security and monitoring
- Automated backup and recovery
- Scalable architecture ready for growth

**Access your system**: `https://your-domain.com`  
**Admin dashboard**: `https://your-domain.com:3001` (Grafana)  
**API documentation**: `https://your-domain.com/api/docs`

## 🧠 **Advanced Modules Configuration**

### **Psychological Intelligence Services**
```bash
# Environment variables for advanced modules
ATTACHMENT_ENGINE_ENABLED=true
SHADOW_MEMORY_ENABLED=true
DREAM_GENERATION_ENABLED=true
MOODSCAPE_AUDIO_ENABLED=true

# Audio file storage
AUDIO_STORAGE_PATH=/app/data/audio
AUDIO_CDN_URL=https://cdn.yourapp.com/audio
```
