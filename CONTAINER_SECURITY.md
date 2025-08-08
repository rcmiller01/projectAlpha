# 🐳 ProjectAlpha Container Security

This directory contains hardened container configurations for ProjectAlpha with enterprise-grade security features.

## 🔒 Security Features

### Container Hardening

- **Non-root user**: All containers run as unprivileged users
- **Read-only filesystem**: Root filesystem is read-only with specific writable mounts
- **Dropped capabilities**: All Linux capabilities dropped, only essential ones added
- **No new privileges**: Prevents privilege escalation
- **Tmpfs mounts**: Temporary filesystems for ephemeral data

### Health Monitoring

- **Health checks**: All services have health check endpoints
- **Dependency monitoring**: Backend health check verifies Mirror, Anchor, and HRM systems
- **Safe mode awareness**: Health endpoint reports safe mode status

## 📁 Files

- `Dockerfile` - Hardened multi-stage build with non-root user
- `docker-compose.yml` - Base service definitions
- `docker-compose.override.yml` - Security hardening overrides
- `.dockerignore` - Optimized build context
- `test-containers.sh` - Linux/macOS security validation script
- `test-containers.bat` - Windows security validation script

## 🚀 Quick Start

1. **Build and start containers:**

   ```bash
   docker-compose up -d
   ```

2. **Check health status:**

   ```bash
   curl http://localhost:8000/health
   ```

3. **Run security tests:**

   ```bash
   # Linux/macOS
   ./test-containers.sh

   # Windows
   test-containers.bat
   ```

## 🏥 Health Check Response

The `/health` endpoint returns detailed status information:

```json
{
  "status": "up",
  "timestamp": "2025-08-07T12:00:00Z",
  "safe_mode": false,
  "deps": {
    "mirror": true,
    "anchor": true,
    "hrm": true
  },
  "config": {
    "port": 8000,
    "server_role": "primary",
    "server_id": "backend-01"
  }
}
```

### Status Values

- `up` - All dependencies available
- `degraded` - Some dependencies unavailable but core functionality works
- `down` - Critical dependencies unavailable

## 🔧 Configuration

### Environment Variables

- `SAFE_MODE_FORCE` - Force safe mode operation
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `REDIS_URL` - Redis connection string
- `MONGODB_URL` - MongoDB connection string

### Volume Mounts

- `./logs:/app/logs:rw` - Application logs (read-write)
- `./data:/app/data:rw` - Persistent data (read-write)
- `./config:/app/config:ro` - Configuration files (read-only)

### Security Settings

- User: `1000:1000` (non-root)
- Capabilities: Minimal required set only
- Read-only root filesystem
- Tmpfs for temporary files
- No new privileges allowed

## 🔍 Security Validation

The test scripts validate:

- ✅ Non-root execution
- ✅ Read-only root filesystem
- ✅ Writable temporary directories
- ✅ Health check functionality
- ✅ Dependency availability

## 🚨 Troubleshooting

### Permission Issues

If you encounter permission issues:

```bash
# Fix directory permissions
sudo chown -R 1000:1000 logs data
```

### Health Check Failures

Check dependency availability:

```bash
docker-compose logs backend
```

### Container Won't Start

Verify resource allocation:

```bash
docker system df
docker system prune  # if needed
```

## 🔗 Integration

This hardened setup integrates with:

- **Mirror Mode**: Self-reflection and transparency system
- **Anchor System**: Safety and approval mechanism
- **HRM Router**: Hierarchical reasoning model routing

All systems are monitored via the health endpoint for operational awareness.

## 📊 Monitoring

Monitor container health:

```bash
# Check all service health
docker-compose ps

# View real-time logs
docker-compose logs -f

# Check resource usage
docker stats
```

## 🛡️ Security Best Practices

1. **Regular Updates**: Keep base images updated
2. **Secret Management**: Use Docker secrets for sensitive data
3. **Network Isolation**: Use custom networks for service communication
4. **Resource Limits**: Set memory and CPU limits in production
5. **Log Monitoring**: Monitor logs for security events
6. **Backup Strategy**: Regular backups of persistent data

---

_This container setup follows OWASP container security guidelines and industry best practices for production deployments._
