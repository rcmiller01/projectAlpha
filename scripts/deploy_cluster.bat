@echo off
REM EmotionalAI UCS M3 Cluster Deployment Script for Windows
REM This script deploys the EmotionalAI system across your UCS M3 servers

setlocal enabledelayedexpansion

REM Colors for output (Windows doesn't support ANSI colors in batch)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Configuration
set "CLUSTER_NAME=emotional_ai_cluster"
set "NETWORK_SUBNET=172.20.0.0/16"
set "MONGODB_PASSWORD=emotional_ai_password"
set "GRAFANA_PASSWORD=emotional_ai_admin"

REM Server configurations (adjust IPs to match your network)
set "M3_PRIMARY_IP=192.168.1.10"
set "M3_GPU_IP=192.168.1.11"

echo %BLUE%🚀 EmotionalAI UCS M3 Cluster Deployment%NC%
echo ==================================================

REM Function to print status
:print_status
echo %GREEN%✅ %~1%NC%
goto :eof

REM Function to print warning
:print_warning
echo %YELLOW%⚠️  %~1%NC%
goto :eof

REM Function to print error
:print_error
echo %RED%❌ %~1%NC%
goto :eof

REM Check prerequisites
call :check_prerequisites
if errorlevel 1 (
    call :print_error "Prerequisites check failed"
    exit /b 1
)

REM Create directories
call :create_directories

REM Create configuration files
call :create_mongodb_init
call :create_prometheus_config
call :create_grafana_datasource
call :create_env_file

REM Deploy cluster
call :deploy_cluster

REM Wait for services
call :wait_for_services
if errorlevel 1 (
    call :print_error "Deployment failed. Check logs with: docker-compose -f docker-compose.cluster.yml logs"
    exit /b 1
)

REM Display cluster information
call :display_cluster_info

goto :eof

:check_prerequisites
call :print_status "Checking prerequisites..."

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not installed. Please install Docker Desktop first."
    exit /b 1
)

REM Check Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit /b 1
)

REM Check if ports are available (basic check)
netstat -an | findstr ":80 " >nul
if not errorlevel 1 (
    call :print_warning "Port 80 is already in use. Make sure it's not needed by another service."
)

netstat -an | findstr ":8000 " >nul
if not errorlevel 1 (
    call :print_warning "Port 8000 is already in use. Make sure it's not needed by another service."
)

call :print_status "Prerequisites check completed"
goto :eof

:create_directories
call :print_status "Creating necessary directories..."

if not exist "config\mongodb" mkdir "config\mongodb"
if not exist "config\nginx\ssl" mkdir "config\nginx\ssl"
if not exist "config\prometheus" mkdir "config\prometheus"
if not exist "config\grafana\dashboards" mkdir "config\grafana\dashboards"
if not exist "config\grafana\datasources" mkdir "config\grafana\datasources"
if not exist "logs" mkdir "logs"
if not exist "data\mongodb" mkdir "data\mongodb"
if not exist "data\redis" mkdir "data\redis"
if not exist "data\prometheus" mkdir "data\prometheus"
if not exist "data\grafana" mkdir "data\grafana"

call :print_status "Directories created"
goto :eof

:create_mongodb_init
call :print_status "Creating MongoDB initialization script..."

(
echo // MongoDB initialization script for EmotionalAI
echo db = db.getSiblingDB^('emotional_ai'^);
echo.
echo // Create collections with proper indexes
echo db.createCollection^('memories'^);
echo db.createCollection^('threads'^);
echo db.createCollection^('relationships'^);
echo db.createCollection^('biometrics'^);
echo db.createCollection^('avatars'^);
echo db.createCollection^('sessions'^);
echo db.createCollection^('analytics'^);
echo.
echo // Create indexes for better performance
echo db.memories.createIndex^({"user_id": 1}^);
echo db.memories.createIndex^({"tags": 1}^);
echo db.memories.createIndex^({"memory_type": 1}^);
echo db.memories.createIndex^({"created_at": -1}^);
echo db.memories.createIndex^({"emotional_tags": 1}^);
echo.
echo db.threads.createIndex^({"user_id": 1}^);
echo db.threads.createIndex^({"thread_id": 1}^);
echo db.threads.createIndex^({"created_at": -1}^);
echo db.threads.createIndex^({"tags": 1}^);
echo.
echo db.relationships.createIndex^({"user_id": 1}^);
echo db.relationships.createIndex^({"persona": 1}^);
echo db.relationships.createIndex^({"created_at": -1}^);
echo.
echo db.biometrics.createIndex^({"user_id": 1}^);
echo db.biometrics.createIndex^({"timestamp": -1}^);
echo.
echo db.sessions.createIndex^({"user_id": 1}^);
echo db.sessions.createIndex^({"session_id": 1}^);
echo db.sessions.createIndex^({"created_at": -1}^);
echo.
echo print^("MongoDB initialization completed"^);
) > "config\mongodb\init.js"

call :print_status "MongoDB initialization script created"
goto :eof

:create_prometheus_config
call :print_status "Creating Prometheus configuration..."

(
echo global:
echo   scrape_interval: 15s
echo   evaluation_interval: 15s
echo.
echo rule_files:
echo   # - "first_rules.yml"
echo   # - "second_rules.yml"
echo.
echo scrape_configs:
echo   - job_name: 'emotional_ai_primary'
echo     static_configs:
echo       - targets: ['m3-primary:8000']
echo     metrics_path: '/metrics'
echo     scrape_interval: 30s
echo.
echo   - job_name: 'emotional_ai_gpu'
echo     static_configs:
echo       - targets: ['m3-gpu:8001']
echo     metrics_path: '/metrics'
echo     scrape_interval: 30s
echo.
echo   - job_name: 'mongodb'
echo     static_configs:
echo       - targets: ['mongodb:27017']
echo     scrape_interval: 30s
echo.
echo   - job_name: 'redis'
echo     static_configs:
echo       - targets: ['redis:6379']
echo     scrape_interval: 30s
echo.
echo   - job_name: 'nginx'
echo     static_configs:
echo       - targets: ['nginx:80']
echo     metrics_path: '/nginx_status'
echo     scrape_interval: 30s
) > "config\prometheus\prometheus.yml"

call :print_status "Prometheus configuration created"
goto :eof

:create_grafana_datasource
call :print_status "Creating Grafana datasource configuration..."

(
echo apiVersion: 1
echo.
echo datasources:
echo   - name: Prometheus
echo     type: prometheus
echo     access: proxy
echo     url: http://prometheus:9090
echo     isDefault: true
echo     editable: true
) > "config\grafana\datasources\prometheus.yml"

call :print_status "Grafana datasource configuration created"
goto :eof

:create_env_file
call :print_status "Creating environment configuration..."

(
echo # EmotionalAI Cluster Configuration
echo CLUSTER_NAME=%CLUSTER_NAME%
echo NETWORK_SUBNET=%NETWORK_SUBNET%
echo.
echo # MongoDB Configuration
echo MONGODB_ROOT_USERNAME=admin
echo MONGODB_ROOT_PASSWORD=%MONGODB_PASSWORD%
echo MONGODB_DATABASE=emotional_ai
echo.
echo # Server Configuration
echo M3_PRIMARY_IP=%M3_PRIMARY_IP%
echo M3_GPU_IP=%M3_GPU_IP%
echo M3_PRIMARY_PORT=8000
echo M3_GPU_PORT=8001
echo.
echo # Redis Configuration
echo REDIS_PASSWORD=emotional_ai_redis_password
echo.
echo # Grafana Configuration
echo GRAFANA_ADMIN_PASSWORD=%GRAFANA_PASSWORD%
echo.
echo # Security ^(for local deployment, no authentication needed^)
echo AUTH_ENABLED=false
echo SSL_ENABLED=false
echo.
echo # GPU Configuration
echo GPU_ENABLED=true
echo NVIDIA_VISIBLE_DEVICES=all
) > ".env"

call :print_status "Environment configuration created"
goto :eof

:deploy_cluster
call :print_status "Deploying EmotionalAI cluster..."

REM Stop any existing containers
docker-compose -f docker-compose.cluster.yml down --remove-orphans

REM Build and start services
docker-compose -f docker-compose.cluster.yml up -d --build

call :print_status "Cluster deployment initiated"
goto :eof

:wait_for_services
call :print_status "Waiting for services to be ready..."

set "max_attempts=30"
set "attempt=1"

:wait_loop
echo Checking services ^(attempt %attempt%/%max_attempt%^)...

REM Check MongoDB
docker-compose -f docker-compose.cluster.yml exec -T mongodb mongosh --eval "db.adminCommand('ping')" >nul 2>&1
if errorlevel 1 (
    echo MongoDB ⚠
) else (
    echo MongoDB ✓
)

REM Check Primary Server
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo Primary Server ⚠
) else (
    echo Primary Server ✓
)

REM Check GPU Server
curl -f http://localhost:8001/health >nul 2>&1
if errorlevel 1 (
    echo GPU Server ⚠
) else (
    echo GPU Server ✓
)

REM Check Nginx
curl -f http://localhost/health >nul 2>&1
if errorlevel 1 (
    echo Nginx ⚠
    if %attempt% equ %max_attempts% (
        call :print_error "Services failed to start within the expected time"
        exit /b 1
    )
    timeout /t 10 /nobreak >nul
    set /a attempt+=1
    goto :wait_loop
) else (
    echo Nginx ✓
    call :print_status "All services are ready!"
    exit /b 0
)

goto :eof

:display_cluster_info
call :print_status "Cluster deployment completed successfully!"
echo.
echo %BLUE%🌐 Cluster Access Information:%NC%
echo ==================================
echo Frontend Application: %GREEN%http://localhost%NC%
echo Primary Server API:   %GREEN%http://localhost:8000%NC%
echo GPU Server API:       %GREEN%http://localhost:8001%NC%
echo MongoDB:              %GREEN%mongodb://localhost:27017%NC%
echo Redis:                %GREEN%redis://localhost:6379%NC%
echo.
echo %BLUE%📊 Monitoring:%NC%
echo ================
echo Prometheus:           %GREEN%http://localhost:9090%NC%
echo Grafana:              %GREEN%http://localhost:3000%NC%
echo Grafana Admin:        %GREEN%admin / %GRAFANA_PASSWORD%%NC%
echo.
echo %BLUE%🔧 Management Commands:%NC%
echo ========================
echo View logs:            %YELLOW%docker-compose -f docker-compose.cluster.yml logs -f%NC%
echo Stop cluster:         %YELLOW%docker-compose -f docker-compose.cluster.yml down%NC%
echo Restart cluster:      %YELLOW%docker-compose -f docker-compose.cluster.yml restart%NC%
echo Scale services:       %YELLOW%docker-compose -f docker-compose.cluster.yml up -d --scale m3-primary=2%NC%
echo.
echo %BLUE%📝 Next Steps:%NC%
echo =============
echo 1. Access the application at http://localhost
echo 2. Configure Grafana dashboards for monitoring
echo 3. Set up SSL certificates if needed
echo 4. Configure backup strategies for MongoDB
echo 5. Monitor cluster health and performance
echo.
goto :eof 