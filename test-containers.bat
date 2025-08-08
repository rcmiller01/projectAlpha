@echo off
REM Container Hardening Test Script for Windows

echo 🐳 ProjectAlpha Container Hardening Test
echo ========================================

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed or not in PATH
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed or not in PATH
    exit /b 1
)

echo ✅ Docker and Docker Compose are available

REM Create necessary directories
echo 📁 Creating required directories...
if not exist "logs" mkdir logs
if not exist "data\redis" mkdir data\redis
if not exist "data\mongodb" mkdir data\mongodb

REM Build and start containers
echo 🔨 Building containers...
docker-compose build

echo 🚀 Starting containers...
docker-compose up -d

REM Wait for services to start
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Test health endpoints
echo 🏥 Testing health endpoints...

REM Test backend health
echo Testing backend health...
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend health check passed
    curl -s http://localhost:8000/health
) else (
    echo ❌ Backend health check failed
)

REM Check container security
echo 🔒 Checking container security...

REM Check if containers are running as non-root
for /f %%i in ('docker-compose exec -T backend id -u 2^>nul') do set backend_user=%%i
if not "%backend_user%"=="0" (
    echo ✅ Backend container running as non-root user ^(UID: %backend_user%^)
) else (
    echo ❌ Backend container running as root
)

REM Check read-only filesystem
docker-compose exec -T backend touch /test_file >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Root filesystem is read-only
) else (
    echo ❌ Root filesystem is writable
)

REM Check if tmp is writable
docker-compose exec -T backend touch /app/tmp/test_file >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Tmp directory is writable
    docker-compose exec -T backend rm -f /app/tmp/test_file >nul 2>&1
) else (
    echo ❌ Tmp directory is not writable
)

echo 🎉 Container hardening test complete!
echo Use 'docker-compose logs' to view container logs
echo Use 'docker-compose down' to stop containers

pause
