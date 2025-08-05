#!/bin/bash

# EmotionalAI Startup Script

echo "🚀 Starting EmotionalAI System..."

# Wait for dependencies
echo "⏳ Waiting for MongoDB..."
until nc -z mongodb 27017; do
    sleep 1
done
echo "✅ MongoDB is ready"

echo "⏳ Waiting for Redis..."
until nc -z redis 6379; do
    sleep 1
done
echo "✅ Redis is ready"

echo "⏳ Waiting for LLM Router..."
until nc -z llm-router 8002; do
    sleep 1
done
echo "✅ LLM Router is ready"

# Initialize database
echo "🗄️ Initializing database..."
python -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from backend.database.connection import init_database

async def init():
    await init_database()
    print('Database initialized successfully')

asyncio.run(init())
"

# Start the application
echo "🎯 Starting FastAPI application..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload 