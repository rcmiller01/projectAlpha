#!/usr/bin/env python3
"""
Setup script for Dolphin AI House of Minds system

This setup script includes all required dependencies for the complete
Dolphin AI system including MCP integration, Ollama support, and backend services.
"""

from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

setup(
    name="dolphin-ai-house-of-minds",
    version="2.1.0",
    author="Dolphin AI System",
    author_email="ai@dolphinsystem.com",
    description="Advanced AI orchestration system with MCP integration and dynamic agent routing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rcmiller01/test2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core web framework
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        
        # HTTP clients and async support
        'httpx>=0.25.0',
        'aiohttp>=3.9.0',
        'aiofiles>=23.2.0',
        
        # Data validation and serialization
        'pydantic>=2.4.0',
        'pydantic-settings>=2.0.0',
        
        # Ollama integration
        'ollama>=0.1.7',
        
        # MCP (Model Context Protocol) integration
        'mcp>=0.1.0',  # Adjust version as needed
        'websockets>=11.0.0',  # For MCP WebSocket support
        
        # Database and storage
        'sqlalchemy>=2.0.0',
        'alembic>=1.12.0',
        'aiosqlite>=0.19.0',  # Async SQLite support
        
        # AI and ML libraries
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'transformers>=4.30.0',
        'torch>=2.0.0',
        
        # Natural language processing
        'nltk>=3.8.0',
        'spacy>=3.6.0',
        
        # Configuration management
        'python-dotenv>=1.0.0',
        'pyyaml>=6.0.0',
        'toml>=0.10.0',
        
        # Logging and monitoring
        'structlog>=23.2.0',
        'rich>=13.0.0',
        
        # File watching for hot-reload
        'watchdog>=3.0.0',
        
        # Date and time handling
        'python-dateutil>=2.8.0',
        'pytz>=2023.3',
        
        # Concurrency and async utilities
        'asyncio-throttle>=1.0.0',
        'tenacity>=8.2.0',  # For retry logic
        
        # Testing (dev dependencies)
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.1.0',
        'httpx[testing]>=0.25.0',
        
        # Development tools
        'black>=23.0.0',
        'isort>=5.12.0',
        'flake8>=6.0.0',
        'mypy>=1.5.0',
        
        # Security
        'cryptography>=41.0.0',
        'python-jose>=3.3.0',
        'passlib>=1.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
            'pre-commit>=3.4.0',
        ],
        'gpu': [
            'torch[gpu]>=2.0.0',
            'transformers[torch]>=4.30.0',
            'accelerate>=0.20.0',
        ],
        'advanced': [
            'redis>=5.0.0',
            'celery>=5.3.0',
            'flower>=2.0.0',  # For Celery monitoring
            'prometheus-client>=0.17.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'dolphin-ai=house_of_minds.main:main',
            'dolphin-server=house_of_minds.start_dolphin_system:main',
            'dolphin-mcp=core1_gateway.start_mcp_server:main',
        ],
    },
    include_package_data=True,
    package_data={
        'house_of_minds': ['config/*.json', 'config/*.yaml'],
        'core1_gateway': ['agents/*.json'],
    },
    zip_safe=False,
)
