# ProjectAlpha Hardened Container
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r projectalpha && useradd -r -g projectalpha projectalpha

# Set up directory structure with proper permissions
WORKDIR /app
RUN mkdir -p /app/logs /app/data /app/tmp \
    && chown -R projectalpha:projectalpha /app \
    && chmod -R u+rwX,go-rwx /app

# Declare writable mount points (root FS can be read-only at runtime)
VOLUME ["/app/logs", "/app/data", "/app/tmp"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY --chown=projectalpha:projectalpha requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=projectalpha:projectalpha . .

# Set proper permissions for executable files
RUN chmod +x /app/backend/app.py

# Switch to non-root user
USER projectalpha

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "backend/app.py"]
