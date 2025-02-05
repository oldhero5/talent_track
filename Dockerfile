FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements.lock ./

# Install Python dependencies
# Install numpy first to ensure correct version
RUN pip install --no-cache-dir numpy==1.24.3
# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pymongo==4.6.0

# Copy application code
COPY talent_track/ ./talent_track/

# Set environment variables
ENV FLASK_APP=talent_track.app
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]