# 1. Use Python 3.10 slim to perfectly match your pipeline runtime environment
FROM python:3.10-slim

# Set the environment variable to stop Python from buffering output
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# 2. Fix Docker Warning: Define PYTHONPATH directly without referencing its uninitialized self
ENV PYTHONPATH=/app/src

# 3. Install essential Linux libraries required to compile/run heavy ML libraries like XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and upgrade package managers
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 10000 

# Command to run your Flask app with Gunicorn (Shell form ensures $PORT replacement)
CMD gunicorn --workers 4 --bind 0.0.0.0:$PORT wsgi:app