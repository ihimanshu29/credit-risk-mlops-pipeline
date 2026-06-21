# Use a slim Python image for a stable base
FROM python:3.9-slim

# Set the environment variable to stop Python from buffering output
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# CRITICAL: Add the current directory to the PYTHONPATH so Python can find 'mlProject'
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure, including:
# - wsgi.py (your app file)
# - mlProject/ (your code module)
# - templates/ (your HTML templates)
# - artifacts/ (your trained model)
COPY . .

# Expose the port (Render handles port mapping automatically)
# We don't strictly need EXPOSE for Render, but it's good practice.
EXPOSE 10000 

# Command to run your Flask app with Gunicorn (Industry Standard)
# -w 4: 4 worker processes for concurrency
# --bind 0.0.0.0:$PORT: Listens on the port provided by the Render environment variable
# wsgi:app: Runs the 'app' Flask object found inside the 'wsgi.py' file.
# CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:$PORT", "wsgi:app"]
# Use the shell format to ensure $PORT is replaced with its value
CMD gunicorn --workers 4 --bind 0.0.0.0:$PORT wsgi:app