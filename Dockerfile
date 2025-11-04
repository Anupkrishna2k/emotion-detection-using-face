# Use a lightweight official Python image
FROM python:3.9-slim

# Prevent Python from writing pyc files and buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (for OpenCV and dnn)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# If you don’t have one, create it — example below 👇
# Flask==3.0.0
# opencv-python-headless==4.9.0.80
# onnxruntime==1.18.0
# numpy==1.26.4

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose the port for Flask
EXPOSE 5000

# Default command to start Flask app
CMD ["python", "app.py"]
