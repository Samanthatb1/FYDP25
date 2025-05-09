# TODO changes when we swap from Raspberry Pi 3 to 4
# Both Pi 3 and Pi 4 use ARMv7 (32-bit) by default with Raspberry Pi OS.
# If you install a 64-bit OS on the Pi 4, you’ll switch to ARMv8/aarch64 — 
# this affects which .whl files or Docker base images you use.

# For Pi 3 (ARMv7), use arm32v7 Python 3.9 base image.
# For Pi 4 (ARMv8/aarch64), change this to `arm64v8/python:3.9-slim` if using a 64-bit OS.

# when you choose which OS to download onto the card from here choose 64 bit https://www.raspberrypi.com/software/operating-systems/
FROM arm32v7/python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libportaudio2 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .

# Install other Python dependencies (after installing scipy)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
WORKDIR /app
COPY . .

# Command to run your application
CMD ["python", "main.py"]
