# using latest as it is the only one with 3.12 support
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest AS builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install pip and venv
RUN apt-get update && apt-get install -y python3-pip python3-venv python3-gdal \
 && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv --system-site-packages /venv
ENV PATH="/venv/bin:/app/src:$PATH"

# Upgrade pip inside the venv and install requirements
COPY requirements.txt .
RUN /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .

# Make all Python scripts executable
RUN find . -name "*.py" -exec chmod +x {} \;

# Entry point executes whatever command is passed
# ENTRYPOINT ["/bin/bash"]
