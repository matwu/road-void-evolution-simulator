FROM python:3.10

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libhdf5-dev \
    libgomp1 \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Configure pip (timeout and retries)
RUN pip config set global.timeout 600 && \
    pip config set global.retries 5

# Install Python dependencies from requirements.txt
# Includes dependencies needed for gprMax build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GPU support (optional - when CUDA is available)
# pycuda requires CUDA Toolkit, so install conditionally
# Note: NVIDIA GPU is typically not available on ARM64 architecture
# Try individual installation even though environment markers are in requirements.txt
RUN pip install --no-cache-dir pycuda || echo "Warning: pycuda installation failed (GPU may not be available)"

# Clone gprMax
RUN git clone https://github.com/gprMax/gprMax.git /tmp/gprMax

# Build gprMax
RUN cd /tmp/gprMax && python setup.py build

# Install gprMax
RUN cd /tmp/gprMax && python setup.py install

# Copy gprMax patch script
COPY patch_gprmax.py /tmp/patch_gprmax.py

# Apply gprMax patch (fix CPU socket detection error)
RUN python /tmp/patch_gprmax.py && rm /tmp/patch_gprmax.py

# Cleanup
RUN rm -rf /tmp/gprMax

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
