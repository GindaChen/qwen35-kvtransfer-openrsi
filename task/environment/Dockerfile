# Preparation recipe: not yet built/validated by RSI-Harness. GB200 host is Linux arm64.
# The Judge additionally requires the Harness rsi-isolate mount at /opt/rsi-isolate.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv build-essential ninja-build git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY benchmark.py isolated_benchmark.py isolated_worker.py gpu_isolate.py /opt/kv-transfer/
COPY isolation /opt/kv-transfer/isolation
COPY LAYOUT.md /opt/kv-transfer/LAYOUT.md
COPY reference /opt/kv-transfer/reference
ENV PYTHONPATH=/opt/kv-transfer CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 NVCC_THREADS=1 OMP_NUM_THREADS=8
RUN mkdir -p /workspace && cp /opt/kv-transfer/reference/candidate.py /workspace/candidate.py
WORKDIR /workspace
