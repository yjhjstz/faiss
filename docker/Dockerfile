# Runtime image for faiss-gpu with SQ4/SQ8 GpuIndexRefine.
#
# Build context: repo root
#   docker build -f docker/Dockerfile -t faiss-gpu-sq:1.8.0 .
#
# Requires:
#   - host: nvidia driver >= 515 (CUDA 11.7 runtime)
#   - runtime: nvidia-container-toolkit (docker run --gpus all ...)

# The bundled faiss wheel is linked against CUDA 11.7 (libcudart.so.11.0,
# libcublas.so.11). Keep this base image in lockstep with the build toolchain.
FROM nvcr.io/nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

# Python 3.8 must match the faiss wheel ABI (the _swigfaiss.so is built
# against CPython 3.8). Install it via deadsnakes PPA and bootstrap pip for
# 3.8 explicitly — do NOT rely on the apt python3-pip, which is tied to
# the image's default Python 3.10.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv python3.8-distutils \
        libgomp1 libopenblas0 \
    && curl -sSL https://bootstrap.pypa.io/pip/3.8/get-pip.py -o /tmp/get-pip.py \
    && python3.8 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python3 \
    && python3.8 -m pip install --no-cache-dir numpy==1.24.4 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheel from the host build (run `python setup.py bdist_wheel`
# inside build/faiss/python before docker build).
COPY build/faiss/python/dist/faiss-*.whl /tmp/

RUN python3.8 -m pip install --no-cache-dir /tmp/faiss-*.whl \
    && rm -f /tmp/faiss-*.whl

# Smoke test at build time — fails fast if the wheel can't load GPU libs.
RUN python3.8 -c "import faiss; print('faiss', faiss.__version__); \
print('SQ4=', faiss.RefineStorageType_SQ4, 'SQ8=', faiss.RefineStorageType_SQ8)"

WORKDIR /workspace
CMD ["python3.8"]
