# Faiss-GPU Docker 交付

内含 CUDA 12.2 运行时 + Python 3.8 + 已编译的 faiss wheel（含 GPU SQ4/SQ8 refine 支持）。

## 构建

```bash
# 首次或代码有改动时，先重新编译 wheel
make -C build -j faiss swigfaiss
cd build/faiss/python && python setup.py bdist_wheel && cd -

# 打镜像
bash docker/build.sh
# 或指定 tag
IMAGE=my-registry/faiss-gpu:1.8.0 bash docker/build.sh
```

镜像默认 tag：`faiss-gpu-sq:1.8.0`。

## 运行

```bash
# 交互式 Python
docker run --rm --gpus all -it faiss-gpu-sq:1.8.0

# 挂载当前目录跑脚本
docker run --rm --gpus all -v $PWD:/workspace faiss-gpu-sq:1.8.0 \
    python tests/test_gpu_refine_sq4.py
```

## 宿主机要求

- NVIDIA driver ≥ 535（对应 CUDA 12.2 运行时）
- 已安装 [`nvidia-container-toolkit`](https://github.com/NVIDIA/nvidia-container-toolkit)

## 快速验证

镜像 build 阶段已内置 smoke test：

```python
import faiss
print('SQ4=', faiss.RefineStorageType_SQ4, 'SQ8=', faiss.RefineStorageType_SQ8)
```

任何错误会在 `docker build` 阶段就暴露。

## 分发

```bash
# 导出离线 tar
docker save faiss-gpu-sq:1.8.0 | gzip > faiss-gpu-sq-1.8.0.tar.gz

# 对方加载
gunzip -c faiss-gpu-sq-1.8.0.tar.gz | docker load
```

或推到 registry：

```bash
docker tag faiss-gpu-sq:1.8.0 <registry>/faiss-gpu-sq:1.8.0
docker push <registry>/faiss-gpu-sq:1.8.0
```
