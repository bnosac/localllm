### Contribute

Run pytest locally

```
pip install uv
uv run pytest -ra -q --doctest-modules
```

Run pytest on small set of changes

```
pip uninstall -y localllm
pip install localllm
pytest -ra -q --doctest-modules --ignore=dist --ignore=build
pytest src/localllm/dspy.py
pytest src/localllm/utils.py
pytest src/localllm/config.py
```

Make sure the source code is formatted

```
pip install pre-commit
pre-commit run --all-files
```


### Installing llama-cpp-python

Check your system

```
nvidia-smi
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2025 NVIDIA Corporation
# Built on Fri_Feb_21_20:42:46_Pacific_Standard_Time_2025
# Cuda compilation tools, release 12.8, V12.8.93
# Build cuda_12.8.r12.8/compiler.35583870_0
```

Install llama-cpp-python on Windows

```
##
## E.g. CUDA 12.8 on your home computer
## With NVIDIA RTX 4090 Geforce 16GB GPU RAM
## E.g. RAM: 32GB
## Specify the compute capabilities of your computer manually e.g.: CMAKE_CUDA_ARCHITECTURES=89-real CMAKE_CUDA_ARCHITECTURES_NATIVE=89-real
##              Compute Capability 8.9 (sm_89) corresponds to NVIDIA's Ada Lovelace architecture, commonly found in RTX 40-series GPUs
##
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python==0.3.20 --upgrade --force-reinstall --no-cache-dir --verbose --log install-llamacpp.log
```

### Note for myself

```
# With CUDA (if you have an NVIDIA GPU — replace cu121 with your CUDA version)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

