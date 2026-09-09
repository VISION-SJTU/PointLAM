# Installation

Use Linux x86-64, Python 3.10, an NVIDIA GPU, and a CUDA development toolkit. The verified combination uses PyTorch 2.1.0 with CUDA 11.8, GCC/G++ 11, spconv 2.3.6, and the bundled Mamba 1.1.1 implementation. H100 was used for the code checks.

A CUDA toolkit containing `nvcc` is required to compile the extensions. The CUDA runtime bundled with PyTorch is insufficient on its own. Set `CUDA_HOME` to your CUDA 11.8 toolkit directory and select a compatible host compiler before building.

## Python environment

Create a separate environment, then run the following commands from the repository root:

```bash
conda create -n pointlam python=3.10 -y
conda activate pointlam

python -m pip install pip==24.0 setuptools==68.2.2 wheel==0.48.0 \
    packaging==26.3 ninja==1.13.2 numpy==1.23.5
python -m pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install --no-build-isolation causal-conv1d==1.2.0.post2
python -m pip install --no-deps waymo-open-dataset-tf-2-11-0==1.6.1
```

The dependency list retains the nuScenes and Argoverse API packages imported by the general dataset registry. Dataset-specific visualization tools may require their upstream optional dependencies.

The Waymo package is installed with `--no-deps` deliberately: this project uses its v1 Frame/TFRecord and detection-metric APIs with `tensorflow-cpu==2.11.0`. Its full package metadata also requests the v2/visualization stack and conflicting exact package versions. Consequently, `pip check` reports Waymo metadata mismatches for this environment; it is not a validated installation of every Waymo API component. The data preparation and detection-metric paths used here were checked separately.

## Build the bundled operators

Set these paths for your machine:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export CC=gcc-11
export CXX=g++-11
export MAX_JOBS=4
```

Build and install the bundled bidirectional Mamba:

```bash
python -m pip install --no-build-isolation --no-deps ./pcdet/ops/mamba
```

This installer builds the bundled CUDA source locally. It does not fetch a stock Mamba wheel. Installing `mamba-ssm` directly from PyPI would not install the project's bidirectional implementation.

Build the OpenPCDet extensions and install the project:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

The build retains eight extension targets required by PointLAM and the retained OpenPCDet modules. For H100, the build can be explicitly targeted with `TORCH_CUDA_ARCH_LIST=9.0`; select the appropriate architecture when building for another GPU. Rebuild the CUDA extensions after changing PyTorch or the CUDA toolchain.

Continue with [Waymo data preparation and training](GETTING_STARTED.md).
