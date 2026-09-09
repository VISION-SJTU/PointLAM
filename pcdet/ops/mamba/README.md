# Bundled bidirectional Mamba

This directory contains the Mamba implementation used by PointLAM, adapted from the Mamba code included in [LION](https://github.com/happinesslz/LION). It retains the custom bidirectional scan, forward/backward CUDA kernels, and the Python interfaces required by the detector.

Follow the repository's [installation guide](../../../docs/INSTALL.md). Installation compiles the source in this directory; a standard upstream Mamba wheel does not contain the same implementation.

The package retains version 1.1.1 and the `mamba_ssm` import name. Language-model heads, generation utilities, and upstream development/benchmark tooling are outside this package's detection scope.

Original project: [state-spaces/mamba](https://github.com/state-spaces/mamba). The original [license](LICENSE), [authors](AUTHORS), and source notices are retained.
