# InSituANN Open-Source Release

This is the open-source release for InSituANN.  It keeps the core
implementation and the scripts needed to reproduce the main paper pipeline,
while leaving out exploratory kernels, temporary remote scripts, large datasets,
and large generated result directories.

## What Is Included

- `cuda/`: core CUDA/C++ implementation for IVF routing, GPU top-k selection,
  k-means/reorder utilities, Exact-CPU fine search, CPU residual-PQ kernels, and
  shared GPU utilities.
- `scripts/`: selected dataset conversion, ground-truth generation, PQ
  training, and fast-build scripts.
- `updates/`: CUDA update-overlay runners for Exact-CPU and PQ-GPU.
- `configs/`: paper configuration templates for 1B and 100M experiments.

## What Is Not Included

- Raw vectors, ground truth files, centroids, assignment arrays, reordered
  vectors, PQ codes, and other large artifacts.
- Paper evaluation CSVs, plotting scripts, and full experiment result
  directories.
- Draft notes, paper figures, local documents, and intermediate experiment logs.
- Internal validation suites and local datasets.
- Generated paper runners and external launch scripts.
- Legacy exploratory implementations such as AoSoA/warpsort variants, IMI,
  OPQ/RabitQ experiments, failed fused-assignment prototypes, and temporary
  remote patches.
- Provider credentials, SSH settings, API keys, or machine-specific launch
  state.

## Build

The original CMake project is preserved.  A typical build is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

For older GPUs, replace `120` with the correct CUDA architecture.  The code was
developed with CUDA 13.0 and C++17.

The update-overlay runners can be built after the core project libraries:

```bash
cd updates
./build.sh
```

## Reproducing Paper Results

The full 100M/1B experiments require external datasets and generated index
artifacts.  The expected configuration is summarized in `configs/`.

The high-level order is:

1. Download or prepare base/query/ground-truth files.
2. Train IVF centroids and assign base vectors.
3. Reorder vectors into cluster-major layout.
4. Optionally train residual-PQ codebooks and encode residual-PQ codes.
5. Run Exact-CPU or PQ-GPU sweeps.
6. Run the paper evaluation harness separately if full result reproduction is
   required.

## License

No license file is included in this release yet.  Add the intended
project license before publishing the repository publicly.
