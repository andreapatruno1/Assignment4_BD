#!/bin/bash
# Records the execution environment in results/environment.txt
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
{
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== CPU =="; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|NUMA|L1d|L2|L3|Hypervisor"
echo "nproc (execution contexts available): $(nproc)"
echo "physical machines used: 1 ($(hostname)); all MPI processes run on this VM"
echo "== memory =="; free -m
echo "== network interfaces =="; ip -br addr 2>/dev/null
echo "== OS =="; grep PRETTY_NAME /etc/os-release; uname -r
echo "== C =="; gcc --version | head -1; echo "flags: -O2 -fno-tree-vectorize -std=c11 -Wall -lm (+ -fopenmp for matmul3)"
echo "== MPI =="; mpirun --version | head -1; echo "mpicc: $(mpicc --showme)"
echo "mpirun options: --oversubscribe --bind-to none --mca pml ob1 --mca btl self,sm | self,tcp (btl_tcp_if_include lo)"
echo "OpenMP (par only): OMP_NUM_THREADS=2, OMP_DYNAMIC=false, no affinity"
echo "Numerical libraries in the benchmark processes: none (no BLAS); every MPI rank is single-threaded"
echo "== Python (inputs and analysis only) =="; python3 --version; python3 -c "import numpy, matplotlib; print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"
} > results/environment.txt
cat results/environment.txt
