#!/bin/bash
# Builds build/matmul3 (Assignment 3 seq/par, unchanged) and build/matmul4 (MPI, this assignment).
# Same flags for both: -O2 -fno-tree-vectorize (scalar kernels as in Assignments 2-3).
# mpicc = Open MPI wrapper around the same gcc 11.4.
set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"          # Open MPI installed with: pip install --user openmpi==5.0.11
mkdir -p build
gcc   -O2 -fno-tree-vectorize -fopenmp -std=c11 -Wall -o build/matmul3 src/matmul3.c -lm
mpicc -O2 -fno-tree-vectorize          -std=c11 -Wall -o build/matmul4 src/matmul4.c -lm
echo "build OK"
