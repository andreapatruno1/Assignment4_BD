# Assignment 4 — Distributed Dense Matrix Multiplication with MPI (C, Open MPI)

Big Data, ULPGC, 2026-2027 · Andrea Patruno · Report: `Assignment04_Patruno_Report.pdf` (source `report/report.md`)

Question: how does the run time of an MPI matrix multiplication change with the number of processes P (strong scaling),
how is it divided among computation, communication and framework overhead, and how does it compare with the
sequential and OpenMP versions of Assignment 3 on the same hardware?

Reused: `src/matmul3.c` (seq and par, unchanged) and `scripts/gen_inputs.py` (generator of Assignments 1-3) from
https://github.com/andreapatruno1/Assignment3_BD, commit f3b3986.

## Contents

| Path | What it is |
|---|---|
| `src/matmul4.c` | MPI row-block multiplication: Scatterv(A), Bcast(B), A3 row kernel, Gatherv(C); phase timing; tests |
| `src/matmul3.c` | Assignment 3 seq / par (OpenMP), unchanged, used for the comparison |
| `scripts/build.sh` | builds `build/matmul3` (gcc) and `build/matmul4` (mpicc), `-O2 -fno-tree-vectorize` |
| `scripts/gen_inputs.py` | inputs: NumPy PCG64, seed 42 + n, uniform [-1,1), raw float64, `A_n.bin`, `B_n.bin` |
| `scripts/run_benchmarks.py` | sweep, transport (sm vs tcp), control rounds, launch overhead (1 warm-up + 5 reps, 20 s budget) |
| `scripts/measure_traffic.sh` | bytes between ranks with Open MPI `pml monitoring` (n = 2048, P = 2, 3, 4) |
| `scripts/env_info.sh` | `results/environment.txt` |
| `scripts/analyze.py` | `results/summary*.csv`, `results/traffic.csv`, `results/fig/{time,speedup,communication}.png` |
| `results/raw_shared.csv` | seq/par rows: method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb |
| `results/raw_mpi.csv` | MPI rows: experiment,transport,method,n,P,rep,total_s,scatter_s,bcast_s,compute_s,gather_s,comp_max_s,comp_min_s,init_s,load_s,checksum,rss_root_kb,rss_worker_max_kb,rss_sum_kb |
| `results/raw_check_shared.csv` | seq/par rows of the control rounds (MPI rows of the rounds: experiment `check1..3` in raw_mpi.csv) |
| `results/launch.csv`, `results/limits.csv`, `results/correctness.csv`, `results/traffic/` | launch overhead, timeouts, tests, monitoring profiles |

## Environment and configuration

One Ubuntu 22.04 Hyper-V VM (2 logical processors, 3.8 GiB) on an Intel i7-1255U laptop: **local execution, one
physical machine**. P = 1, 2, 3, 4 single-threaded MPI processes (P > 2 oversubscribed).
`mpirun --oversubscribe --bind-to none --mca pml ob1 --mca btl self,sm` (transport experiment also
`--mca btl self,tcp --mca btl_tcp_if_include lo`). Open MPI 5.0.11, GCC 11.4, Python 3.10 + numpy, pandas, matplotlib.

## Exact commands

```bash
pip install --user openmpi==5.0.11                         # provides mpicc / mpirun in ~/.local/bin
export PATH=$HOME/.local/bin:$PATH MATMUL_DATA=$HOME/data  # inputs (~270 MB) kept outside the repository
bash scripts/build.sh
./build/matmul3 test > /dev/null                            # A3 tests
echo "test,n,P,max_abs_diff,pass" > results/correctness.csv
for P in 1 2 3 4; do mpirun --oversubscribe --bind-to none --mca pml ob1 --mca btl self,sm -np $P \
    build/matmul4 test | tail -n +2 >> results/correctness.csv; done   # must print ALL TESTS PASSED
python3 scripts/gen_inputs.py 512 1024 1536 2048 3072
python3 scripts/run_benchmarks.py --n 512 1024 1536 2048 3072   # main sweep
python3 scripts/run_benchmarks.py --transport-n 2048 --launch   # sm vs tcp, launch overhead
for r in 1 2 3; do python3 scripts/run_benchmarks.py --check-round $r; done   # control rounds
bash scripts/measure_traffic.sh
bash scripts/env_info.sh
python3 scripts/analyze.py
python3 report/build_pdf.py Assignment04_Patruno_Report.pdf   # needs python markdown + playwright
```

Rows are appended: delete `results/raw*.csv`, `results/launch.csv`, `results/limits.csv` before a full re-run.
In the original run each shell call was limited to 3 minutes, so the sweep was run in several invocations
(`--n 2048 --cfg seq par mpi1`, `--cfg mpi2 mpi3 mpi4`, ...); the protocol per configuration is the same.
