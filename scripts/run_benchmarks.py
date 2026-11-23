"""Benchmark sweep of Assignment 4 (appends raw rows to results/).

Protocol (fixed before running, same as Assignments 2-3):
  warm-up 1 run (not reported), 5 measured repetitions per configuration, one process (or one mpirun) per configuration
  budget  20 s for ONE timed multiplication (BUDGET_S); a timeout goes to results/limits.csv, never as a time.
          Memory budget 3 GB (sum of the peak RSS of all processes).
  order   for each n: seq, par (p = 2), then mpi with P = 1, 2, 3, 4 ranks, all close in time, same input files.
  mpirun  --oversubscribe --bind-to none --mca pml ob1 --mca btl self,<transport>
          transport sm  = shared memory between the processes (default experiment)
          transport tcp = TCP over the loopback interface (lo), only in the transport experiment

Experiments:
  sweep      --n 512 1024 1536 2048 3072
  transport  --transport-n 2048 : P = 2, 4 with sm and tcp, interleaved
  check      --check-round R : control experiment at n = 2048, configurations interleaved in one round
             (seq, mpi P=1, par p=2, mpi P=2), 1 warm-up + 3 repetitions each; run once per round
  launch     wall time of `mpirun -np P matmul4 init` (launch + MPI_Init + MPI_Finalize), P = 1..4

Each shell call of the original run was limited to 3 minutes, so the sweep was executed in several
invocations with --cfg (same protocol per configuration, same order seq, par, mpi1..4 inside each n).

Outputs: results/raw_shared.csv  (A3 format)  method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb
         results/raw_mpi.csv     experiment,transport,<matmul4 HEADER>
         results/launch.csv      P,transport,rep,wall_s
         results/limits.csv      method,n,P,event,limit
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
B3, B4 = os.path.join(ROOT, "build", "matmul3"), os.path.join(ROOT, "build", "matmul4")
RES = os.path.join(ROOT, "results")
DATA = os.environ.get("MATMUL_DATA", os.path.join(ROOT, "data"))
WARMUP, REPS, BUDGET_S = 1, 5, 20
MPIRUN = os.path.expanduser("~/.local/bin/mpirun")
HEAD = {"raw_shared.csv": "method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb",
        "raw_mpi.csv": "experiment,transport,method,n,P,rep,total_s,scatter_s,bcast_s,compute_s,gather_s,"
                       "comp_max_s,comp_min_s,init_s,load_s,checksum,rss_root_kb,rss_worker_max_kb,rss_sum_kb",
        "launch.csv": "P,transport,rep,wall_s",
        "raw_check_shared.csv": "round,method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb",
        "limits.csv": "method,n,P,event,limit"}


def append(name, lines):
    path = os.path.join(RES, name)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(HEAD[name] + "\n")
        f.writelines(x + "\n" for x in lines)


def mpi_cmd(P, transport):
    cmd = [MPIRUN, "--oversubscribe", "--bind-to", "none", "--mca", "pml", "ob1", "--mca", "btl", f"self,{transport}"]
    if transport == "tcp":
        cmd += ["--mca", "btl_tcp_if_include", "lo"]
    return cmd + ["-np", str(P)]


def run_shared(method, n, p, out="raw_shared.csv", prefix=""):
    env = dict(os.environ, BUDGET_S=str(BUDGET_S), OMP_NUM_THREADS=str(p), OMP_DYNAMIC="false")
    r = subprocess.run([B3, "bench", method, str(n), str(p), str(WARMUP), str(REPS), DATA],
                       capture_output=True, text=True, env=env)
    rows = [x for x in r.stdout.splitlines() if x.strip()]
    if r.returncode == 0:
        append(out, [prefix + x for x in rows])
    else:
        append("limits.csv", [f"{method},{n},{p},{'timeout' if r.returncode == 3 else 'failure rc ' + str(r.returncode)},{BUDGET_S} s"])
    print(method, n, p, "rc", r.returncode, rows[-1] if rows else "", flush=True)
    return r.returncode == 0


def run_mpi(n, P, transport, experiment):
    env = dict(os.environ, BUDGET_S=str(BUDGET_S))
    r = subprocess.run(mpi_cmd(P, transport) + [B4, "bench", str(n), str(WARMUP), str(REPS), DATA],
                       capture_output=True, text=True, env=env)
    rows = [x for x in r.stdout.splitlines() if x.startswith("mpi,")]
    ok = r.returncode == 0 and len(rows) == REPS
    if ok:
        append("raw_mpi.csv", [f"{experiment},{transport},{x}" for x in rows])
    else:
        ev = "timeout" if "TIMEOUT" in r.stderr else f"failure rc {r.returncode}"
        append("limits.csv", [f"mpi-{transport},{n},{P},{ev},{BUDGET_S} s"])
        if ev != "timeout":
            print(r.stderr[-2000:], file=sys.stderr)
    print("mpi", transport, n, P, "ok" if ok else "FAIL", rows[-1] if rows else "", flush=True)
    return ok


def launch(Ps, transports):
    for transport in transports:
        for P in Ps:
            for rep in range(-WARMUP, REPS):
                t0 = time.perf_counter()
                subprocess.run(mpi_cmd(P, transport) + [B4, "init"], check=True, capture_output=True)
                w = time.perf_counter() - t0
                if rep >= 0:
                    append("launch.csv", [f"{P},{transport},{rep},{w:.6f}"])
            print("launch", transport, P, "done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs="*", type=int, default=[])
    ap.add_argument("--P", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--transport-n", type=int, default=0)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--check-round", type=int, default=0)
    ap.add_argument("--cfg", nargs="+", default=None,
                    help="subset of configurations, e.g. seq par mpi1 mpi2 (default: all, in the order above)")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    stopped = set()
    for n in a.n:                                    # main sweep
        for cfg in [("seq", 1), ("par", 2)] + [("mpi", P) for P in a.P]:
            if cfg in stopped or (a.cfg and (cfg[0] if cfg[0] != "mpi" else f"mpi{cfg[1]}") not in a.cfg):
                continue
            ok = run_shared(cfg[0], n, cfg[1]) if cfg[0] != "mpi" else run_mpi(n, cfg[1], "sm", "sweep")
            if not ok:
                stopped.add(cfg)
    if a.transport_n:                                # transport experiment (interleaved)
        for P in (2, 4):
            for transport in ("sm", "tcp"):
                run_mpi(a.transport_n, P, transport, "transport")
    if a.check_round:                                # control experiment, one round per call
        global REPS
        REPS = 3
        k = a.check_round
        run_shared("seq", 2048, 1, "raw_check_shared.csv", f"{k},")
        run_mpi(2048, 1, "sm", f"check{k}")
        run_shared("par", 2048, 2, "raw_check_shared.csv", f"{k},")
        run_mpi(2048, 2, "sm", f"check{k}")
    if a.launch:
        launch(a.P, ["sm", "tcp"])


if __name__ == "__main__":
    main()
