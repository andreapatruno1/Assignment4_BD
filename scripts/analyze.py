"""Builds the summary tables and figures of Assignment 4 from the raw CSV files.

Statistic: median of the 5 repetitions; variability: IQR = Q3 - Q1.
Speedups use medians: S_seq = T_seq / T, S_par = T_par(p=2) / T, strong scaling S_mpi(P) = T_mpi(1) / T_mpi(P),
efficiency E = S_mpi(P) / P.
Outputs: results/summary.csv, results/summary_phases.csv, results/summary_transport.csv,
         results/summary_launch.csv, results/traffic.csv, results/fig/*.png
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "fig")
os.makedirs(FIG, exist_ok=True)
MiB = 1024.0


def q(x, p):
    return float(np.percentile(x, p))


sh = pd.read_csv(os.path.join(RES, "raw_shared.csv"))
mp = pd.read_csv(os.path.join(RES, "raw_mpi.csv"))
sw = mp[mp.experiment == "sweep"]

# every checksum of one n must be identical (same result for every method and P)
chk = pd.concat([sh[["n", "checksum"]], mp[["n", "checksum"]]])
assert (chk.groupby("n").checksum.nunique() == 1).all(), "checksums differ between methods"

rows = []
for (m, n, p), g in sh.groupby(["method", "n", "p"]):
    t = g.time_s.values
    rows.append(dict(method=f"{m}" if m == "seq" else f"par", n=n, P=p, reps=len(t), median_s=np.median(t),
                     q1_s=q(t, 25), q3_s=q(t, 75), iqr_s=q(t, 75) - q(t, 25),
                     rss_total_MiB=g.peak_rss_kb.max() / MiB, rss_root_MiB=g.peak_rss_kb.max() / MiB,
                     rss_worker_max_MiB=0.0, storage_MiB=3 * 8 * n * n / 2**20))
for (n, P), g in sw.groupby(["n", "P"]):
    t = g.total_s.values
    w = 8 * n * n * (1 + 2 / P) / 2**20 if P > 1 else 0.0    # B + band of A + band of C on a worker
    rows.append(dict(method="mpi", n=n, P=P, reps=len(t), median_s=np.median(t), q1_s=q(t, 25), q3_s=q(t, 75),
                     iqr_s=q(t, 75) - q(t, 25), rss_total_MiB=g.rss_sum_kb.max() / MiB,
                     rss_root_MiB=g.rss_root_kb.max() / MiB, rss_worker_max_MiB=g.rss_worker_max_kb.max() / MiB,
                     storage_MiB=3 * 8 * n * n / 2**20 + (P - 1) * w))
s = pd.DataFrame(rows).sort_values(["n", "method", "P"])
seq = s[s.method == "seq"].set_index("n").median_s
par = s[s.method == "par"].set_index("n").median_s
mp1 = s[(s.method == "mpi") & (s.P == 1)].set_index("n").median_s
s["S_vs_seq"] = s.n.map(seq) / s.median_s
s["S_vs_par2"] = s.n.map(par) / s.median_s
s["S_mpi"] = np.where(s.method == "mpi", s.n.map(mp1) / s.median_s, np.nan)
s["E_mpi"] = s.S_mpi / s.P
s["oversubscribed"] = ((s.method == "mpi") & (s.P > 2)).astype(int)
s.to_csv(os.path.join(RES, "summary.csv"), index=False, float_format="%.6g")

ph = sw.groupby(["n", "P"]).agg(total=("total_s", "median"), scatter=("scatter_s", "median"),
                                bcast=("bcast_s", "median"), compute=("compute_s", "median"),
                                gather=("gather_s", "median"), comp_max=("comp_max_s", "median"),
                                comp_min=("comp_min_s", "median"), init=("init_s", "median"),
                                load=("load_s", "median")).reset_index()
ph["comm"] = ph.scatter + ph.bcast + ph.gather
ph["comm_frac"] = ph.comm / ph.total
ph["imbalance"] = ph.comp_max / ph.comp_min
ph["bytes_moved_MB"] = [(2 * (n - n // P - (1 if n % P else 0)) * n + (P - 1) * n * n) * 8 / 1e6
                        for n, P in zip(ph.n, ph.P)]   # scatter + gather + (P-1) logical copies of B
ph.to_csv(os.path.join(RES, "summary_phases.csv"), index=False, float_format="%.6g")

tr = mp[mp.experiment == "transport"].groupby(["n", "P", "transport"]).agg(
    total=("total_s", "median"), iqr=("total_s", lambda x: q(x, 75) - q(x, 25)), scatter=("scatter_s", "median"),
    bcast=("bcast_s", "median"), compute=("compute_s", "median"), gather=("gather_s", "median")).reset_index()
tr["comm"] = tr.scatter + tr.bcast + tr.gather
tr.to_csv(os.path.join(RES, "summary_transport.csv"), index=False, float_format="%.6g")

if os.path.exists(os.path.join(RES, "launch.csv")):
    la = pd.read_csv(os.path.join(RES, "launch.csv")).groupby(["transport", "P"]).wall_s.agg(
        median="median", q1=lambda x: q(x, 25), q3=lambda x: q(x, 75)).reset_index()
    la.to_csv(os.path.join(RES, "summary_launch.csv"), index=False, float_format="%.6g")

# measured point-to-point traffic (pml monitoring)
tl = []
for f in sorted(glob.glob(os.path.join(RES, "traffic", "P*.prof"))):
    P = int(re.search(r"P(\d+)\.", os.path.basename(f)).group(1))
    for line in open(f):
        m = re.match(r"E\s+(\d+)\s+(\d+)\s+(\d+) bytes\s+(\d+) msgs", line)
        if m:
            tl.append(dict(n=2048, P=P, src=int(m[1]), dst=int(m[2]), bytes=int(m[3]), msgs=int(m[4])))
if tl:
    tl = pd.DataFrame(tl)
    tl.to_csv(os.path.join(RES, "traffic.csv"), index=False)
    big_msgs = tl[tl.bytes > 1024]
    tt = big_msgs.groupby("P").agg(total_MiB=("bytes", lambda x: x.sum() / 2**20)).reset_index()
    tt["root_sent_MiB"] = [big_msgs[(big_msgs.P == P) & (big_msgs.src == 0)].bytes.sum() / 2**20 for P in tt.P]
    tt["root_recv_MiB"] = [big_msgs[(big_msgs.P == P) & (big_msgs.dst == 0)].bytes.sum() / 2**20 for P in tt.P]
    tt["model_MiB"] = [(2 * (2048 - 2048 // P) * 2048 + (P - 1) * 2048 ** 2) * 8 / 2**20 for P in tt.P]
    tt.to_csv(os.path.join(RES, "summary_traffic.csv"), index=False, float_format="%.6g")
    print(tt.to_string())

# control experiment: interleaved rounds at n = 2048 (1 warm-up + 3 repetitions per configuration and round)
ck = []
if os.path.exists(os.path.join(RES, "raw_check_shared.csv")):
    cs = pd.read_csv(os.path.join(RES, "raw_check_shared.csv"))
    for (r, m, p), g in cs.groupby(["round", "method", "p"]):
        ck.append(dict(round=r, config=f"{m} p={p}", median_s=g.time_s.median(), min_s=g.time_s.min(), max_s=g.time_s.max()))
    for e, g in mp[mp.experiment.str.startswith("check")].groupby(["experiment", "P"]):
        ck.append(dict(round=int(e[0][5:]), config=f"mpi P={e[1]}", median_s=g.total_s.median(),
                       min_s=g.total_s.min(), max_s=g.total_s.max()))
    ck = pd.DataFrame(ck).sort_values(["round", "config"])
    ck.to_csv(os.path.join(RES, "summary_check.csv"), index=False, float_format="%.6g")
    print(ck.pivot(index="round", columns="config", values="median_s").to_string())

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 9})
col = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}

fig, ax = plt.subplots(figsize=(6, 3.8))
for m, lab, st in [("seq", "seq (A3 baseline)", "k-o"), ("par", "par OpenMP p=2 (A3)", "k--s")]:
    d = s[s.method == m]
    ax.errorbar(d.n, d.median_s, yerr=[d.median_s - d.q1_s, d.q3_s - d.median_s], fmt=st, ms=4, capsize=2, label=lab)
for P in (1, 2, 3, 4):
    d = s[(s.method == "mpi") & (s.P == P)]
    ax.errorbar(d.n, d.median_s, yerr=[d.median_s - d.q1_s, d.q3_s - d.median_s], fmt="-^", color=col[P], ms=4,
                capsize=2, label=f"MPI P={P}" + (" (oversubscribed)" if P > 2 else ""))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(sorted(s.n.unique())); ax.set_xticklabels(sorted(s.n.unique())); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("matrix size n"); ax.set_ylabel("time per multiplication [s] (median, IQR bars)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "time.png"), dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
d = ph[(ph.n == 2048) & (ph.P > 1)].sort_values("P")
x = np.arange(len(d)); bot = np.zeros(len(d))
for c, lab, cc in [("scatter", "scatter A", "#9467bd"), ("bcast", "bcast B", "#8c564b"), ("gather", "gather C", "#e377c2")]:
    axs[0].bar(x, d[c] * 1e3, bottom=bot, color=cc, label=lab); bot += d[c].values * 1e3
for xi, (tot, cm) in enumerate(zip(d.total, d.comm)):
    axs[0].text(xi, cm * 1e3 + 0.5, f"{100 * cm / tot:.2f}% of total", ha="center", fontsize=7)
axs[0].set_xticks(x); axs[0].set_xticklabels([f"P={p}" for p in d.P])
axs[0].set_ylabel("median time [ms]"); axs[0].set_title("communication phases, n = 2048", fontsize=9)
axs[0].set_ylim(0, bot.max() * 1.25); axs[0].legend(fontsize=7); axs[0].grid(axis="y", alpha=.3)
for P in (2, 3, 4):
    d = ph[ph.P == P].sort_values("n")
    axs[1].plot(d.n, 100 * d.comm_frac, "-o", ms=4, color=col[P], label=f"P={P}")
axs[1].set_xscale("log"); axs[1].set_yscale("log")
axs[1].set_xticks(sorted(ph.n.unique())); axs[1].set_xticklabels(sorted(ph.n.unique())); axs[1].minorticks_off()
axs[1].set_xlabel("matrix size n"); axs[1].set_ylabel("communication / total [%]")
axs[1].set_title("share of communication in the total", fontsize=9); axs[1].grid(alpha=.3, which="both"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "communication.png"), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for n in sorted(s.n.unique()):
    d = s[(s.method == "mpi")].query("n == @n").sort_values("P")
    if len(d) > 1 and (d.P == 1).any():
        ax.plot(d.P, d.S_mpi, "-o", ms=4, label=f"n={n}")
ax.plot([1, 4], [1, 4], ":", color="gray", label="ideal S=P")
ax.axhline(2, color="gray", lw=.8); ax.axvspan(2.5, 4.2, color="#eee", label="P > 2 vCPU")
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("MPI processes P"); ax.set_ylabel("strong-scaling speedup T(1)/T(P)")
ax.set_ylim(0, 4.2); ax.grid(alpha=.3); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "speedup.png"), dpi=160); plt.close(fig)

print(s.to_string()); print(ph.to_string()); print(tr.to_string())
if os.path.exists(os.path.join(RES, "summary_launch.csv")):
    print(pd.read_csv(os.path.join(RES, "summary_launch.csv")).to_string())

