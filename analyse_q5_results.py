#!/usr/bin/env python3
"""
analyse_q5_results.py — Q5 helper

Reads the runner output directory (same format as Q4) and
summarizes performance as the number of drones K increases.
It prints tables and optionally saves plots vs K for each complexity level.

Usage:
  python analyse_q5_results.py --indir results_q4_k_part1 --make-plots 1
  # or if you merged everything:
  python analyse_q5_results.py --indir results_q4_k_all --make-plots 1
"""
import argparse, os, json, math
import numpy as np

def fmt_ci(mean, ci):
    if any([math.isnan(x) for x in [mean]+ci]):
        return "n/a"
    return f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', type=str, required=True, help='Directory containing summary_q4.json')
    ap.add_argument('--make-plots', type=int, default=0)
    args = ap.parse_args()

    summary_path = os.path.join(args.indir, 'summary_q4.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} not found. Run the runner first or merge CSVs to build it.")
    with open(summary_path, 'r') as f:
        S = json.load(f)

    Ks = sorted(int(k) for k in S.keys())
    levels = sorted({int(L) for k in Ks for L in S[str(k)].keys()})

    print("# Q5: Summary vs K (for each complexity level)\n")
    for L in levels:
        print(f"\n## Level {L}\n")
        print(" K | Trials | Success Rate (mean [CI95]) | Time on Successes (s) | Joint Path Length (a.u.)")
        print(":--:|:------:|:---------------------------:|:----------------------:|:------------------------:")
        for K in Ks:
            s = S[str(K)].get(str(L))
            if s is None:
                print(f" {K} | 0 | n/a | n/a | n/a")
                continue
            sr = fmt_ci(s['success_rate_mean'], s['success_rate_CI95'])
            tm = fmt_ci(s['time_sec_mean_successes'], s['time_sec_CI95_successes'])
            pl = fmt_ci(s['path_len_mean_successes'], s['path_len_CI95_successes'])
            print(f" {K} | {s['n_trials']} | {sr} | {tm} | {pl}")

    if args.make_plots:
        import matplotlib.pyplot as plt
        # Success vs K for each level
        for L in levels:
            y, lo, hi = [], [], []
            for K in Ks:
                s = S[str(K)].get(str(L))
                if s is None:
                    y.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                m, (a,b) = s['success_rate_mean'], s['success_rate_CI95']
                y.append(m); lo.append(m-a); hi.append(b-m)
            plt.figure()
            plt.errorbar(Ks, y, yerr=[lo,hi], fmt='o-')
            plt.xlabel('K (number of drones)'); plt.ylabel('Success rate')
            plt.title(f'Q5: Success vs K (Level {L})'); plt.grid(True, linestyle=':')
            plt.tight_layout(); plt.savefig(os.path.join(args.indir, f'q5_success_vs_K_level{L}.png'), dpi=150)

        # Time vs K (on successes)
        for L in levels:
            y, lo, hi = [], [], []
            for K in Ks:
                s = S[str(K)].get(str(L))
                if s is None:
                    y.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                m, (a,b) = s['time_sec_mean_successes'], s['time_sec_CI95_successes']
                y.append(m); lo.append(m-a); hi.append(b-m)
            plt.figure()
            plt.errorbar(Ks, y, yerr=[lo,hi], fmt='o-')
            plt.xlabel('K (number of drones)'); plt.ylabel('Time on successes (s)')
            plt.title(f'Q5: Time vs K (Level {L})'); plt.grid(True, linestyle=':')
            plt.tight_layout(); plt.savefig(os.path.join(args.indir, f'q5_time_vs_K_level{L}.png'), dpi=150)

        # Path length vs K (on successes)
        for L in levels:
            y, lo, hi = [], [], []
            for K in Ks:
                s = S[str(K)].get(str(L))
                if s is None:
                    y.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                m, (a,b) = s['path_len_mean_successes'], s['path_len_CI95_successes']
                y.append(m); lo.append(m-a); hi.append(b-m)
            plt.figure()
            plt.errorbar(Ks, y, yerr=[lo,hi], fmt='o-')
            plt.xlabel('K (number of drones)'); plt.ylabel('Joint path length (a.u.)')
            plt.title(f'Q5: Joint path length vs K (Level {L})'); plt.grid(True, linestyle=':')
            plt.tight_layout(); plt.savefig(os.path.join(args.indir, f'q5_pathlen_vs_K_level{L}.png'), dpi=150)

        print("\nSaved plots to:")
        for L in levels:
            print(" -", os.path.join(args.indir, f'q5_success_vs_K_level{L}.png'))
            print(" -", os.path.join(args.indir, f'q5_time_vs_K_level{L}.png'))
            print(" -", os.path.join(args.indir, f'q5_pathlen_vs_K_level{L}.png'))

if __name__ == '__main__':
    main()
