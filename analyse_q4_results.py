#!/usr/bin/env python3
"""
Analyse Q4 results: pretty-print summary by K and level, and (optional) plots.
"""
import argparse, json, os, math
import numpy as np

def fmt_ci(mean, ci):
    if any([math.isnan(x) for x in [mean]+ci]):
        return "n/a"
    return f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', type=str, default='results_q4')
    ap.add_argument('--make-plots', type=int, default=0)
    args = ap.parse_args()

    summary_path = os.path.join(args.indir, 'summary_q4.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print("# Q4 Summary by K and Complexity Level\n")
    for K in sorted(int(k) for k in summary.keys()):
        print(f"\n## K={K}\n")
        levels = sorted(int(l) for l in summary[str(K)].keys())
        print("Level | Trials | Success Rate (mean [CI95]) | Time on Successes (s) | Joint Path Length (a.u.)")
        print(":---: | :----: | :-------------------------: | :--------------------: | :----------------------: ")
        for L in levels:
            s = summary[str(K)][str(L)]
            sr = fmt_ci(s['success_rate_mean'], s['success_rate_CI95'])
            tm = fmt_ci(s['time_sec_mean_successes'], s['time_sec_CI95_successes'])
            pl = fmt_ci(s['path_len_mean_successes'], s['path_len_CI95_successes'])
            print(f" {L} | {s['n_trials']} | {sr} | {tm} | {pl}")

    if args.make_plots:
        import matplotlib.pyplot as plt
        for K in sorted(int(k) for k in summary.keys()):
            levels = sorted(int(l) for l in summary[str(K)].keys())
            y = [summary[str(K)][str(L)]['success_rate_mean'] for L in levels]
            lo = [ (y[i] - summary[str(K)][str(levels[i])]['success_rate_CI95'][0]) for i in range(len(levels)) ]
            hi = [ (summary[str(K)][str(levels[i])]['success_rate_CI95'][1] - y[i]) for i in range(len(levels)) ]
            plt.figure()
            plt.errorbar(levels, y, yerr=[lo, hi], fmt='o-')
            plt.xlabel('Complexity level')
            plt.ylabel('Success rate')
            plt.title(f'Q4: Success vs complexity (K={K})')
            plt.grid(True, linestyle=':')
            plt.tight_layout()
            plt.savefig(os.path.join(args.indir, f'q4_success_vs_complexity_K{K}.png'), dpi=150)
        print("\nSaved plots to:")
        for K in sorted(int(k) for k in summary.keys()):
            print(" -", os.path.join(args.indir, f'q4_success_vs_complexity_K{K}.png'))

if __name__ == '__main__':
    main()
