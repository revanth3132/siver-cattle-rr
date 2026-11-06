"""make_tables.py - regenerate Tables 3, 4 and 5 of the paper.

The reported results aggregate 120 animal-disjoint splits x 12 two-domain training runs. This script
 recomputes the tables from the per-run correlations the modelling produced (data/run_correlations.pkl) 
 rather than refitting the models.
`run_protocol.py --seeds N --check` refits those splits from the feature vectors and diffs them
against the stored values, so the shortcut is verifiable rather than asserted.

Correlations are combined in two stages: the 12 runs within a split, then the 120 splits, both in Fisher z 
and back-transformed. The unit is the run within a split -- 1,440 run-level correlations -- each weighted 
equally rather than by test-set size, since runs hold 52-59 clips. MAE is averaged arithmetically at both 
stages, having no bounded-range problem. Pass --agg arithmetic to see what naive averaging of r would have given; 
MAE is identical either way and every correlation moves toward zero.

Table 4 is not aggregated: it is a single deterministic run.

Usage:  python make_tables.py [--agg fisher|arithmetic] [--latex]
"""

import argparse
import collections
import json
import os
import pickle

import numpy as np
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TIERS = ("seen", "unseen", "combined")

# (row label, Data column, Instantiation column, model key in the stored runs)
TABLE3 = [
    ("Image features", "L", "Pooled DINOv2 + regressor", "Deep: dinov2s (pooled + regressor)"),
    ("Video features", "L", "Pooled VideoMAE + regressor", "Deep: videomae (pooled + regressor)"),
    ("Temporal visual model", "L", "Transformer + frame-level DINOv2",
     "Temporal visual model (transformer over frame DINOv2)"),
    ("Motion spectrum", "L", "Masked EVM + flow + FFT", "Motion spectrum (EVM flow, 1 view)"),
    ("Learned fusion", "L", "Visual + motion stacking", "Learned fusion (visual + motion stacking)"),
    ("Conventional flank tracking", "L", "Fixed torso band + EVM flow", "Flank + EVM"),
    ("Conventional flank tracking", "L", "Fixed torso band + frame difference", "Flank, no EVM"),
    ("SiVeR", "L", "10 signal regressors + median", "SiVeR (10 views, median)"),
    ("SiVeR-10+", "L", "Rarity weights + regressor averaging", "SiVeR-10+ (weights + averaging)"),
    ("SiVeR-ST", "L+P", "SiVeR + pseudo-label retraining", "SiVeR-TS (all pseudo, median)"),
    ("SiVeR-10+ST", "L+P", "SiVeR-10+ + aligned pseudo-labels", "SiVeR-10+ TS (pseudo-labels)"),
]

# (row label, Change Tested, Views, Agg., Pseudo, model key, which run supplies it)
TABLE5 = [
    ("Single-signal motion", "Masked EVM + flow + FFT", "1", "--", "--",
     "Motion spectrum (EVM flow, 1 view)", "reported"),
    ("Multi-signal motion", "10 respiratory signal views", "10", "Mean", "--",
     "Multi-signal (10 views, mean)", "reported"),
    ("SiVeR", "Median signal-view aggregation", "10", "Median", "--",
     "SiVeR (10 views, median)", "reported"),
    ("Learned fusion", "Replace median with learned stacking", "10", "Learned", "--",
     "Learned fusion (stacking, 10 views)", "reported"),
    ("SNR weighting", "Replace median with SNR-weighted averaging", "10", "SNR", "--",
     "SNR-weighted (10 views)", "reported"),
    ("Temporal TTA", "Average sub-window predictions", "10", "Temporal", "--",
     "Temporal TTA (10 views, median)", "reported"),
    ("SiVeR, no EVM", "Remove EVM-derived views", "7", "Median", "--",
     "SiVeR, no-EVM views (median)", "reported"),
    ("SiVeR + weighting only", "Add RR rarity weighting only", "10", "Median", "--",
     "SiVeR-10+ weighting only (single regressor)", "reported"),
    ("SiVeR + averaging only", "Add regressor averaging only", "10", "Median", "--",
     "SiVeR-10+ averaging only (no rarity weights)", "reported"),
    ("SiVeR-10+", "Rarity weights + regressor averaging", "10", "Median", "--",
     "SiVeR-10+ (weights + averaging)", "reported"),
    ("SiVeR-ST", "Direct pseudo-label retraining", "10", "Median", "All",
     "SiVeR-TS (all pseudo, median)", "reported"),
    ("SiVeR-10+ + direct ST", "Add auxiliary pseudo-label retraining", "10", "Median", "All",
     "SiVeR-10+ TS (pseudo-labels)", "direct_st"),
    ("SiVeR-10+ + aligned ST", "Add pseudo-label distribution alignment", "10", "Median", "All",
     "SiVeR-10+ TS (pseudo-labels)", "aligned_st"),
    ("SiVeR-10+ + aligned ST + calibration", "Add training-only output calibration", "10",
     "Median", "All", "SiVeR-10+ TS (pseudo-labels)", "reported"),
    ("SiVeR-10+ST", "Aligned pseudo-label self-training", "10", "Median", "All",
     "SiVeR-10+ TS (pseudo-labels)", "reported"),
]

TABLE4 = [("Image features", "Deep: dinov2s (pooled + regressor)"),
          ("Video features", "Deep: videomae (pooled + regressor)"),
          ("Temporal visual model", "Temporal visual model (transformer over frame DINOv2)"),
          ("Motion spectrum", "Motion spectrum (EVM flow, 1 view)"),
          ("Learned fusion", "Learned fusion (visual + motion stacking)"),
          ("Flank + EVM", "Flank + EVM"), ("Flank, no EVM", "Flank, no EVM"),
          ("SiVeR", "SiVeR (10 views, median)"), ("SiVeR-ST", "SiVeR-TS (all pseudo, median)"),
          ("SiVeR-10+", "SiVeR-10+ (weights + averaging)"),
          ("SiVeR-10+ST", "SiVeR-10+ TS (pseudo-labels)")]


def aggregate(series, how):
    """series: {split: {'r_runs': [...12...], 'mae': x}} -> (r, mae)."""
    runs = [np.asarray(v["r_runs"], float) for v in series.values()]
    mae = float(np.mean([v["mae"] for v in series.values()]))
    if how == "arithmetic":
        return float(np.mean([x.mean() for x in runs])), mae
    z = [np.arctanh(np.clip(x, -0.999999, 0.999999)).mean() for x in runs]
    return float(np.tanh(np.mean(z))), mae


def marks(values, lower_is_better, dp=2):
    """Return {index: 'best'|'second'}, judged on the exact values but marking every row that
    PRINTS the same, so two rows holding the same configuration are not marked differently."""
    ok = [i for i in range(len(values)) if np.isfinite(values[i])]
    if not ok:
        return {}
    order = sorted(ok, key=lambda i: values[i], reverse=not lower_is_better)
    best = _trunc(values[order[0]], dp)
    out = {i: "best" for i in ok if _trunc(values[i], dp) == best}
    rest = [i for i in order if i not in out]
    if rest:
        second = _trunc(values[rest[0]], dp)
        out.update({i: "second" for i in rest if _trunc(values[i], dp) == second})
    return out


def _trunc(x, dp):
    """Cut at dp decimals rather than rounding, so 0.458 prints 0.45 and 0.464 prints 0.46."""
    import math
    sign = -1 if x < 0 else 1
    return sign * math.floor(abs(x) * 10 ** dp) / 10 ** dp


def DP(which):
    """Decimals printed: correlations at 2, MAE in bpm at 1. `which` is 0 for r, 1 for MAE."""
    return 1 if which else 2


def fmt(x, mark, latex, dp=2):
    s = f"{_trunc(x, dp):.{dp}f}"
    if not latex or mark is None:
        return s
    return r"\textbf{%s}" % s if mark == "best" else r"\underline{%s}" % s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", choices=["fisher", "arithmetic"], default="fisher")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX rows as well as the grid")
    ap.add_argument("--out", default=os.path.join(HERE, "tables"))
    ap.add_argument("--store", default="run_correlations.pkl",
                    help="per-run results to build Tables 3 and 5 from")
    args = ap.parse_args()

    store = pickle.load(open(os.path.join(DATA, args.store), "rb"))
    nsplits = len(next(iter(store["reported"].values())))
    print(f"aggregation: {args.agg}   splits: {nsplits}   runs per split: 12\n")

    # ---------------- Table 3 ----------------
    cells = {}
    for lab, data, inst, key in TABLE3:
        cells[lab, inst] = [aggregate(store["reported"][f"{key}|{t}"], args.agg) for t in TIERS]
    grid = [cells[l, i] for l, _, i, _ in TABLE3]
    mk = {}
    for c, (tier, which) in enumerate([(t, w) for t in range(3) for w in (0, 1)]):
        vals = [g[tier][which] for g in grid]
        mk[(tier, which)] = marks(vals, lower_is_better=bool(which), dp=DP(which))
    print("== Table 3: multi-domain results ==")
    print(f"{'Strategy':<30}{'Data':<6}{'seen r':>8}{'MAE':>7}{'unseen r':>10}{'MAE':>7}"
          f"{'comb r':>9}{'MAE':>7}")
    lat3 = []
    for n, (lab, data, inst, key) in enumerate(TABLE3):
        g = grid[n]
        flat = [g[t][w] for t in range(3) for w in (0, 1)]
        print(f"{lab:<30}{data:<6}" + "".join(
            f"{fmt(v, None, False, DP(i % 2)):>{10 if i in (2,4) else (8 if i==0 else 7)}}"
            for i, v in enumerate(flat)))
        lat3.append(f"{lab}\n& {data}\n& {inst}\n& " + " & ".join(
            fmt(g[t][w], mk[(t, w)].get(n), args.latex, DP(w))
            for t in range(3) for w in (0, 1)) + r" \\")

    # ---------------- Table 5 ----------------
    cells5 = []
    for lab, chg, vw, ag, ps, key, src in TABLE5:
        cells5.append([aggregate(store[src][f"{key}|{t}"], args.agg) for t in TIERS])
    mk5 = {}
    for t in range(3):
        for w in (0, 1):
            mk5[(t, w)] = marks([g[t][w] for g in cells5], lower_is_better=bool(w), dp=DP(w))
    print("\n== Table 5: ablations ==")
    print(f"{'Model Variant':<40}{'seen r':>8}{'MAE':>7}{'unseen r':>10}{'MAE':>7}{'comb r':>9}{'MAE':>7}")
    lat5 = []
    for n, (lab, chg, vw, ag, ps, key, src) in enumerate(TABLE5):
        g = cells5[n]
        flat = [g[t][w] for t in range(3) for w in (0, 1)]
        print(f"{lab:<40}" + "".join(
            f"{fmt(v, None, False, DP(i % 2)):>{10 if i in (2,4) else (8 if i==0 else 7)}}"
            for i, v in enumerate(flat)))
        lat5.append(f"{lab}\n& {chg}\n& {vw} & {ag} & {ps}\n& " + " & ".join(
            fmt(g[t][w], mk5[(t, w)].get(n), args.latex, DP(w))
            for t in range(3) for w in (0, 1)) + r" \\")

    # ---------------- Table 4 ----------------
    rows = [r for r in pickle.load(open(f"{DATA}/multiview_preds.pkl", "rb"))
            if r["farm"] == "Farm-B"]
    clip_event = {r["clip"]: (r["event"] or r["clip"]) for r in rows}
    event_clips = collections.defaultdict(set)
    for clip, event in clip_event.items():
        event_clips[event].add(clip)
    paired = {e for e, c in event_clips.items() if len(c) == 2}
    animals = {r["animal"] for r in rows if clip_event[r["clip"]] in paired}
    r_of = lambda p, y: float(pearsonr(p, y)[0]) if len(p) > 2 and np.std(p) > 1e-6 else np.nan

    print("\n== Table 4: paired multi-view on Farm-B ==")
    print(f"   {len(paired)} paired events from {len(animals)} cattle, "
          f"{2*len(paired)} synchronized clips")
    print(f"{'Method':<24}{'Single-view r':>14}{'MAE':>7}{'Paired r':>11}{'MAE':>7}")
    lat4 = []
    res4 = []
    for lab, key in TABLE4:
        per_clip, truth = collections.defaultdict(list), {}
        for r in rows:
            if r["model"] == key:
                per_clip[r["clip"]].append(r["pred_rr"])
                truth[r["clip"]] = r["gt_rr"]
        clip_pred = {c: float(np.mean(v)) for c, v in per_clip.items()}
        sv = sorted(c for c in clip_pred if clip_event[c] in paired)          # the 32 clips
        ev_pred, ev_gt = [], []                                               # the 16 events
        for e in sorted(paired):
            cl = sorted(c for c in event_clips[e] if c in clip_pred)
            if not cl:
                continue
            ev_pred.append(float(np.mean([clip_pred[c] for c in cl])))
            ev_gt.append(truth[cl[0]])
        allc = sorted(clip_pred)                                              # all 37, unreported
        def pair(keys, pred, gt):
            p, y = np.array(pred), np.array(gt)
            return r_of(p, y), float(np.mean(np.abs(p - y)))
        r37, m37 = pair(allc, [clip_pred[c] for c in allc], [truth[c] for c in allc])
        r32, m32 = pair(sv, [clip_pred[c] for c in sv], [truth[c] for c in sv])
        rev, mev = pair(None, ev_pred, ev_gt)
        res4.append((r37, m37, r32, m32, rev, mev))
    mk4 = {c: marks([r[c] for r in res4], lower_is_better=c in (1, 3, 5),
                dp=DP(c % 2)) for c in range(6)}
    for n, (lab, key) in enumerate(TABLE4):
        v = res4[n]
        print(f"{lab:<24}"
              f"{fmt(v[2], None, False, 2):>14}{fmt(v[3], None, False, 1):>7}"
              f"{fmt(v[4], None, False, 2):>11}{fmt(v[5], None, False, 1):>7}")
        lat4.append(f"{lab}\n& " + " & ".join(
            fmt(v[c], mk4[c].get(n), args.latex, DP(c % 2))
            for c in (2, 3, 4, 5)) + r" \\")   # sv(32) + paired: the columns the paper reports

    if args.latex:
        os.makedirs(args.out, exist_ok=True)
        for name, rows in (("table3", lat3), ("table4", lat4), ("table5", lat5)):
            open(f"{args.out}/{name}.tex", "w").write("\n\n".join(rows) + "\n")
        print(f"\nwrote {args.out}/table3.tex, table4.tex, table5.tex")


if __name__ == "__main__":
    main()
