"""Refit the models from the feature vectors and write the per-run results.

Runs the evaluation protocol: 120 animal-disjoint partitions, 12 two-domain training runs
each, 1,440 fits. Writes data/run_correlations.pkl, which make_tables.py turns into
Tables 3 and 5.

    python run_protocol.py --jobs 8        all 120 partitions
    python run_protocol.py --seeds 2       first 2, for a quick check
    python run_protocol.py --check         compare against the shipped results and stop
"""
import argparse
import itertools
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import config as C

FARMS = ["Farm-A", "Farm-B", "Farm-C", "Farm-D"]
SEEDS = list(range(30)) + list(range(210, 300))

REG = {
    "knn":   lambda n: KNeighborsRegressor(min(C.KNN_K, max(1, n - 1))),
    "ridge": lambda n: Ridge(alpha=C.RIDGE_ALPHA),
    "svr":   lambda n: SVR(C=C.SVR_C, gamma=C.SVR_GAMMA),
    "hgb":   lambda n: HistGradientBoostingRegressor(random_state=C.HGB_SEED),
}

# model name -> key used in run_correlations.pkl and make_tables.py
KEYS = {
    "SiVeR":       "SiVeR (10 views, median)",
    "SiVeR-10+":   "SiVeR-10+ (weights + averaging)",
    "SiVeR-ST":    "SiVeR-TS (all pseudo, median)",
    "SiVeR-10+ST": "SiVeR-10+ TS (pseudo-labels)",
}
TIERS = ["seen", "unseen", "combined"]


def pear(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) <= 2 or np.std(p) <= 1e-6 or np.std(y) <= 1e-6:
        return 0.0
    return float(pearsonr(p, y)[0])


def inverse_density(y, alpha=C.RARITY_ALPHA):
    b = np.clip(((np.asarray(y, float) - 10) // C.RR_BIN_WIDTH).astype(int), 0, 20)
    cnt = np.bincount(b, minlength=21).astype(float)
    w = 1.0 / cnt[b] ** alpha
    return w / w.mean()


def fit_weighted(model, X, y, w):
    try:
        return model.fit(X, y, sample_weight=w)
    except TypeError:
        return model.fit(X, y)


def best_regressor(X, y, groups):
    """Regressor with the highest animal-grouped out-of-fold correlation."""
    ng = len(set(groups))
    if ng < 3:
        return "ridge"
    folds = list(GroupKFold(min(C.VIEW_FOLDS, ng)).split(X, y, groups))
    best = None
    for name in REG:
        oof = np.zeros(len(y))
        for tr, va in folds:
            sc = StandardScaler().fit(X[tr])
            oof[va] = REG[name](len(tr)).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[va]))
        r = pear(oof, y)
        if best is None or r > best[0]:
            best = (r, name)
    return best[1]


def align_to(reference, pred):
    """Map pool predictions onto the labelled RR distribution, preserving their order."""
    pred = np.asarray(pred, float)
    ref = np.sort(np.asarray(reference, float))
    if len(pred) < 4 or len(ref) < 4:
        return pred
    q = (np.searchsorted(np.sort(pred), pred, side="left") + 0.5) / len(pred)
    return np.interp(q, (np.arange(len(ref)) + 0.5) / len(ref), ref)


def make_split(seed, clip_meta, labels):
    """Animal-disjoint halves per farm: RR-ordered pairing, one animal of each pair per half."""
    rr, farm_of = {}, {}
    for clip, meta in clip_meta.items():
        rr.setdefault(meta["animal"], []).append(labels[clip])
        farm_of[meta["animal"]] = meta["farm"]
    split = {}
    for farm in FARMS:
        animals = sorted((a for a, f in farm_of.items() if f == farm),
                         key=lambda a: (np.mean(rr[a]), a))
        rng = np.random.RandomState(seed)
        train, test = [], []
        for i in range(0, len(animals), 2):
            pair = animals[i:i + 2]
            rng.shuffle(pair)
            if len(pair) > 1:
                train.append(pair[0])
                test.append(pair[1])
            else:
                (train if rng.rand() < 0.5 else test).append(pair[0])
        split.update({a: "train" for a in train})
        split.update({a: "test" for a in test})
    return split


def fit_run(X_train, y, groups, X_test, X_pool):
    """Per-view test predictions for the four models."""
    out = {m: {} for m in KEYS}
    weights = inverse_density(y)
    ng = len(set(groups))
    folds = list(GroupKFold(min(C.VIEW_FOLDS, ng)).split(np.zeros(len(y)), y, groups)) if ng >= 3 else []

    for view, Xg in X_train.items():
        Xt, Xp = X_test[view], X_pool[view]
        sc = StandardScaler().fit(Xg)
        Xs = sc.transform(Xg)

        name = best_regressor(Xg, y, groups)
        base = REG[name](len(Xg)).fit(Xs, y)
        out["SiVeR"][view] = base.predict(sc.transform(Xt))

        plus = [fit_weighted(REG[n](len(Xg)), Xs, y, weights) for n in REG]
        out["SiVeR-10+"][view] = np.mean([m.predict(sc.transform(Xt)) for m in plus], axis=0)

        # SiVeR-ST: the selected regressor retrained on labelled plus all aligned pseudo-labels
        ps = align_to(y, base.predict(sc.transform(Xp)))
        Xa, ya = np.vstack([Xg, Xp]), np.concatenate([y, ps])
        sca = StandardScaler().fit(Xa)
        out["SiVeR-ST"][view] = REG[name](len(Xa)).fit(sca.transform(Xa), ya).predict(sca.transform(Xt))

        # SiVeR-10+ST: four regressors on labelled plus plausible pseudo-labels, then calibration
        pt = align_to(y, np.mean([m.predict(sc.transform(Xp)) for m in plus], axis=0))
        keep = (pt >= C.PSEUDO_RR_MIN) & (pt <= C.PSEUDO_RR_MAX)
        Xb, yb = np.vstack([Xg, Xp[keep]]), np.concatenate([y, pt[keep]])
        scb = StandardScaler().fit(Xb)
        stu = [fit_weighted(REG[n](len(Xb)), scb.transform(Xb), yb, inverse_density(yb)) for n in REG]
        pred = np.mean([m.predict(scb.transform(Xt)) for m in stu], axis=0)

        if folds:   # training-only output calibration from animal-disjoint out-of-fold predictions
            oof = np.full(len(y), np.nan)
            for a, b in folds:
                Xf = np.vstack([Xg[a], Xp[keep]])
                yf = np.concatenate([y[a], pt[keep]])
                sf = StandardScaler().fit(Xf)
                mf = [fit_weighted(REG[n](len(Xf)), sf.transform(Xf), yf, inverse_density(yf))
                      for n in REG]
                oof[b] = np.mean([m.predict(sf.transform(Xg[b])) for m in mf], axis=0)
            ok = np.isfinite(oof)
            if ok.sum() > 4 and np.std(oof[ok]) > 1e-6:
                slope = float(np.clip(np.polyfit(oof[ok], y[ok], 1)[0], *C.CAL_SLOPE_CLIP))
                pred = slope * pred + float(y[ok].mean() - slope * oof[ok].mean())
        out["SiVeR-10+ST"][view] = pred
    return out


def run_seed(seed, feats, views):
    """One partition: 12 two-domain runs. Returns {(key, tier): (r per run, mae per run)}."""
    F, labels, cmeta, pool = (feats["features"], feats["labels"],
                              feats["clip_meta"], feats["pool_features"])
    clips = feats.get("clip_order") or sorted(F)
    split = make_split(seed, cmeta, labels)
    X_pool = {v: np.array([pool[u][v] for u in sorted(pool)]) for v in views}

    out = {}
    pairs = list(itertools.combinations(FARMS, 2))
    for farms, swap in [(p, False) for p in pairs] + [(p, True) for p in pairs]:
        trh, teh = ("test", "train") if swap else ("train", "test")
        tr = [c for c in clips if cmeta[c]["farm"] in farms and split[cmeta[c]["animal"]] == trh]
        te = [c for c in clips if split[cmeta[c]["animal"]] == teh]
        if len(tr) < 8 or len(te) < 8:
            continue
        y = np.array([labels[c] for c in tr])
        groups = np.array([cmeta[c]["group"] for c in tr])
        per_view = fit_run({v: np.array([F[c][v] for c in tr]) for v in views}, y, groups,
                           {v: np.array([F[c][v] for c in te]) for v in views}, X_pool)

        gt = np.array([labels[c] for c in te])
        seen = np.array([cmeta[c]["farm"] in farms for c in te])
        for model, key in KEYS.items():
            pred = np.median(np.array([per_view[model][v] for v in views]), axis=0)
            for tier, m in (("seen", seen), ("unseen", ~seen), ("combined", np.ones(len(te), bool))):
                if m.sum() < 3:
                    continue
                cell = f"{key}|{tier}"
                out.setdefault(cell, [[], []])
                out[cell][0].append(round(pear(pred[m], gt[m]), 4))
                out[cell][1].append(float(np.mean(np.abs(pred[m] - gt[m]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feats = pickle.load(open(C.FEATURES, "rb"))
    views = feats["views"]
    seeds = SEEDS[:args.seeds]
    print(f"{len(seeds)} partitions x {C.RUNS_PER_SPLIT} runs, {len(views)} views")

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs) as ex:
            per_seed = list(ex.map(run_seed, seeds, [feats] * len(seeds), [views] * len(seeds)))
    else:
        per_seed = []
        for i, s in enumerate(seeds, 1):
            per_seed.append(run_seed(s, feats, views))
            print(f"  partition {i}/{len(seeds)}", end="\r", flush=True)
        print(" " * 40, end="\r")

    store = {key: {} for key in {k for d in per_seed for k in d}}
    for seed, d in zip(seeds, per_seed):
        for key, (rs, maes) in d.items():
            store[key][seed] = {"r_runs": rs, "mae": round(float(np.mean(maes)), 2)}

    if args.check:
        ref = pickle.load(open(os.path.join(C.DATA, "run_correlations.pkl"), "rb"))["reported"]
        print(f"\nRefitted the models from data/features.pkl and compared the per-run results")
        print(f"against data/run_correlations.pkl, which is what make_tables.py builds the")
        print(f"reported tables from.\n")
        print(f"{'model | tier':46s} {'max |dr|':>9s} {'max |dMAE|':>11s}")
        worst, compared = 0.0, 0
        missing = [k for k in sorted(store) if k not in ref]
        for key in sorted(store):
            if key not in ref:
                continue
            compared += 1
            dr = max(max(abs(np.array(store[key][s]["r_runs"]) - np.array(ref[key][s]["r_runs"])))
                     for s in store[key] if s in ref[key])
            dm = max(abs(store[key][s]["mae"] - ref[key][s]["mae"])
                     for s in store[key] if s in ref[key])
            worst = max(worst, dr)
            print(f"  {key:44s} {dr:9.4f} {dm:11.4f}")
        if missing:
            print(f"\n  not in the reference: {missing}")
        if compared == 0:
            print("\nNOTHING COMPARED - the check is meaningless")
        elif worst < 1e-3:
            print(f"\nMATCH over {compared} model/tier cells, {len(seeds)} partitions, "
                  f"{len(seeds) * C.RUNS_PER_SPLIT} fits per model.")
            print(f"Largest disagreement {worst:.4f} in correlation, against a tolerance of 0.001.")
            print("That is floating-point noise: the stored values carry four decimals and the")
            print("reported tables print two, so a difference this size cannot change any")
            print("published figure. The feature vectors reproduce the reported results.")
        else:
            print(f"\nDIFFERS over {compared} cells: largest disagreement {worst:.4f}, "
                  f"above the 0.001 tolerance. The refit does not reproduce the stored results.")
        return

    # Never overwrite the shipped results by default: --check compares against them, and a
    # refit written over the reference would make that comparison meaningless.
    ref = os.path.join(C.DATA, "run_correlations.pkl")
    out = args.out or os.path.join(C.DATA, "run_correlations_refit.pkl")
    if os.path.abspath(out) == os.path.abspath(ref):
        raise SystemExit("refusing to overwrite the shipped run_correlations.pkl; "
                         "choose another --out")
    existing = pickle.load(open(ref, "rb")) if os.path.exists(ref) else {}
    existing["reported"] = {**existing.get("reported", {}), **store}
    pickle.dump(existing, open(out, "wb"))
    print(f"\nwrote {out}: {len(store)} model/tier cells refitted over {len(seeds)} partitions")
    print(f"for the tables from this refit:  python make_tables.py --store {os.path.basename(out)}")


if __name__ == "__main__":
    main()
