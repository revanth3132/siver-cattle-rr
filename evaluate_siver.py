"""Score the fitted models on their training/test configuration.

Checks the models against the predictions recorded when they were fitted, then reports
correlation and MAE by evaluation tier and for single- versus paired-camera inference.

Usage:
    python evaluate_siver.py
"""
import pickle

import numpy as np
from scipy.stats import pearsonr

import config as C
from predict_siver import _load_model, group_events, predict, predict_event

MATCH_TOL = 1e-6


def trunc(x, dp):
    """Truncate to dp decimals."""
    sign = -1.0 if x < 0 else 1.0
    return sign * np.floor(abs(x) * 10 ** dp) / 10 ** dp


def score(pred, gt):
    """Pearson r and MAE in bpm."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    r = float(pearsonr(pred, gt)[0]) if len(pred) > 2 and np.std(pred) > 1e-6 else float("nan")
    return r, float(np.mean(np.abs(pred - gt)))


def report_config(cfg):
    print("Configuration")
    print(f"  trained on   : {' + '.join(sorted(cfg['train_farms']))}"
          f"   ({cfg['n_train_clips']} clips, {cfg['n_train_cows']} animals)")
    print(f"  pool         : {cfg['n_pool']} unlabelled clips")
    print(f"  signal views : {len(cfg['views'])}")
    print(f"  features     : 7-D spectral shape\n")


def report_check(models, F):
    """Compare re-derived predictions against the values recorded at fit time."""
    print("Check against recorded predictions")
    ok = True
    for role, model in models.items():
        ref = model.get("reference_predictions") or {}
        if not ref:
            print(f"  {role:8s} : none stored")
            continue
        worst = max(abs(predict(model, F[c]) - ref[c]) for c in ref if c in F)
        ok &= worst < MATCH_TOL
        print(f"  {role:8s} : {len(ref)} clips, max difference {worst:.2e} bpm  "
              f"{'MATCH' if worst < MATCH_TOL else 'MISMATCH'}")
    print(f"  {'models reproduce the fit' if ok else 'MISMATCH - results below are unreliable'}\n")
    return ok


def report_tiers(models, F, labels, cmeta, train):
    test = [c for c in models["teacher"]["test_clips"] if c in F]
    tiers = {"seen": [c for c in test if cmeta[c]["farm"] in train],
             "unseen": [c for c in test if cmeta[c]["farm"] not in train],
             "combined": test}
    print(f"Held-out scores, {len(test)} clips")
    print(f"  {'model':9s} {'tier':12s} {'n':>4s} {'r':>7s} {'MAE':>7s}")
    for role, model in models.items():
        pred = {c: predict(model, F[c]) for c in test}
        for tier, clips in tiers.items():
            if len(clips) < 3:
                continue
            r, mae = score([pred[c] for c in clips], [labels[c] for c in clips])
            print(f"  {role:9s} {tier:12s} {len(clips):4d} "
                  f"{trunc(r, 2):7.2f} {trunc(mae, 1):7.1f}")


def report_multiview(models, F, labels, cmeta):
    events = {e: c for e, c in group_events(cmeta).items() if all(x in F for x in c)}
    if not events:
        return
    clips = sorted(c for cl in events.values() for c in cl)
    animals = {cmeta[c]["animal"] for c in clips}
    order = sorted(events)
    print(f"\nSingle versus paired camera, {len(events)} events, "
          f"{len(clips)} clips, {len(animals)} animals")
    print(f"  {'model':9s} {'inference':22s} {'n':>4s} {'r':>7s} {'MAE':>7s}")
    for role, model in models.items():
        r1, m1 = score([predict(model, F[c]) for c in clips],
                       [labels[c] for c in clips])
        r2, m2 = score([predict_event(model, [F[c] for c in events[e]]) for e in order],
                       [np.mean([labels[c] for c in events[e]]) for e in order])
        print(f"  {role:9s} {'single-view':22s} {len(clips):4d} "
              f"{trunc(r1, 2):7.2f} {trunc(m1, 1):7.1f}")
        print(f"  {role:9s} {'paired, averaged':22s} {len(order):4d} "
              f"{trunc(r2, 2):7.2f} {trunc(m2, 1):7.1f}")


def main():
    feats = pickle.load(open(C.FEATURES, "rb"))
    F, labels, cmeta = feats["features"], feats["labels"], feats["clip_meta"]
    models = {"teacher": _load_model(C.TEACHER_MODEL),
              "student": _load_model(C.STUDENT_MODEL)}

    cfg = models["teacher"]["trained_on"]
    report_config(cfg)
    report_check(models, F)
    report_tiers(models, F, labels, cmeta, set(cfg["train_farms"]))
    report_multiview(models, F, labels, cmeta)

    print("\nScores are for this configuration only. The reported tables aggregate 120 "
          "animal-disjoint\nsplits x 12 runs and are not computed here.")


if __name__ == "__main__":
    main()
