"""Predict respiration rate from the per-clip feature vectors.

Usage:
    python predict_siver.py                     all clips, both models
    python predict_siver.py --clip C35_c01      one clip, per-view breakdown
    python predict_siver.py --model student     one model only
    python predict_siver.py --multiview         per event, cameras averaged
"""
import argparse
import pickle

import numpy as np

import config as C


FITTED_WITH = "scikit-learn 1.7.2, numpy 2.2.6, scipy 1.15.3, python 3.10"


def _load_model(path):
    """Unpickle a fitted model, reporting a version mismatch rather than a traceback."""
    try:
        return pickle.load(open(path, "rb"))
    except Exception as exc:
        try:
            from importlib.metadata import version
            have = f"scikit-learn {version('scikit-learn')}, numpy {version('numpy')}"
        except Exception:
            have = "scikit-learn not installed"
        raise SystemExit(
            f"could not load {path}\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"  the models are pickled scikit-learn estimators, fitted with {FITTED_WITH}\n"
            f"  this environment has {have}\n"
            f"  make_tables.py reproduces the reported tables and needs only numpy and scipy"
        ) from None


def load(model="both", features=None):
    feats = pickle.load(open(features or C.FEATURES, "rb"))
    out = {}
    if model in ("teacher", "both"):
        out["teacher"] = _load_model(C.TEACHER_MODEL)
    if model in ("student", "both"):
        out["student"] = _load_model(C.STUDENT_MODEL)
    return feats, out


def predict_per_view(model, clip_features):
    """Per view: scale, average the four regressors, apply the view calibration."""
    out = {}
    for view, part in model["views"].items():
        if view not in clip_features:
            continue
        x = np.asarray(clip_features[view], dtype=np.float64).reshape(1, -1)
        xs = part["scaler"].transform(x)
        p = float(np.mean([m.predict(xs)[0] for m in part["models"]]))
        out[view] = part.get("cal_a", 1.0) * p + part.get("cal_b", 0.0)
    return out


def predict(model, clip_features):
    """Clip rate in bpm: median over the signal views."""
    per_view = predict_per_view(model, clip_features)
    if not per_view:
        return float("nan")
    return float(np.median([per_view[v] for v in model["aggregate_views"] if v in per_view]))


def predict_event(model, clip_features_list):
    """Event rate in bpm: mean of the camera-level estimates."""
    return float(np.mean([predict(model, f) for f in clip_features_list]))


def group_events(clip_meta):
    """event id -> sorted clip ids, for clips belonging to a two-camera event."""
    out = {}
    for clip, meta in clip_meta.items():
        if meta["event"]:
            out.setdefault(meta["event"], []).append(clip)
    return {e: sorted(c) for e, c in out.items()}


def _show_clip(clip, F, labels, cmeta, models):
    meta = cmeta[clip]
    print(f"{clip}   {meta['farm']}   animal {meta['animal']}   "
          f"reference {labels[clip]:.1f} bpm")
    for role, model in models.items():
        per_view = predict_per_view(model, F[clip])
        print(f"\n  {role} ({model['name']})")
        for view in model["aggregate_views"]:
            if view in per_view:
                print(f"    {view:22s} {per_view[view]:6.1f}")
        print(f"    {'median':22s} {predict(model, F[clip]):6.1f} bpm")


def _show_events(F, labels, cmeta, models):
    events = group_events(cmeta)
    print(f"{'event':8s} {'farm':8s} {'animal':8s} {'cams':>4s} {'ref':>6s}"
          + "".join(f" {r:>9s}" for r in models))
    for event in sorted(events):
        clips = events[event]
        meta = cmeta[clips[0]]
        row = (f"{event:8s} {meta['farm']:8s} {meta['animal']:8s} {len(clips):4d} "
               f"{np.mean([labels[c] for c in clips]):6.1f}")
        for model in models.values():
            row += f" {predict_event(model, [F[c] for c in clips]):9.1f}"
        print(row)
    animals = {cmeta[c]["animal"] for clips in events.values() for c in clips}
    print(f"\n{len(events)} events, {len(animals)} animals")


def _show_clips(F, labels, cmeta, models):
    print(f"{'clip':28s} {'farm':8s} {'ref':>6s}"
          + "".join(f" {r:>9s}" for r in models))
    for clip in sorted(F):
        row = f"{clip[:28]:28s} {cmeta[clip]['farm']:8s} {labels[clip]:6.1f}"
        for model in models.values():
            row += f" {predict(model, F[clip]):9.1f}"
        print(row)
    print(f"\n{len(F)} clips, {len(models)} model(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["teacher", "student", "both"], default="both")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--multiview", action="store_true")
    args = ap.parse_args()

    feats, models = load(args.model, args.features)
    F, labels, cmeta = feats["features"], feats["labels"], feats["clip_meta"]

    if args.clip:
        if args.clip not in F:
            raise SystemExit(f"unknown clip {args.clip!r}; {len(F)} available, "
                             f"e.g. {sorted(F)[:3]}")
        _show_clip(args.clip, F, labels, cmeta, models)
    elif args.multiview:
        _show_events(F, labels, cmeta, models)
    else:
        _show_clips(F, labels, cmeta, models)


if __name__ == "__main__":
    main()
