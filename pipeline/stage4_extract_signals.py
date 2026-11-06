"""
Stage 4 - Respiratory signal extraction.
Loads the masked cutout sequence and computes the ten 1-D signal views (EVM flow, dense flow,
frame differences, optical strain, mask geometry, flank motion) and their 17-D FFT descriptors.

Usage:
    python stage4_extract_signals.py --clip data/work/clip01
"""
# Requires the source video, which is not redistributed. See ../README.md.

import os, glob, json, argparse, numpy as np, cv2
import config as C
import siver_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    args = ap.parse_args()

    cut = os.path.join(args.clip, "cutout")
    fs = sorted(glob.glob(os.path.join(cut, "frame_*.png")))
    grays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in fs]
    grays = [g for g in grays if g is not None]
    if len(grays) < 40:
        raise SystemExit(f"[stage4] too few frames ({len(grays)}) in {cut}; need >= 40.")

    # NOTE ON FRAME RATE: C.FPS is the nominal 30 fps. The reported results use each
    # recorder's measured rate -- in particular C.BARN_FPS = 14.58 for Farm-D -- because the
    # spectrum is read in bpm and a wrong rate rescales every frequency. Per-clip rates are
    # carried in the shipped signal file.
    fps = C.FPS
    views = L.extract_signal_views(grays, fps)
    # Median/MAD centring and tanh compression, applied before the spectra.
    cond  = L.condition_views(views)
    feats = L.feature_matrix(cond, fps)                        # view -> 17-D vector
    shape = L.shape_feature_matrix(cond, fps,                  # view -> the reported 7-D
                                   resp_low=C.RESP_LOW_BPM / 60.0,
                                   resp_high=C.RESP_HIGH_BPM / 60.0)

    out = {"views": {k: v.tolist() for k, v in views.items()},
           "conditioned": {k: v.tolist() for k, v in cond.items()},
           "features": {k: feats[k].tolist() for k in L.SIGNAL_VIEWS},
           "shape_features": {k: shape[k].tolist() for k in shape},
           "shape_feature_names": list(L.SHAPE_FEATURE_NAMES)}
    json.dump(out, open(os.path.join(args.clip, "signals.json"), "w"))
    print(f"[stage4] extracted {len(L.SIGNAL_VIEWS)} signal views from {len(grays)} frames.")
    print(f"[stage4] wrote {args.clip}/signals.json  -> ../predict_siver.py predicts the rate.")


if __name__ == "__main__":
    main()
