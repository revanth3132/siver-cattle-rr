"""
Stage 2 - Select the target animal (the one manual step in the pipeline).
After watching tracking_preview.mp4 from Stage 1, the user passes the track ID of the animal
to estimate. This writes a small file recording the choice.

Usage:
    python stage2_select_animal.py --clip data/work/clip01 --track-id 3
"""
# Requires the source video, which is not redistributed. See ../README.md.

import os, json, argparse
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="clip work folder from stage 1")
    ap.add_argument("--track-id", type=int, required=True, help="target cow's track ID (read from the preview)")
    args = ap.parse_args()

    tj = json.load(open(os.path.join(args.clip, "tracks.json")))
    boxes = tj["boxes"]
    present = [int(fi) for fi, fb in boxes.items() if str(args.track_id) in fb]
    if not present:
        ids = sorted({tid for fb in boxes.values() for tid in fb})
        raise SystemExit(f"[stage2] track ID {args.track_id} not found. Available IDs: {ids}")

    json.dump({"track_id": args.track_id, "n_frames": len(present)},
              open(os.path.join(args.clip, "target.json"), "w"))
    print(f"[stage2] target cow set to track ID {args.track_id} "
          f"({len(present)} frames). Wrote {args.clip}/target.json")
    print("[stage2] -> run stage 3 to segment this cow.")


if __name__ == "__main__":
    main()
