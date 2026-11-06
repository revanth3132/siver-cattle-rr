"""
Stage 3 - ROI refinement (segmentation).
For the selected cow, crops the padded bounding box each frame and runs SAM 2 with an inset
box prompt (re-segmenting every K-th frame and reusing the mask in between), producing a
masked cow-on-black cutout sequence saved as frames.

Usage:
    python stage3_mask_roi.py --clip data/work/clip01
"""
# Requires the source video, which is not redistributed. See ../README.md.

import os, json, glob, argparse, numpy as np, cv2
from ultralytics import SAM
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    args = ap.parse_args()

    tj = json.load(open(os.path.join(args.clip, "tracks.json")))
    target = json.load(open(os.path.join(args.clip, "target.json")))["track_id"]
    boxes = tj["boxes"]; video = tj["video"]; nW, nH = tj["native_res"]
    start_frame = tj.get("start_frame", 0)   # stage 1 may have skipped ahead; match it here
    out_dir = os.path.join(args.clip, "cutout"); os.makedirs(out_dir, exist_ok=True)

    sam = SAM(C.SAM_WEIGHTS)
    cap = cv2.VideoCapture(video)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    last_mask = None; saved = 0
    n = max(int(fi) for fi in boxes)
    for fi in range(n + 1):
        ok, f = cap.read()
        if not ok:
            break
        fb = boxes.get(str(fi), boxes.get(fi, {}))
        if str(target) not in fb:
            continue
        x1, y1, x2, y2 = fb[str(target)]
        bw, bh = x2 - x1, y2 - y1
        X1 = max(0, int(x1 - C.BOX_PAD * bw)); Y1 = max(0, int(y1 - C.BOX_PAD * bh))
        X2 = min(nW, int(x2 + C.BOX_PAD * bw)); Y2 = min(nH, int(y2 + C.BOX_PAD * bh))
        roi = f[Y1:Y2, X1:X2]
        if roi.size == 0:
            continue
        if saved % C.SAM_K == 0:
            h, w = roi.shape[:2]
            inset = [C.SAM_INSET * w, C.SAM_INSET * h, (1 - C.SAM_INSET) * w, (1 - C.SAM_INSET) * h]
            res = sam(roi, bboxes=[inset], verbose=False)
            if res and res[0].masks is not None and len(res[0].masks):
                last_mask = res[0].masks.data[0].cpu().numpy().astype(np.uint8)
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if last_mask is not None and last_mask.shape == g.shape:
            g = (g * (last_mask > 0)).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"frame_{saved:04d}.png"), g)
        saved += 1
        if saved >= C.CUTOUT_MAXF:
            break
    cap.release()
    print(f"[stage3] saved {saved} masked cutout frames to {out_dir}")
    print("[stage3] -> run stage 4 to extract the respiratory signal views.")


if __name__ == "__main__":
    main()
