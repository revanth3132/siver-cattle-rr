"""
Stage 1 - Detection + Tracking.
Runs YOLOv8x + BoT-SORT over a video, saving per-frame cow boxes with track IDs to a JSON,
and renders an annotated preview video so the user can read off track IDs in Stage 2.

Usage:
    python stage1_detect_track.py --video data/videos/test/clip01.mp4 --out data/work/clip01
"""
# Requires the source video, which is not redistributed. See ../README.md.

import os, json, argparse, numpy as np, cv2
from ultralytics import YOLO
import config as C

PALETTE = [(60, 76, 231), (72, 201, 60), (231, 159, 40), (40, 198, 231),
           (231, 72, 200), (180, 120, 255), (60, 230, 255), (255, 120, 120)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output folder for this clip")
    ap.add_argument("--max-frames", type=int, default=C.CUTOUT_MAXF)
    ap.add_argument("--start-sec", type=float, default=0.0,
                    help="seconds to skip before tracking (use target_window_sec from labels.csv "
                         "for the full test recordings)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    yolo = YOLO(C.YOLO_WEIGHTS)
    cap = cv2.VideoCapture(args.video)
    nW, nH = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5) or C.FPS
    start_frame = int(args.start_sec * fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    SW, SH = 1280, int(1280 * nH / nW)
    vw = cv2.VideoWriter(os.path.join(args.out, "tracking_preview.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (SW, SH))
    boxes = {}
    for fi in range(args.max_frames):
        ok, f = cap.read()
        if not ok:
            break
        r = yolo.track(f, classes=[C.COW_CLASS], conf=C.DET_CONF, imgsz=C.DET_IMGSZ,
                       tracker=C.TRACKER, persist=True, verbose=False)
        frame_boxes = {}
        im = cv2.resize(f, (SW, SH)); sx, sy = SW / nW, SH / nH
        if r[0].boxes is not None and r[0].boxes.id is not None:
            for xy, tid in zip(r[0].boxes.xyxy.cpu().numpy(), r[0].boxes.id.cpu().numpy()):
                tid = int(tid); frame_boxes[tid] = [float(v) for v in xy]
                c = PALETTE[tid % len(PALETTE)]
                cv2.rectangle(im, (int(xy[0]*sx), int(xy[1]*sy)), (int(xy[2]*sx), int(xy[3]*sy)), c, 2)
                cv2.putText(im, f"ID {tid}", (int(xy[0]*sx), int(xy[1]*sy)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        boxes[fi] = frame_boxes
        vw.write(im)
    cap.release(); vw.release()

    json.dump({"video": os.path.abspath(args.video), "native_res": [nW, nH], "fps": fps,
               "start_frame": start_frame, "boxes": boxes},
              open(os.path.join(args.out, "tracks.json"), "w"))
    # report which IDs persist (helps the user choose in stage 2)
    from collections import Counter
    cnt = Counter(tid for fb in boxes.values() for tid in fb)
    print(f"[stage1] {len(boxes)} frames processed "
          f"(from {args.start_sec:.0f}s, i.e. frame {start_frame}).")
    print(f"[stage1] track IDs and how many frames each appears in: "
          f"{dict(sorted(cnt.items(), key=lambda kv: -kv[1]))}")
    print(f"[stage1] wrote {args.out}/tracks.json and tracking_preview.mp4")
    print(f"[stage1] -> watch the preview, then pass the target ID to stage 2.")


if __name__ == "__main__":
    main()
