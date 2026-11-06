# Pipeline stages

These produce the respiratory signals from raw video. They require the source recordings,
which are not included, and stages 1 and 3 require the public YOLOv8x and SAM 2.1 checkpoints.
Everything downstream is supplied as data; see [../README.md](../README.md).

| Stage | Script | Function |
|---|---|---|
| 1 | `stage1_detect_track.py` | YOLOv8x detection, BoT-SORT association, preview video |
| 2 | `stage2_select_animal.py` | records the target track id (manual step) |
| 3 | `stage3_mask_roi.py` | SAM 2.1 box-prompt segmentation, background suppression |
| 4 | `stage4_extract_signals.py` | ten signal views, conditioning, spectral descriptors |

