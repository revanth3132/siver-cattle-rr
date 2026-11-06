"""
config.py - pipeline parameters and artifact paths.

Paths resolve relative to this file.
"""
import os

ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")

# ---------------------------------------------------------------- front end
COW_CLASS      = 19             # COCO "cow"
DET_CONF       = 0.25           # YOLOv8x detection confidence
DET_IMGSZ      = 1280           # detector input size
TRACKER        = "botsort.yaml" # BoT-SORT association
BOX_PAD        = 0.15           # bbox padding applied before cropping
SAM_INSET      = 0.13           # inset applied to the SAM box prompt
SAM_K          = 8              # re-prompt SAM every K-th frame
CUTOUT_MAXF    = 600            # frames kept per clip

# Public checkpoints, not redistributed: Ultralytics YOLOv8x (COCO) and SAM 2.1 large.
WEIGHTS_DIR    = os.path.join(MODELS, "weights")
YOLO_WEIGHTS   = os.path.join(WEIGHTS_DIR, "yolov8x.pt")
SAM_WEIGHTS    = os.path.join(WEIGHTS_DIR, "sam2.1_l.pt")

# ---------------------------------------------------------------- signals
FPS            = 30.0           # nominal rate; per-clip rates are carried in the data
BARN_FPS       = 14.58          # measured rate of the barn recorder
EVM_AMP        = 20.0           # Eulerian magnification gain
EVM_LEVELS     = 3              # Laplacian pyramid levels
RESP_LOW_BPM   = 18.0           # respiration band, lower edge
RESP_HIGH_BPM  = 150.0          # respiration band, upper edge
SIGNAL_VARIANT = "rs30cond"     # conditioned signal variant

# ---------------------------------------------------------------- features
DESCRIPTOR_DIM = 7              # siver_lib.spectral_shape_descriptor

# ---------------------------------------------------------------- regression
REGRESSORS     = ("knn", "ridge", "svr", "hgb")   # fitted and averaged per view
KNN_K          = 5
RIDGE_ALPHA    = 1.0
SVR_C          = 10.0
SVR_GAMMA      = "scale"
HGB_SEED       = 0
RARITY_ALPHA   = 1.0            # inverse-density RR weighting exponent
RR_BIN_WIDTH   = 10.0           # bpm, bin width for the weights

# ---------------------------------------------------------------- self-training (student)
PSEUDO_RR_MIN  = 20.0           # plausibility window on aligned pseudo-labels
PSEUDO_RR_MAX  = 110.0
CAL_SLOPE_CLIP = (0.2, 3.0)     # bounds on the output calibration slope

# Pseudo-labelled clips carry no separate loss weight; the inverse-density weighting above
# is recomputed over the combined labelled and pseudo-labelled set.
PSEUDO_LOSS_WEIGHT = None

# ---------------------------------------------------------------- protocol
N_SPLITS       = 120            # animal-disjoint splits
RUNS_PER_SPLIT = 12             # two-domain training runs per split
VIEW_FOLDS     = 5              # GroupKFold folds, grouped by animal

# ---------------------------------------------------------------- artifacts
TEACHER_MODEL  = os.path.join(MODELS, "teacher_siver10plus.pkl")
STUDENT_MODEL  = os.path.join(MODELS, "student_siver10plus_st.pkl")
FEATURES       = os.path.join(DATA,   "features.pkl")
