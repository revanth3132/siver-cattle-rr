"""
siver_lib.py - signal processing for SiVeR.

Turns a masked animal cutout sequence into the ten one-dimensional respiratory signal
views, conditions them, and reduces each to a spectral feature vector. No dependence on
dataset layout; shared by all pipeline stages.

Parameters:
  - cutout 160x120 grayscale, foreground threshold 10
  - respiration band 0.3-2.5 Hz, EVM amplification 20x, 3 pyramid levels, order-4 Butterworth
  - dense Farneback flow (0.5, 3, 15, 3, 5, 1.2, 0)
  - flank ROI rows [0.35, 0.85] x cols [0.20, 0.80] of the cutout
"""
import numpy as np
import cv2
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

CUTOUT_SIZE = (160, 120)   # (w, h)
FG_THRESH   = 10           # foreground threshold (cutouts are animal-on-black)
RESP_LOW, RESP_HIGH = 0.3, 2.5   # respiration band, Hz (18-150 bpm)
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# the ten signal views, in fixed order
SIGNAL_VIEWS = ["evm_flow", "dense_farneback", "frame_diff", "frame_diff_bp", "optical_strain",
                "mask_area", "mask_height", "mask_centroidy", "flank_evm", "flank_frame_diff"]


# ----------------------------------------------------------------------------- EVM
def evm_amplify(grays, fps, lo=RESP_LOW, hi=RESP_HIGH, amp=20.0, levels=3):
    """Eulerian Video Magnification of subtle motion in the respiration band."""
    if len(grays) < 27:
        return grays
    nyq = fps / 2.0
    if lo / nyq >= 1 or hi / nyq >= 1:
        return grays
    pyrs = []
    for f in grays:
        pyr = [f.astype(np.float32)]
        for _ in range(levels):
            f = cv2.pyrDown(f); pyr.append(f.astype(np.float32))
        pyrs.append(pyr)
    out = [[p.copy() for p in pyrs[t]] for t in range(len(grays))]
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    for lev in range(levels + 1):
        stack = np.array([pyrs[t][lev] for t in range(len(grays))])
        if stack.shape[0] < 27:
            continue
        try:
            filt = filtfilt(b, a, stack, axis=0)
            for t in range(len(grays)):
                out[t][lev] = pyrs[t][lev] + amp * filt[t]
        except Exception:
            continue
    return [np.clip(out[t][0], 0, 255).astype(np.uint8) for t in range(len(grays))]


def _bandpass(sig, fps, lo=RESP_LOW, hi=RESP_HIGH, order=4):
    if len(sig) < 27:
        return sig
    try:
        b, a = butter(order, [lo / (fps / 2), hi / (fps / 2)], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def _flank_mask(h, w):
    r = np.zeros((h, w), bool)
    r[int(h * 0.35):int(h * 0.85), int(w * 0.20):int(w * 0.80)] = True
    return r


# ----------------------------------------------------------------------- signal views
def extract_signal_views(gray_frames, fps):
    """Given a list of masked grayscale cutout frames (cow on black), return a dict
    of the ten 1-D respiratory signal views."""
    G = [cv2.resize(g, CUTOUT_SIZE) for g in gray_frames]
    if len(G) < 40:
        return None
    ev = evm_amplify(G, fps)
    h, w = CUTOUT_SIZE[1], CUTOUT_SIZE[0]
    fr = _flank_mask(h, w)
    out = {k: [] for k in SIGNAL_VIEWS}

    # mask-geometry signals (from the raw cutout mask)
    for g in G:
        m = (g > FG_THRESH)
        out["mask_area"].append(float(m.sum()))
        ys = np.where(m.any(1))[0]
        out["mask_height"].append(float(ys.max() - ys.min()) if len(ys) else 0.0)
        yy, _ = np.where(m)
        out["mask_centroidy"].append(float(yy.mean()) if len(yy) else 0.0)

    # motion signals (frame-to-frame)
    for i in range(1, len(ev)):
        me = (ev[i] > FG_THRESH); mf = me & fr; mraw = (G[i] > FG_THRESH)
        fl = cv2.calcOpticalFlowFarneback(ev[i - 1], ev[i], None, **FARNEBACK)
        mag = np.hypot(fl[..., 0], fl[..., 1])
        out["evm_flow"].append(float(mag[me].mean()) if me.sum() > 10 else 0.0)
        out["flank_evm"].append(float(mag[mf].mean()) if mf.sum() > 10 else 0.0)
        strain = np.abs(np.gradient(fl[..., 0], axis=1)) + np.abs(np.gradient(fl[..., 1], axis=0))
        out["optical_strain"].append(float(strain[me].mean()) if me.sum() > 10 else 0.0)
        flr = cv2.calcOpticalFlowFarneback(G[i - 1], G[i], None, **FARNEBACK)
        magr = np.hypot(flr[..., 0], flr[..., 1])
        out["dense_farneback"].append(float(magr[mraw].mean()) if mraw.sum() > 10 else 0.0)
        fd = np.abs(G[i].astype(np.float32) - G[i - 1].astype(np.float32))
        out["frame_diff"].append(float(fd[mraw].mean()) if mraw.sum() > 10 else 0.0)
        out["flank_frame_diff"].append(float(fd[(G[i] > FG_THRESH) & fr].mean())
                                       if ((G[i] > FG_THRESH) & fr).sum() > 10 else 0.0)
    out["frame_diff_bp"] = list(_bandpass(np.array(out["frame_diff"]), fps))
    return {k: np.asarray(v, float) for k, v in out.items()}


# --------------------------------------------------------------- signal conditioning
def condition_signal(x, scale=4.0):
    """Median/MAD centring then tanh compression. Run on every view before the FFT.

    Bounds the influence of transient excursions (steps, head swings, tracking jumps)
    while preserving periodic structure. Robust statistics are used rather than absolute
    amplitude, which varies with animal scale and camera distance.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return np.tanh((x - med) / (scale * mad))


def condition_views(signal_dict, scale=4.0):
    """condition_signal applied to every view of one clip."""
    return {k: condition_signal(v, scale) for k, v in signal_dict.items()}


# --------------------------------------------------------------------- spectral feature
def fft_feature_vector(signal, fps, n_feats=17, resp_low=RESP_LOW, resp_high=RESP_HIGH):
    """17-D spectral descriptor of a 1-D signal over the respiration band."""
    if signal is None or len(signal) < 27:
        return np.zeros(n_feats, dtype=np.float32)
    sig = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    N = len(sig)
    yf = np.abs(fft(sig))[:N // 2]
    xf = fftfreq(N, 1.0 / fps)[:N // 2]
    rmask = (xf >= resp_low) & (xf <= resp_high)
    rf, rp = xf[rmask], yf[rmask]
    if len(rp) < 2:
        return np.zeros(n_feats, dtype=np.float32)
    di = np.argmax(rp)
    pred_rr = float(rf[di]) * 60.0
    peak_p = float(rp[di])
    snr = peak_p / (np.mean(rp) + 1e-9)
    top3 = np.argsort(rp)[-3:][::-1]
    feats = [pred_rr, peak_p, snr, float(np.std(sig)), float(np.max(np.abs(sig))),
             float(np.mean(np.abs(sig))), float(np.median(np.abs(sig))),
             float(np.min(sig)), float(np.max(sig)),
             *[float(rf[i]) for i in top3], *[float(rp[i]) for i in top3],
             float(np.percentile(np.abs(sig), 90)), float(np.percentile(np.abs(sig), 75))]
    return np.array(feats[:n_feats], dtype=np.float32)


def feature_matrix(signal_dict, fps):
    """view -> 17-D FFT descriptor."""
    return {k: fft_feature_vector(signal_dict[k], fps) for k in SIGNAL_VIEWS}


# ------------------------------------------------------------------ shape descriptor
# Index map of the 17-D vector:
#   0 peak bpm    1 peak power   2 SNR          3 std         4 max|x|      5 mean|x|
#   6 median|x|   7 min          8 max          9,10,11 top-3 peak frequencies (Hz)
#   12,13,14 top-3 peak powers   15 p90         16 p75
SHAPE_FEATURE_NAMES = ("peak_bpm", "snr", "peak1_hz", "peak2_hz", "peak3_hz",
                       "power_ratio_2_1", "power_ratio_3_1")


def spectral_shape_descriptor(f17):
    """Reduce the 17-D vector to the seven dimensionless descriptors used for regression.

        peak_bpm          dominant in-band frequency, breaths/min
        snr               dominant peak power / mean in-band power
        peak1_hz          dominant peak, Hz
        peak2_hz          second peak, Hz
        peak3_hz          third peak, Hz
        power_ratio_2_1   second peak power / dominant peak power
        power_ratio_3_1   third peak power / dominant peak power

    Amplitude terms are dropped, so the descriptor is invariant to animal scale, camera
    distance and contrast. peak_bpm and peak1_hz differ only by the 60 s conversion.
    """
    f = np.asarray(f17, dtype=np.float64)
    return np.array([f[0], f[2], f[9], f[10], f[11],
                     f[13] / (f[12] + 1e-9), f[14] / (f[12] + 1e-9)], dtype=np.float32)


def shape_feature_matrix(signal_dict, fps, resp_low=RESP_LOW, resp_high=RESP_HIGH):
    """view -> 7-D shape descriptor."""
    return {k: spectral_shape_descriptor(
                fft_feature_vector(signal_dict[k], fps, resp_low=resp_low, resp_high=resp_high))
            for k in SIGNAL_VIEWS if k in signal_dict}
