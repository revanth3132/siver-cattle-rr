# Signal-Aware Cattle Respiration Rate Estimation from Commodity Video


The source recordings are subject to data-use and institutional constraints and are not
included. This package contains the pipeline code, the spectral feature vectors derived from
the respiratory signals, and the two fitted models.

## Install and run

    python -m venv siver-env && source siver-env/bin/activate
    pip install -r requirements.txt

Install into a fresh environment rather than an existing one: the models are pickled
scikit-learn estimators, so the version matters, and installing over a working setup can
break it. `make_tables.py` needs only numpy and scipy and runs without installing anything
if those are already present.

    python make_tables.py                      Tables 3, 4 and 5 from the paper
    python predict_siver.py --clip C35_c01     one clip, per-view breakdown
    python predict_siver.py                    all clips, both models
    python predict_siver.py --multiview        per event, cameras averaged
    python evaluate_siver.py                   correlation and MAE

    python run_protocol.py --jobs 8            refit all 1,440 models from the features

## Layout

    config.py                   pipeline parameters and paths
    siver_lib.py                signal views, conditioning, spectral descriptors
    run_protocol.py             refit the models from the features
    make_tables.py              Tables 3, 4, 5
    predict_siver.py            feature vectors + model -> bpm
    evaluate_siver.py           scores and verification
    pipeline/                   stages 1-4, require the source video
    models/                     fitted teacher and student
    data/features.pkl           feature vectors
    data/run_correlations.pkl   per-run results behind Tables 3 and 5
    data/multiview_preds.pkl    per-clip predictions behind Table 4

## Feature vectors

`data/features.pkl` holds, for each clip, the seven spectral descriptors of each of the ten
signal views. The descriptors are defined in `siver_lib.spectral_shape_descriptor`:

| name | definition |
|---|---|
| `peak_bpm` | dominant frequency in the 18-150 bpm band, breaths/min |
| `snr` | dominant peak power / mean in-band power |
| `peak1_hz` | dominant peak, Hz |
| `peak2_hz` | second peak, Hz |
| `peak3_hz` | third peak, Hz |
| `power_ratio_2_1` | second peak power / dominant peak power |
| `power_ratio_3_1` | third peak power / dominant peak power |

`siver_lib.fft_feature_vector` returns a 17-D intermediate that includes amplitude terms; the
seven above are taken from it by `spectral_shape_descriptor`.

Keys in the file:

    features         111 clips -> {view: 7-D vector}
    pool_features     86 auxiliary clips -> {view: 7-D vector}
    labels           reference respiration rate per clip
    clip_meta        per clip: farm, animal, split, posture, event, camera, group
    pool_farm        auxiliary site
    fps              per-clip frame rate
    clip_order       the clip ordering the protocol iterates in
    views, top4, descriptor, descriptor_names

`group` is the animal key `run_protocol.py` passes to `GroupKFold`. Its ordering is part of the
protocol: `GroupKFold` breaks ties by group label, so substituting a different animal key
changes the fold assignment and the results. `event` and `camera` carry the two-camera
pairing, which the clip ids do not encode.

Identifiers are anonymised. A clip id is `{farm letter}{animal number}_c{clip number}`, so
`A01_c01` is the first clip of the first animal of Farm-A; auxiliary clips are `AUX_001`
onward.

## Models

| file | model | prediction |
|---|---|---|
| `models/teacher_siver10plus.pkl` | SiVeR-10+ | per view, scale the descriptor, average four regressors fitted with inverse-density RR weights; median over the ten views |
| `models/student_siver10plus_st.pkl` | SiVeR-10+ST | teacher pseudo-labels the auxiliary clips, regressors refit on labelled plus aligned pseudo-labelled clips, per-view calibration `a*pred + b` from training-animal out-of-fold predictions; median over the ten views |

Each file is a dict with `name`, `role`, `how`, `views`, `aggregate_views`, `descriptor`,
`descriptor_names`, `regressors`, `trained_on`, `test_clips`, `reference_predictions` and
`schema`. It loads without this codebase:

```python
import pickle
m = pickle.load(open("models/student_siver10plus_st.pkl", "rb"))
print(m["name"], m["trained_on"])
print(m["views"]["flank_evm"]["scaler"], m["views"]["flank_evm"]["models"])
```

`reference_predictions` holds the predictions recorded when the models were fitted.
`evaluate_siver.py` re-derives them from the feature vectors and compares; it reports MATCH
at 8.3e-7 bpm, the float32 storage precision of the features.

Both models are fitted on the single configuration named in `trained_on`, so the scores
`evaluate_siver.py` prints are for that configuration, not the aggregated table values.

## Multi-view

Some events were recorded by two synchronised cameras. `clip_meta` carries an `event` id and a
`camera` index for those clips. Each camera is processed independently by the same single-view
model and the two estimates are averaged; there is no multi-view training or cross-camera
correspondence.

The data contain 23 such events from 14 animals: 7 at Farm-A, 16 at Farm-B.

## Pipeline

`pipeline/` contains the four stages that produce the respiratory signals from video:
detection and tracking, target selection, segmentation, and signal extraction. They require
the source recordings and the public YOLOv8x and SAM 2.1 checkpoints.

## Tables

`make_tables.py` regenerates Tables 3, 4 and 5 as printed in the paper, from
`data/run_correlations.pkl`.

`run_protocol.py` regenerates that file from the feature vectors. It runs the evaluation
protocol in full: 120 animal-disjoint partitions, 12 two-domain training runs each, 1,440
model fits, about nine minutes on eight cores. Running it and then `make_tables.py` gives the
same tables, cell for cell.

    python run_protocol.py --seeds 2 --check    verify two partitions against the shipped file
    python run_protocol.py --jobs 8             refit all 120
    python make_tables.py

`--check` reports agreement to 0.001 in correlation and 0.01 bpm in MAE, the precision those
values are stored at. The printed tables are identical either way.

`run_protocol.py` refits the four proposed models. The baseline rows of Tables 3 and 5 depend
on the pretrained image and video embeddings and on the raw signals, neither of which is
included, so those cells come from the stored file.

Correlations are pooled with Fisher's z, MAE arithmetically; `--agg arithmetic` shows the
alternative.

Table 4 is computed directly from the per-clip predictions in `data/multiview_preds.pkl`, over
the 16 two-camera events.
