# Astronomaly Technical Reference & BLUSE Application Guide

**Purpose:** dense, agent-oriented context on Astronomaly's algorithms, code API, and the concrete steps to apply it to subsets of BLUSE data.

**Primary sources:**
- **[L&B21]** Lochner & Bassett, *"Astronomaly: Personalised Active Anomaly Detection in Astronomical Data"*, Astronomy & Computing (2021), arXiv:2010.11202v2 — `papers/astronomaly_2020.pdf` (26 pp.)
- **[L&R24]** Lochner & Rudnick, *"Astronomaly: Protégé — Discovery Through Human-Machine Collaboration"*, ApJ (2024), arXiv:2411.04188v3 — `papers/astronomaly_2024.pdf` (36 pp.)
- **Code:** `~/software/astronomaly` @ `111cdd1`, `__version__ = "2.0"`, origin `github.com/MichelleLochner/astronomaly`. All API details below were read from this checkout, not from the papers.
- **BLUSE context:** `papers/BLUSE-technical-reference.md`

---

## 1. What Astronomaly is

A framework for **ranking a dataset by how likely each object is to interest a specific human**, using a small number of interactive labels. Python backend (usable standalone as a library) + JavaScript/Flask frontend for labelling.

It is **not** a classifier, and it makes no claim to completeness. Both papers invoke the No Free Lunch theorem explicitly: the feature extractor bounds the class of anomalies that are detectable at all.

### The two operating modes

| | **Classic Astronomaly** [L&B21] | **Protégé** [L&R24] |
|---|---|---|
| Pipeline | features → anomaly detector → active-learning reweighting | features → PCA → GP active regression (**no anomaly detector**) |
| Regressor | Random Forest (`n_estimators=200` in code) | Gaussian Process, `Matern() + WhiteKernel()` |
| Rank column | `trained_score` | `acquisition` |
| Initial sample | top-N by anomaly score | 10 sources equally spaced along PC1 |
| Label scale | 0–5 | 1–5 |
| Best for | hand-crafted low-dim features; anomalies at the *edge* of feature space | deep/self-supervised features; interesting objects *buried inside* feature space |
| Code | `anomaly_detection/human_loop_learning.py` | `anomaly_detection/protege.py` |

**The central 2024 finding:** with deep (BYOL) features, interesting sources do **not** lie on the boundary of feature space — deep losses have no term encouraging separation of similar groups. Isolation Forest and LOF therefore fail on them. Reframing as regression-only (Protégé) fixes this.

---

## 2. Algorithms

### 2.1 Classic active-learning score [L&B21 §2.2, Eqs. 1–3]

```
Ŝ = S · tanh( δ − 1 + arctanh(Ũ) )                        (1)
Ũ = ε₁ + ε₂ · (U / U_max)                                  (2)   ε₁=0.1, ε₂=0.85, U_max=5
δ = exp( α · d / d₀ )                                      (3)
```
- `S` — raw anomaly score, normalised to [0, 5] by `ScoreConverter`
- `U` — user relevance score (predicted for unlabelled objects by RF regression)
- `d` — Euclidean distance in feature space to nearest human-labelled object (`scipy.spatial.cKDTree`)
- `d₀` — mean nearest-labelled-neighbour distance over the dataset
- `α` — trust in predicted user scores vs. raw anomaly score. **`α = 1` in [L&B21]; `α = 0.1` in [L&R24]** (they found user scores more informative than iForest scores). Code default `alpha=1`.

Limiting behaviour (this is the design intent):
- `δ ≫ 1` (far from any label) → `tanh → 1` → `Ŝ ≈ S`. Novel classes are never suppressed for lack of labels.
- `d = 0` (labelled object) → `δ = 1` → `Ŝ = S · Ũ`. Anomaly score weighted by relevance.
- Low `U`, small `d` → `Ŝ` strongly suppressed. Rejected artefacts stay rejected.

`ε₁`, `ε₂` exist purely for numerical stability (`arctanh` diverges at 1). `tanh` is not special — a sigmoid would do.

**Code note:** `NeighbourScore.anom_func` implements `f_u = min_score + 0.85*(user_score/max_score)`, i.e. `ε₁` is bound to the `min_score` kwarg (default `0.1`).

### 2.2 Protégé [L&R24 §6.1]

No closed-form score. The procedure:

1. **Initial query:** sort by first PCA component, take 10 equally spaced sources (`utils.pca_based_initial_selection`). [L&R24] found this beats the random initialisation used by Walmsley et al. (2022b) — higher performing and more reproducible across repeats.
2. **Human scores** them 1–5 in the frontend.
3. **GP regression** on labelled points predicts `(mean, std)` for the entire dataset.
4. **Expected Improvement** acquisition (Mockus & Mockus 1991; Jones et al. 1998):
   ```
   z   = (mean − max_val − ε) / std
   EI  = (mean − max_val − ε)·Φ(z) + std·φ(z)
   ```
   `max_val` = highest user score seen so far; `ε` = `ei_tradeoff`. Low ε → exploitation; high ε → exploration.
5. **Next query** = the 10 highest-EI sources.
6. Repeat until the top of the list is dominated by interesting sources. Final ranking = GP predicted score.

**ε (`ei_tradeoff`):** paper uses **3** (midpoint of the 1–5 score range), deliberately exploratory. **Code default is `0.5`** — pass `ei_tradeoff=3` explicitly to match the paper. Appendix C.4: ε below ~0.5 gives much higher variance; above that, all values are within the noise.

**Batch size of 10** per query: [L&R24] found it a good balance between GP training data and update responsiveness.

**Cold start:** if no labels exist yet, `GaussianProcess.update` returns `scores = acq = 1` for everything rather than erroring.

### 2.3 Anomaly detectors (classic path only)

- **Isolation Forest** (Liu et al. 2008) — random splits isolate points; shorter decision paths = more anomalous. Robust in high dimensions with redundant features. `sklearn`, `bootstrap=True` in code, `n_estimators` default 200 (papers used 100 and 200 in different places).
- **Local Outlier Factor** (Breunig et al. 2000) — density-based via reachability distance. Better when anomalies sit *close* to the normal population. `n_neighbors` default 50 in code; 100 in [L&B21] simulations; 50 in [L&R24].
- Both output **lower = more anomalous**, hence `ScoreConverter(lower_is_weirder=True)`.

### 2.4 Feature extractors in the code

| Module | Class | Notes |
|---|---|---|
| `shape_features.py` | `EllipseFitFeatures` | The [L&B21] ellipse method. 21 features. |
| `shape_features.py` | `HuMomentsFeatures` | Image moments. |
| `power_spectrum.py` | `PSD_Features` | 2D FFT power spectral density, binned. |
| `wavelet_features.py` | `WaveletFeatures` | Stationary wavelet decomposition (`pywt`), default `sym2`, level 2. |
| `flux_histogram.py` | `FluxHistogramFeatures` | Pixel-value histogram. |
| `flatten_features.py` | `Flatten_Features` | Raw pixels flattened. |
| `pretrained_cnn.py` | `CNN_Features` | Pretrained network embeddings. |
| `autoencoder.py` | `AutoencoderFeatures` | Explored in [L&B21], not recommended there. |
| `byol_features.py` | `BYOL_Features` | Self-supervised; the [L&R24] method. |
| `feets_features.py` | `Feets_Features` | Light-curve features (`feets` package). |

#### Ellipse-fitting features [L&B21 Appendix B]
1. Sigma-clip the image (astropy), 3σ.
2. OpenCV contour at 3σ enclosing the image centre → cut out central source, removing noise and neighbours.
3. Contours at brightness percentiles **90, 80, 70, 60, 50, 0**.
4. Fit an ellipse to each contour (OpenCV).
5. Rotation-invariant parameters per ellipse, all *relative to the 90th-percentile (innermost) ellipse*:
   - **Residual** — sum of differences between fitted ellipse and contour
   - **Offset** — Euclidean distance between ellipse centre and the 90th-percentile ellipse centre
   - **Aspect** — major/minor axis ratio ÷ that of the 90th-percentile ellipse
   - **Theta** — |rotation angle − 90th-percentile rotation angle|, degrees
6. **21 features** (only `residual` for the 90th-percentile ellipse, since everything else is relative to it).
7. Fails on ~1% of data (usually a source so bright the central object falls below 3σ); those objects are excluded.

Intuition: boring galaxies → concentric, aligned, similar ellipses. Interesting ones → misaligned, offset, differing aspect ratios, poor fits. Sensitive to **morphology**, blind to **colour**.

#### BYOL [L&R24 §4]
Self-supervised. Two identical networks (online + target) see two different augmentations of the same image; the online net is trained to predict the target's representation. Loss = MSE between online and target representations. The **second-to-last layer (`avgpool`)** of the online network becomes the features.

| Setting | Value used in [L&R24] |
|---|---|
| Architecture | EfficientNet-B0, **initialised with ImageNet weights** (not random — much faster convergence, better performance) |
| Feature dim | 1280 → PCA to **52** at 95% explained variance |
| Optimiser | Adam |
| Batch size | 32 (better recall₁₀₀ than 64/128 in the end-to-end test, though 64 gave slightly better validation loss) |
| Base LR | 0.0005, scaled by `batch_size/256` |
| Epochs | 100 (sufficient; see their Fig. 23) |
| Packages | `pytorch`, `byol-pytorch`, `kornia` |
| Training time | ~10 min for 1031 sources on an RTX 3060 |

**Augmentations [L&R24 Table 4]** (`kornia.augmentation`):

| Augmentation | Probability | Hyperparameters |
|---|---|---|
| `RandomRotation` | 1.0 | 0°–360° |
| `RandomHorizontalFlip` | 0.5 | — |
| `RandomVerticalFlip` | 0.5 | — |
| `CenterCrop` | 1.0 | 110 px |
| `RandomResizedCrop` | 1.0 | 80–100% of image |
| `RandomGaussianBlur` | 0.1 | kernel 15 px, σ 10–15 px |
| `ColorJiggle` | 0.8 | all parameters 0.5 |

Note the tension they report: removing `ColorJiggle` **reduced BYOL loss by a factor of 3** but **worsened** recall₁₀₀ (36 vs 39). They kept it. Loss is not a reliable proxy for downstream performance at fine granularity, though it correlates broadly (their Fig. 22).

---

## 3. Evaluation metrics

**Rank Weighted Score** [L&B21 §4, Eqs. 4–5] — order matters in anomaly detection, so weight early ranks more:
```
S_RWS = (1/S₀) Σᵢ₌₁ᴺ wᵢ Iᵢ ,   wᵢ = N + 1 − i ,   S₀ = N(N+1)/2
```
`Iᵢ = 1` if object `i` is a true anomaly. Range 0 → 1.

**Recall@N** — how many true anomalies appear in the top N. [L&R24] uses **recall₁₀₀** throughout as its primary metric.

Standard metrics (ROC AUC, log loss) are explicitly discouraged: with highly unbalanced classes they are dominated by the majority class.

### Benchmark results

[L&B21] Galaxy Zoo: 61,578 galaxies, 924 ground-truth anomalies (Class 6.1 "odd" probability > 0.9), ellipse features + iForest, 200 labels. Active learning **roughly doubles** anomalies found in the first 100 objects. ~2/3 of detected anomalies were mergers; 60% of catalogued mergers appear in the first 10% of the ranked list.

[L&R24] MGCLS evaluation subset: 1031 sources, 86 scored 4–5 (8.34%), **top 100 viewed**:

| Method | Found |
|---|---|
| Random | 8 |
| Complexity (pyBDSF Gaussian count) | 20 |
| Ellipse features + iForest + NeighbourScore | 32 |
| **BYOL (full 6161) + Protégé** | **48** |
| BYOL features + NeighbourScore + iForest | 24 |
| BYOL features + NeighbourScore + LOF | 14 |

**Read the last two rows carefully:** deep features fed into the *classic* anomaly-detection path perform **worse than hand-crafted features**. Feature extractor and downstream algorithm must be matched.

Inter-rater agreement between the two authors on 100 sources: Pearson **0.70**, 12% of scores differing by more than one point.

---

## 4. Code API reference

### 4.1 The `run_pipeline()` contract

Any script passed to `run_server.py` must define a module-level `run_pipeline()` returning a dict with **exactly these five keys**:

```text
{
  'dataset':         <astronomaly Dataset subclass>,
  'features':        <pd.DataFrame, index = object id (str)>,
  'anomaly_scores':  <pd.DataFrame with a 'score' column, same index>,
  'visualisation':   <pd.DataFrame with 2 columns (t-SNE/UMAP)>,
  'active_learning': <PipelineStage implementing .run() and .combine_data_frames()>,
}
```

Launch:
```bash
python astronomaly/frontend/run_server.py <path/to/your_script>.py
# → http://127.0.0.1:5000/
```

### 4.2 `PipelineStage` base (`base/base_pipeline.py`)

Every stage accepts these kwargs:

| kwarg | Default | Meaning |
|---|---|---|
| `output_dir` | `'./'` | Where outputs, logs and caches go |
| `force_rerun` | `False` | Bypass the checksum cache |
| `save_output` | `True` | Write intermediate outputs to disk |
| `file_format` | parquet | Intermediate format |
| `drop_nans` | `True` | Drop NaN rows before the function runs |

Caching is by **checksum of the input args** (`logging_tools.check_if_inputs_same`). Every instantiation is logged to `astronomaly.log` in `output_dir`.

> **Gotcha:** the cache keys on *arguments*, not data content. If you swap the data under an unchanged argument set, you may silently get stale results. Use `force_rerun=True` while iterating.

### 4.3 Data readers (`data_management/`)

All `Dataset` subclasses accept `filename=`, `directory=`, `list_of_files=`, `output_dir=`.

**`raw_features.RawFeatures`** — *the most likely entry point for BLUSE tabular data.*
- Reads `.npy`, `.csv`, `.parquet`.
- **Dispatch is by filename substring:** if `'labels'` appears in the file path it is loaded as a `labels` DataFrame; otherwise as features. Multiple files are `pd.concat`'d.
- Index is forced to `str`.
- Sets `data_type = 'raw_features'`; frontend displays each object as a feature-vs-index plot.

> **Gotcha:** any BLUSE path containing the substring `labels` (e.g. `.../labels_run3/hits.csv`) will be silently misread as a label file. Keep paths clean or pass explicit `list_of_files`.

**`image_reader.ImageThumbnailsDataset`** — pre-cut thumbnails (png/jpg/fits). Key kwargs: `directory`, `transform_function` (list, applied in order), `display_transform_function` (separate transforms for what the *human* sees), `display_image_size`, `display_interpolation`, `fits_format`, `catalogue`, `additional_metadata`, `check_corrupt_data`. Index = filename stem.

**`image_reader.ImageDataset`** — large FITS images + a source catalogue; extracts cutouts on demand (avoids loading everything into memory).

**`light_curve_reader.LightCurveDataset`** — 1D time series (`data_type = 'light_curve'`).

### 4.4 Preprocessing (`preprocessing/image_preprocessing.py`)

Composable functions, passed as an ordered list to a Dataset's `transform_function`:

```
image_transform_log            image_transform_inverse_sinh      image_transform_root
image_transform_scale          image_transform_zscale            image_transform_resize
image_transform_crop           image_transform_cv2_resize        image_transform_gaussian_window
image_transform_sigma_clipping (sigma=3, central=True)           image_transform_greyscale
image_transform_remove_negatives   image_transform_sum_channels  image_transform_band_reorder
image_transform_colour_correction  image_transform_axis_shift
```

### 4.5 Post-processing / dimensionality reduction

```python
postprocessing.scaling.FeatureScaler()                      # zero mean, unit variance
dimensionality_reduction.pca.PCA_Decomposer(n_components=0, threshold=0)
dimensionality_reduction.truncated_svd.Truncated_SVD_Decomposer(...)
```
`threshold` = explained-variance fraction (alternative to `n_components`). **[L&R24] used 0.95; the shipped example uses 0.85.** Their Appendix C found the number of PCA components had surprisingly little impact (recall₁₀₀ 41 vs 44).

### 4.6 Anomaly detection & active learning

```text
isolation_forest.IforestAlgorithm(n_estimators=200, **stage_kwargs)
lof.LOF_Algorithm(n_neighbors=50, **stage_kwargs)

human_loop_learning.ScoreConverter(
    lower_is_weirder=True, new_min=0, new_max=5,
    convert_integer=False, column_name='score')     # column_name='all' converts every column

human_loop_learning.NeighbourScore(
    min_score=0.1, max_score=5, alpha=1,
    regression_algorithm='RF',                      # 'RF' or 'GP'
    column_to_sort_by='trained_score',
    show_unlabelled_first=True)

protege.GaussianProcess(
    features,                                       # positional; the feature DataFrame
    ei_tradeoff=0.5,                                # PASS 3 to match the paper
    column_to_sort_by='acquisition',
    show_unlabelled_first=True)

utils.utils.pca_based_initial_selection(features, N)  # -> DataFrame with 'score'; N sources = 5, rest 0
```

`NeighbourScore` outputs columns `['predicted_user_score', 'trained_score']`, plus `'acquisition'` when `regression_algorithm='GP'`.
`protege.GaussianProcess` outputs `['score', 'trained_score', 'acquisition']` (score duplicated into `trained_score` so the frontend treats it as post-human).

> **Verified code gotcha — feature leakage in the classic path.** `NeighbourScore.compute_nearest_neighbour` and `.train_regression` both build their feature matrix as `features_with_labels.drop(columns=['human_label', 'score'])`. On the **second and subsequent** retrains, `anomaly_scores` also carries `predicted_user_score` and `trained_score`, which are concatenated into `features_with_labels` by `combine_data_frames` and are therefore **not dropped** — they leak into both the KDTree distance and the RF regression. `protege.GaussianProcess.update` does not have this problem: it filters against the full list `['human_label','score','trained_score','acquisition']`. Observed by reading the source; not empirically tested. If you use the classic path iteratively, either verify this or prefer Protégé.

### 4.7 Visualisation

```text
visualisation.tsne_plot.TSNE_Plot(perplexity=50, ...)
visualisation.umap_plot.UMAP_Plot(min_dist=0.1, n_neighbors=15, max_samples=2000, metric='euclidean', shuffle=False)
```
[L&R24] used `n_neighbors=15`, `min_dist=0.01`. **Both are visualisation-only** — never feed a t-SNE/UMAP embedding into a downstream algorithm as reduced features. Use PCA for that (linear, preserves global structure; UMAP can manufacture artificial clusters).

### 4.8 Frontend behaviour (`frontend/interface.py`)

- Labels are stored in an `int` column `human_label` on `anomaly_scores`; **`-1` means unlabelled**.
- Every label write persists the whole frame to `<output_dir>/ml_scores.csv`. Reload it in your script to resume a session (see the shipped Protégé example).
- "Retrain" → `run_active_learning()`, which calls `combine_data_frames(features, anomaly_scores)` then `.run(...)`, merges the returned columns back into `anomaly_scores`, and re-sorts by `column_to_sort_by` (honouring `show_unlabelled_first`).
- Retraining with zero labels set returns `{"status": "failed"}` and does nothing.
- `delete_labels()` resets everything to `-1`.

---

## 5. Installation

```bash
cd ~/software/astronomaly
python -m venv venv_astronomaly && source venv_astronomaly/bin/activate
pip install -r requirements.txt
pip install .
# verify with a shipped example:
python astronomaly/frontend/run_server.py astronomaly/scripts/raw_features_example.py
```

**Environment notes verified on this machine:**
- Astronomaly is **not currently installed** in the default environment.
- System Python is **3.14.4**; `astronomaly_env.yml` pins **3.8.5** and pins `numpy 1.19.1` / `scikit-learn 0.23.2` / `flask 1.1.2`. Use a dedicated venv/conda env — the pinned stack will not build on 3.14. `requirements.txt` uses `>=` bounds and is the more permissive of the two; try it first on a 3.10/3.11 interpreter.
- **`torch`, `torchvision`, `byol-pytorch` and `kornia` are NOT in `requirements.txt`.** `byol_features.py` raises `ImportError` at import time without them. Install separately only if you need BYOL:
  ```bash
  pip install torch torchvision byol-pytorch kornia
  ```
- Everything except BYOL runs CPU-only. BYOL wants a GPU (`byol_features.py` selects `cuda` when available, else CPU).

Shipped examples in `astronomaly/scripts/`: `galaxy_zoo_example.py` (images + ellipse + iForest + NeighbourScore), `galaxy_zoo_byol_protege_example.py` (**the best template for Protégé**), `raw_features_example.py` (**the best template for tabular data**), `goods_example.py`, `CRTS_example.py`.

---

## 6. Applying Astronomaly to BLUSE data

### 6.1 Mapping BLUSE products onto Astronomaly inputs

BLUSE emits four product types (see `BLUSE-technical-reference.md` §4.5). Each implies a **different search**:

| BLUSE product | Shape | Astronomaly route | What you would find |
|---|---|---|---|
| **Hit files** | Tabular: frequency, drift rate, SNR, beam, coarse channel, timestamp | `RawFeatures` + `FeatureScaler` → iForest/LOF + `NeighbourScore`, **or** PCA + Protégé | Hits with unusual *parameter combinations*: odd drift/frequency pairings, anomalous SNR-vs-band behaviour, beams with strange hit statistics |
| **Stamp files** | HDF5; per-antenna upchannelised time-frequency around a detection (62–64 antennas × time × freq × 2 pol) | Render to 2D cutouts → `ImageThumbnailsDataset`; or engineer per-antenna coherence statistics → `RawFeatures` | Unusual *dynamic-spectrum morphology*; **or, using the antenna axis, unusual spatial coherence** — the axis that actually separates sky signals from local RFI |
| **Filterbank HDF5** | Waterfalls per coherent beam + incoherent sum | 2D image cutouts → `ImageThumbnailsDataset` | Unusual broadband structure, RFI environments, instrumental states |
| **Raw GUPPI voltages** | Only for primary-time observations | Not directly usable — must be reduced to one of the above first | — |

**Recommendation for a one-week workshop:** start with **hit files → `RawFeatures` → Protégé**. It has no image-rendering step, no GPU dependency, runs in seconds, and gets you to the interactive labelling loop — which is where the actual insight is — on day one. Add stamp-image work as a second track only if time allows.

### 6.2 The dynamic-spectrum caveat (important)

BYOL's default augmentations are written for **galaxy images**, where rotation and flips are physically meaningless transformations. **This assumption is false for a time-frequency waterfall.** Time and frequency are not interchangeable axes; a drifting narrowband signal rotated by 90° is a physically different object, and its drift rate — the single most diagnostic parameter in the search — is exactly the quantity a rotation augmentation destroys.

If you use BYOL on BLUSE dynamic spectra, you **must** override `augmentation_params`:
- **Remove** `RandomRotation` (set `aug_rotation_p = 0`).
- **Remove or reconsider** `RandomVerticalFlip` / `RandomHorizontalFlip` — a flip in frequency negates the drift rate sign. Keep only if you genuinely want drift-sign invariance.
- **Keep** crops/blur/jiggle, which correspond to plausible nuisance variation.

Ignoring this will produce features that are deliberately blind to drift rate.

If in doubt for the workshop, prefer **hand-crafted features you understand** (`PSD_Features`, `WaveletFeatures`, `EllipseFitFeatures`, or your own drift/bandwidth/kurtosis statistics via `RawFeatures`) over BYOL. Interpretability is worth more than raw recall when you have five days and need to explain your findings.

### 6.3 The other preprocessing caveat

[L&R24] §9.3: **most of Protégé's failures traced to preprocessing, not to the algorithm.** High-dynamic-range radio images left faint structure invisible to the network even though humans (viewing with `asinh`) could see it plainly.

BLUSE dynamic spectra have this problem in an acute form — strong RFI can be orders of magnitude above a threshold-level hit in the same frame. Consequences:
- Use **separate transforms** for the machine and for the human display. `ImageThumbnailsDataset` supports this directly via `transform_function` vs. `display_transform_function`. [L&R24] found the machine-optimal transform was *not* the human-optimal one.
- [L&R24] Table 3, on their data: **linear scaling + sigma clipping** (recall₁₀₀ = 42.9) beat linear-no-clipping (38.2), asinh+clipping (31.0) and asinh-no-clipping (36.1). Do not assume this transfers to BLUSE — test it, since their own paper cautions the result is unlikely to be general.
- Consider stacking **multiple scalings of the same frame as separate channels** — [L&R24] §9.3 suggests this explicitly, and CNNs already ingest 3-channel input.
- **Per-frame normalisation will destroy absolute SNR information.** Decide whether you want that. If absolute brightness is diagnostic for you, normalise globally, not per-object.

### 6.4 Template: BLUSE hits → Protégé

```python
# bluse_hits_protege.py
# Run: python astronomaly/frontend/run_server.py bluse_hits_protege.py
import os
import pandas as pd
from astronomaly.data_management import raw_features
from astronomaly.postprocessing import scaling
from astronomaly.dimensionality_reduction import pca
from astronomaly.anomaly_detection import protege
from astronomaly.visualisation import umap_plot
from astronomaly.utils.utils import pca_based_initial_selection

data_dir   = '/home/bester/projects/bluse/aug_2026_workshop/data'
output_dir = '/home/bester/projects/bluse/aug_2026_workshop/astronomaly_output/hits/'
os.makedirs(output_dir, exist_ok=True)

# NB: RawFeatures treats any path containing the substring 'labels' as a LABEL
# file. Keep feature filenames free of that substring.
input_files = [os.path.join(data_dir, 'bluse_hits_features.csv')]

force_rerun = False


def run_pipeline():
    dataset = raw_features.RawFeatures(list_of_files=input_files,
                                       output_dir=output_dir)

    # Standardise: essential before distance-based methods and PCA.
    features = scaling.FeatureScaler(
        output_dir=output_dir, force_rerun=force_rerun).run(dataset.features)

    # PCA: Protege's initial selection assumes column 0 is the first PC.
    features = pca.PCA_Decomposer(
        threshold=0.95, output_dir=output_dir, force_rerun=force_rerun
    ).run(features)
    print('Features shape:', features.shape)

    # Initial query: 10 sources equally spaced along PC1.
    anomalies = pca_based_initial_selection(features, 10)

    # Resume any labels from a previous session.
    try:
        if 'human_label' not in anomalies.columns:
            df = pd.read_csv(os.path.join(output_dir, 'ml_scores.csv'),
                             index_col=0, dtype={'human_label': 'int'})
            df.index = df.index.astype('str')
            if len(anomalies) == len(df):
                anomalies = pd.concat((anomalies, df['human_label']),
                                      axis=1, join='inner')
    except FileNotFoundError:
        pass

    # ei_tradeoff=3 matches Lochner & Rudnick 2024; the code default is 0.5.
    active_learning = protege.GaussianProcess(
        features, output_dir=output_dir, force_rerun=force_rerun,
        ei_tradeoff=3)

    vis = umap_plot.UMAP_Plot(output_dir=output_dir,
                              force_rerun=False).run(features)

    return {'dataset': dataset,
            'features': features,
            'anomaly_scores': anomalies,
            'visualisation': vis,
            'active_learning': active_learning}
```

### 6.5 Template: BLUSE stamp/waterfall cutouts → classic path

```python
# bluse_stamps_classic.py
import os
from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import umap_plot

image_dir  = '/home/bester/projects/bluse/aug_2026_workshop/data/stamp_pngs/'
output_dir = '/home/bester/projects/bluse/aug_2026_workshop/astronomaly_output/stamps/'
os.makedirs(output_dir, exist_ok=True)

# What the MACHINE sees.
machine_transforms = [
    image_preprocessing.image_transform_greyscale,
    image_preprocessing.image_transform_scale,
]

# What the HUMAN sees -- deliberately different (Lochner & Rudnick 2024 s9.3).
display_transforms = [
    image_preprocessing.image_transform_inverse_sinh,
    image_preprocessing.image_transform_scale,
]

force_rerun = False


def run_pipeline():
    dataset = image_reader.ImageThumbnailsDataset(
        directory=image_dir, output_dir=output_dir,
        transform_function=machine_transforms,
        display_transform_function=display_transforms,
        display_image_size=256)

    features = power_spectrum.PSD_Features(
        nbins=50, output_dir=output_dir, force_rerun=force_rerun
    ).run_on_dataset(dataset)

    features = scaling.FeatureScaler(
        output_dir=output_dir, force_rerun=force_rerun).run(features)

    anomalies = isolation_forest.IforestAlgorithm(
        n_estimators=200, output_dir=output_dir, force_rerun=force_rerun
    ).run(features)

    # iForest: lower == more anomalous, so lower_is_weirder=True.
    anomalies = human_loop_learning.ScoreConverter(
        output_dir=output_dir).run(anomalies)
    anomalies = anomalies.sort_values('score', ascending=False)

    active_learning = human_loop_learning.NeighbourScore(
        alpha=0.1, regression_algorithm='RF',
        output_dir=output_dir, force_rerun=True)

    vis = umap_plot.UMAP_Plot(output_dir=output_dir).run(features)

    return {'dataset': dataset,
            'features': features,
            'anomaly_scores': anomalies,
            'visualisation': vis,
            'active_learning': active_learning}
```
*Verified against the checkout: `PSD_Features(nbins='auto')` and `PipelineStage.run_on_dataset(dataset)` both exist as used. But note the docstring's own words — PSD features are **translation and rotation invariant**. On a dynamic spectrum that means they discard drift direction and time-frequency orientation, i.e. exactly the diagnostic you probably care about. Use them as a fast baseline, not as your primary feature set.*

### 6.6 Suggested hand-crafted features for BLUSE hits

Not from the papers — proposed for this application, to be validated:

| Group | Candidate features |
|---|---|
| Signal | drift rate, |drift rate|, SNR, frequency, fractional offset within coarse channel, bandwidth |
| Context | hits per beam in the same recording, hits in the same coarse channel across the recording, band (UHF/L/S), F-engine mode |
| Coherence | coherent-beam SNR ÷ incoherent-beam SNR; number of beams containing a hit at the same frequency (**high count ⇒ RFI**) |
| Temporal | signal duration as a fraction of the 290 s recording; on/off structure |
| Per-antenna (stamps) | variance of per-antenna SNR; number of antennas above threshold; phase-closure-like statistics |

The **coherence group is the most technosignature-relevant**: a genuine sky signal appears in few beams with high coherent gain, whereas RFI appears across many beams with a coherent/incoherent ratio near 1. Feeding those ratios in as features lets the active-learning loop learn "RFI-like ⇒ score 0" very quickly.

---

## 7. Workflow checklist

1. Create an isolated Python environment (**not** system Python 3.14); install from `requirements.txt`; `pip install .`
2. Verify with `raw_features_example.py` before touching BLUSE data.
3. Decide **which search you are running** — parameter-space, morphology, or coherence. They are different searches with different features.
4. Build features. Cache them. This is the expensive step (~1000× everything else).
5. `FeatureScaler` → `PCA_Decomposer(threshold=0.95)`.
6. Choose the path: **Protégé** (deep/high-dim features, or targets not at the boundary) or **classic** (low-dim hand-crafted features, targets at the boundary). If unsure, run both and compare recall curves.
7. Write `run_pipeline()`; launch `run_server.py`; label in batches of 10; retrain; iterate.
8. Target **100+ labels**. Gains appear after ~10; both papers report strong performance at 100–200. [L&R24] used 400 for their full-dataset run.
9. Inspect the UMAP colour-coded by score to understand *where* your anomalies live. If the interesting objects are interior rather than boundary, that alone justifies Protégé over the classic path.
10. Record everything: feature set, hyperparameters, labeller identity, number of labels. Results are subjective by construction and are not reproducible without them.

---

## 8. Quick-reference constants

```
--- Classic Astronomaly (Lochner & Bassett 2021) ---
Score equation          S_hat = S * tanh(delta - 1 + arctanh(U_tilde))
Normalisation           eps1 = 0.1, eps2 = 0.85, U_max = 5
Distance penalty        delta = exp(alpha * d / d0)
alpha                   1 (L&B21) | 0.1 (L&R24) | code default 1
Label scale             0-5 integer
Regressor               Random Forest, n_estimators 100 (paper) / 200 (code)
Detectors               iForest (n_estimators 100/200) | LOF (n_neighbors 50/100)
Ellipse features        21, from percentile contours 90/80/70/60/50/0
Ellipse failure rate    ~1% of objects
Rank column             trained_score

--- Protege (Lochner & Rudnick 2024) ---
Regressor               Gaussian Process, kernel Matern() + WhiteKernel()
Acquisition             Expected Improvement
ei_tradeoff (epsilon)   3 in paper | 0.5 code default | >0.5 to keep variance low
Initial query           10 sources equally spaced along PC1
Query batch size        10
Labels used             100 (evaluation) | 400 (full dataset run)
Label scale             1-5 integer
Rank column             acquisition
No anomaly detection step at all

--- BYOL feature extraction ---
Architecture            EfficientNet-B0, ImageNet-initialised
Raw feature dim         1280 (avgpool layer)
PCA                     95% variance -> 52 features
Optimiser / batch / LR  Adam / 32 / 5e-4 scaled by batch_size/256
Epochs                  100
Deps                    torch, torchvision, byol-pytorch, kornia (NOT in requirements.txt)

--- Benchmarks (recall in top 100) ---
MGCLS eval subset       1031 sources, 86 interesting (8.34%)
Random 8 | Complexity 20 | Ellipse+iForest 32 | BYOL+Protege 48
BYOL + classic path     iForest 24 | LOF 14   (worse than hand-crafted)
Inter-rater agreement   Pearson 0.70, 12% differ by >1 point

--- Timing (L&B21 Appendix C; laptop, CPU-only, no parallelisation) ---
Galaxy Zoo 61,578 x 400x400 px:  features 56 min | iForest 2.2 s | AL 3.9 s | t-SNE 40 s
Simulations 50,000 x 100 feats:  LOF 199.9 s | AL 4.3 s | t-SNE 40.6 s
=> Feature extraction dominates by ~3 orders of magnitude. Cache it.
```

---

## 9. Caveats for downstream reasoning

- **The feature extractor bounds what is findable.** No Free Lunch. Never report a null result as a limit without stating the feature set.
- **Results are subjective by construction.** They depend on who labelled and how. Record the labeller.
- **Nothing generalises across datasets.** BYOL is trained on the data being searched; [L&R24] §9.2 explicitly declines to claim generalisation, and notes a held-out test set would be actively counterproductive here (it would mean ignoring part of the data you are trying to search).
- **Small-sample variance is large.** [L&R24] Appendix C concluded most hyperparameters had no significant effect against run-to-run noise on ~1000 sources. Do not over-tune on a workshop subset; repeat runs and quote a spread.
- **Preprocessing, not the algorithm, is the usual failure mode.** Especially with high dynamic range — which BLUSE has in abundance.
- **Astronomaly v2.0 is research software.** The README warns of bugs and instabilities and asks users to make contact for support. The classic-path leakage noted in §4.6 is one example of why the source is worth reading before trusting a result.
- **Nothing here detects technosignatures.** It ranks data by unusualness-weighted-by-your-interest. Every candidate needs the normal follow-up: RFI checks, coherence tests, repeat observations.

## 10. Citation

Cite **Lochner & Bassett (2021)** (arXiv:2010.11202) for Astronomaly, and **additionally Lochner & Rudnick (2024)** (arXiv:2411.04188) if Protégé is used. Software DOI: 10.5281/zenodo.14441057.
