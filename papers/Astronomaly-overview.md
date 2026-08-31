# Astronomaly: A Quick-Start Overview

**A high-level guide for human readers, oriented toward applying Astronomaly to BLUSE data**

Sources:
- Lochner, M. & Bassett, B. A. — *"Astronomaly: Personalised Active Anomaly Detection in Astronomical Data"*, Astronomy and Computing (2021), arXiv:2010.11202v2. (`papers/astronomaly_2020.pdf`) — **the framework paper.**
- Lochner, M. & Rudnick, L. — *"Astronomaly: Protégé — Discovery Through Human-Machine Collaboration"*, ApJ (2024), arXiv:2411.04188v3. (`papers/astronomaly_2024.pdf`) — **the Protégé extension.**
- Code: `~/software/astronomaly` (v2.0, `github.com/MichelleLochner/astronomaly`)
- Companion: `papers/BLUSE-overview.md`, `papers/BLUSE-technical-reference.md`

---

## The core idea in one paragraph

Anomaly detection algorithms are good at finding things that are *statistically unusual*. They are terrible at finding things that are *interesting*, because "interesting" is subjective and depends entirely on who is asking. Run a plain outlier detector on any real astronomical dataset and the top of the ranked list fills up with instrumental artefacts and RFI — technically anomalous, scientifically worthless. Astronomaly's insight is to put a human in the loop: the machine ranks, the human scores a handful of examples 0–5 on *relevance*, and a second machine-learning layer learns to predict that relevance score across the whole dataset. Objects that look like things you called boring get pushed down; objects that look like things you called interesting get pushed up; objects in unexplored regions of feature space keep their raw anomaly score so you never silently lose a genuinely new class. In the papers' tests, roughly 100–200 labels — a few minutes of clicking — roughly doubles the number of interesting anomalies found in the first 100 objects viewed.

---

## Why this matters for the BLUSE workshop

BLUSE produces exactly the situation Astronomaly was built for:

| BLUSE property | Consequence |
|---|---|
| ~1.5 million coherent beams processed, ~29,000 new objects/month | Nobody can eyeball the data |
| Detection threshold at SNR 6 | The "hit" list is dominated by RFI, not signals |
| No labelled training set of technosignatures | Supervised classification is impossible by construction |
| "Interesting" is defined by the observer | Relevance is subjective — exactly Astronomaly's premise |

The honest framing for a technosignature search: Astronomaly will *not* find you an alien. What it will do is efficiently surface the small fraction of your data subset that looks *unlike everything else* — which in practice means unusual RFI environments, unexpected instrumental behaviour, odd beam-to-beam structure, and the occasional genuinely strange time-frequency morphology. That is a legitimate and useful workshop outcome, and it is also the fastest route to understanding what your data actually contain.

---

## The two algorithms — and which one to use

Astronomaly ships **two distinct approaches**. They are philosophically different, and the 2024 paper is largely an argument that the second one is better when you have good features.

### 1. Classic Astronomaly (2020): anomaly detection *then* active learning

```
features → anomaly detector (iForest / LOF) → rank → human scores top objects
                                                  ↓
                              random-forest regressor predicts relevance
                                                  ↓
                        combine: anomaly score × f(relevance, uncertainty) → re-rank
```

The combination step is the paper's novel contribution:

> **Ŝ = S · tanh( δ − 1 + arctanh(Ũ) )**

where `S` is the raw anomaly score, `Ũ` is the normalised predicted user relevance, and `δ = exp(α · d/d₀)` is a **distance penalty** — `d` being the distance in feature space to the nearest human-labelled object. The behaviour is the point:

- **Far from any label** (`δ` large) → `tanh → 1` → `Ŝ ≈ S`. You fall back to the raw anomaly score, so novel classes are never suppressed just because you haven't seen one yet.
- **Near labels you scored high** → score is preserved or amplified.
- **Near labels you scored zero** → score is crushed. Artefacts you've already rejected stop coming back.

**Use this when:** your features are hand-crafted and low-dimensional, and the interesting objects genuinely sit at the *edge* of feature space.

### 2. Protégé (2024): skip anomaly detection entirely

The 2024 paper found that with modern deep-learning features, classic anomaly detection *fails*. Interesting sources turn out to be **buried inside** feature space rather than sitting at its boundary — deep networks have no loss term encouraging outliers to separate out. Isolation forest simply can't reach them.

So Protégé throws the anomaly detector away and reframes the problem as **active regression**:

```
features → PCA → pick 10 spread-out sources → human scores them
                              ↓
        Gaussian Process regression predicts (score, uncertainty) everywhere
                              ↓
        Expected Improvement acquisition picks the next 10 most informative
                              ↓
                        repeat ~10–40 times
```

There is no anomaly score at all. Protégé just learns "what does this person find interesting?" as a smooth function over feature space, and uses the Gaussian Process's uncertainty estimate to decide what to ask about next. The **Expected Improvement** acquisition function has a tunable ε that trades exploitation (ε low) against exploration (ε high); the paper uses ε = 3, deliberately biased toward exploration so it doesn't get stuck in one corner.

**Use this when:** you have deep/self-supervised features, or your targets aren't extreme outliers. In the MGCLS test it was decisively better — see below.

### The headline comparison

On a 1031-source evaluation set containing 86 "interesting" sources, after a user views only the **top 100** objects:

| Method | Interesting sources found |
|---|---|
| Random ordering | 8 |
| Source complexity (# Gaussian components) | 20 |
| Classic Astronomaly (ellipse features + iForest + active learning) | 32 |
| **BYOL features + Protégé** | **48 (56% of all of them)** |

Two details worth internalising:
- Protégé needed only **100 human scores** to get there.
- Using BYOL features with the *classic* anomaly-detection path scored just **24** (iForest) or **14** (LOF) — *worse than the simple hand-crafted features*. Feature extractor and detection algorithm must be matched to each other; you cannot mix and match freely.

---

## Features are the whole ball game

Both papers hammer the same point: **the feature extractor determines what kinds of anomaly you are capable of finding.** This is the No Free Lunch theorem in practical dress. The 2020 ellipse-fitting features are exquisitely sensitive to weird galaxy *morphology* and completely blind to weird *colour* — not a bug, a consequence.

For BLUSE, this translates into a decision you must make deliberately:

- If you feed Astronomaly **tabular hit properties** (frequency, drift rate, SNR, beam number…), you will find hits with unusual *parameter combinations*.
- If you feed it **time-frequency images** (stamps, filterbank waterfalls), you will find signals with unusual *morphology in the dynamic spectrum*.
- If you feed it **per-antenna structure**, you will find signals with unusual *spatial coherence* — arguably the most technosignature-relevant axis, since that is precisely what distinguishes a sky signal from local RFI.

These are different searches. Pick one on purpose, or run several and compare.

Available feature extractors in the code: ellipse-fitting shape features, 2D power spectral density, wavelet decomposition, flux histograms, raw flattened data, pretrained CNN embeddings, autoencoders, and BYOL self-supervised features.

### About BYOL

BYOL (Bootstrap Your Own Latent) is the self-supervised feature extractor Protégé was built around. The idea: take an image, produce two randomly *augmented* views of it (rotate, flip, crop, blur), and train two coupled networks so both views map to the same representation. Because rotating an image doesn't change what it *is*, the network is forced to learn what actually matters and discard what doesn't — with **no labels required**. The second-to-last layer becomes your features (1280 dims for EfficientNet-B0), then PCA compresses that to ~50.

The critical caveat for BLUSE: **your choice of augmentations encodes your assumptions about what is physically meaningless.** For galaxy images, rotation is meaningless — a galaxy rotated 90° is the same galaxy. **For a dynamic spectrum this is emphatically false.** Time and frequency are not interchangeable axes, and a drifting signal rotated 90° is a physically different thing. If you use BYOL on BLUSE waterfalls, you must revisit the augmentation list. The defaults in the code will actively destroy the information you care about most.

---

## Practical shape of a session

1. **Prepare data.** Get your BLUSE subset into either a features table or a directory of image cutouts.
2. **Write a pipeline script.** One Python file that defines `run_pipeline()` and returns a dictionary with five keys. That's the entire contract.
3. **Launch.** `python astronomaly/frontend/run_server.py <your_script>.py`, then open `http://127.0.0.1:5000/`.
4. **Score.** Work down the ranked list, giving each object 0–5 for how interesting it is. Aim for 100+ labels; you can start seeing gains after 10.
5. **Retrain.** Press the button. The list re-sorts.
6. **Iterate.** Repeat until the top of the list is dominated by things you actually want to look at.
7. **Inspect.** Use the UMAP/t-SNE view to understand the structure of your feature space and where your anomalies sit in it.

Labels persist between sessions in `ml_scores.csv` in your output directory, so you can stop and resume.

Timing intuition from the 2020 paper (ordinary laptop, no GPU, no parallelisation): feature extraction on 61,578 400×400 images took ~56 minutes; anomaly detection took 2 seconds; active learning took 4 seconds. **Feature extraction dominates everything else by three orders of magnitude.** Budget accordingly, cache aggressively (the framework does this automatically via checksums), and start with a small subset.

---

## Two things the papers are refreshingly honest about

**Protégé finds things humans miss.** Section 7.5 of the 2024 paper shows sources Protégé ranked highly that both authors had scored *low* — and on re-inspection, agreed they'd been wrong. Faint diffuse structure and subtle filaments that get missed when a human is grinding through hundreds of images. The algorithm is not merely reproducing your judgement; it can sharpen it.

**Preprocessing causes most of the failures.** Nearly every high-human-score source that Protégé *missed* was missed because the preprocessing (linear scaling) made faint structure invisible to the network, even though the humans had viewed the same source with `asinh` scaling. The fix is not a better algorithm — it's better preprocessing. Note the counter-intuitive corollary the paper reports: the transform that works best for the *machine* was **not** the one that works best for *humans*. They deliberately use different scalings for the two, and you probably should too. With BLUSE's enormous dynamic range between strong RFI and threshold-level hits, this will bite.

---

## Honest limitations

- **No completeness guarantee.** No Free Lunch: for any algorithm there exists an anomaly it cannot see. Mitigate with multiple feature sets and multiple detectors; never claim a null result.
- **Subjective by design.** Two of the 2024 authors scoring the same 100 sources agreed only to a Pearson correlation of 0.70, with 12% differing by more than one point — and they had *talked to each other first*. Your results are your results, not the field's.
- **Doesn't generalise across datasets.** BYOL is trained on the dataset you're searching; the whole procedure must be repeated for new data.
- **Small-data variance is real.** The 2024 hyperparameter study found most hyperparameters had no significant effect against the run-to-run noise. Don't over-tune on a small workshop subset.
- **The code carries a warning.** The README says it is actively developed and may contain bugs, and recommends contacting the author for support. Treat it as research software.

---

## Glossary

| Term | Meaning |
|---|---|
| **Feature extraction** | Turning complex data into a short vector of numbers a learning algorithm can use. |
| **Anomaly score** | How unusual an object is, per an unsupervised algorithm. Higher = weirder. |
| **Relevance / user score** | 0–5 (classic) or 1–5 (Protégé): how *interesting* a human finds it. The subjective part. |
| **Active learning** | Iteratively asking a human to label the most *informative* examples rather than random ones. |
| **Acquisition function** | The rule for choosing what to ask about next. Protégé uses Expected Improvement. |
| **iForest** | Isolation Forest — isolates points with random splits; fewer splits needed = more anomalous. |
| **LOF** | Local Outlier Factor — density-based; flags points in low-density regions. Good when anomalies sit close to normal data. |
| **Gaussian Process** | Regression that returns a mean *and* an uncertainty — the uncertainty is what makes Protégé's acquisition function work. |
| **BYOL** | Self-supervised deep feature extractor; learns from augmented image pairs, no labels needed. |
| **t-SNE / UMAP** | 2D embeddings for *visualising* high-dimensional feature space. Visualisation only — not dimensionality reduction for downstream use. |
| **Oracle** | Active-learning jargon for the human doing the labelling. That's you. |
