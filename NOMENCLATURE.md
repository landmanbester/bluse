# Nomenclature

The words this project uses, what they actually mean, and what they do **not**
mean. Written because `family` appeared a dozen times across three documents
and was defined in none of them.

This is the single source for these definitions. Other documents link here
rather than restating them.

---

## The chain

```
  hit  ──►  cluster  ──►  family  ──►  explained / residue
 (one      (batch-local,   (pooled       (matched against
 detection) meaningless    across all     documented radio
            on its own)    batches)       allocations)
```

Everything below is measured on `lband_short` (856–1068 MHz), which is the
worked example throughout.

## Hit

**One narrowband detection.** The telescope's search software (`seticore`)
sweeps the data and reports "something at 874.9955 MHz, drifting at −0.11 Hz/s,
in beam 44, SNR 1783". A hit is one row in a table.

`lband_short` has **866,002 hits**; the whole survey has 2,022,171
(2,014,055 after removing duplicates that appear in two files).

*Not* a signal, a source, or a detection of anything real. Most hits are
interference, and one physical transmitter produces thousands of hits across
beams and observations.

## Stamp

**The small waterfall image saved with each hit** — frequency across, time
downward. This is what `bluse-explore stamps` plots: a drifting carrier shows
as a slanted bright line.

## Beam

MeerKAT observes **64 positions on the sky simultaneously**. This is the single
most powerful discriminator available:

- a real signal from one direction appears in **one or two** beams;
- terrestrial interference enters through the antenna sidelobes and appears in
  **all 64**.

`n_beams` is how many beams a hit's frequency was seen in at the same time. A
population sitting in 64 beams is interference no matter what else is true
about it.

## Feature

**One of the numbers computed per hit** — frequency, |drift rate|, SNR,
spectral shape statistics, bandwidth, and so on. Together they place each hit
as a point in a multi-dimensional space, which is what the clustering acts on.

**16 are registered; 15 are used by default.** `f08_turning_bw_hz` is off
because it is unresolved for ~72% of hits — our spectral window is far narrower
than the one the feature was designed for — so including it mostly clusters on
"hit the window edge". Both counts appear in this repo; they are not a
contradiction.

Defined and registered in `src/bluse/features.py`; run
`bluse-features --list` to see them.

## Cluster

**A group of hits whose 15 features are similar**, found by HDBSCAN.

The catch, and it matters: we cannot cluster 866,002 hits at once, so it runs
in **batches of 3,000, eight times over** — and HDBSCAN numbers its clusters
0, 1, 2… **from scratch in every batch**. So "cluster 7" in one batch has
nothing to do with "cluster 7" in the next.

`lband_short` produces **32,508 clusters**. That number is bookkeeping, not a
finding. Measured: two runs of an identical configuration differing only in
shuffle seed agreed at **ARI 0.024** — essentially not at all — because the
identities are batch artefacts.

## Family

**A group of clusters that are the same thing, merged across all batches.**

Each cluster's average position in the 15-D space is computed, a similarity
tree is built over those averages, and the tree is cut. Clusters landing
together become one family.

> Sorting a warehouse of photographs 3,000 at a time. Each batch you make piles
> and label them 1, 2, 3 — but the labels restart every batch. A *family* is
> what you get afterwards by comparing every pile with every other and merging
> the ones showing the same thing.

`lband_short`: **32,508 clusters → 40 families**, and reproducibility across
seeds goes from 0.024 to about 0.5. Families are the level at which the
batching artefact cancels, which is why every result is quoted per family.

### Three things a family is not

**Not a discovered count.** Every cut of the similarity tree is uniquely
determined by how many groups you ask for — verified over 1,400 thresholds with
zero exceptions. There is no natural number of families in the data. 40 was
chosen; it was not found. Always state the count alongside any family result.
See [`docs/matching-cut-experiment-2026-09.md`](docs/matching-cut-experiment-2026-09.md).

**Not a transmitter.** The typical family spans several MHz (median
interquartile range 9.4 MHz on `lband_short`). "GSM downlink" is a 35 MHz
allocation shared by thousands of handsets and base stations; the family
covering it is that whole population, not one device. A family is closer to *a
band of spectrum with characteristic behaviour*.

**Not one-to-one with a population.** Families fragment. All five unexplained
`lband_short` families contain the same 874.9955 MHz emitter, which alone is
20.4% of the residue. So a family count is an **upper bound** on the number of
distinct populations, not an estimate of it.

## Explained / residue / candidate

Each family's hits are matched against documented radio allocations — the SARAO
knowledge-base table plus ITU allocations, in `src/bluse/rfi_masks.py`.

- **Explained** — over half the family's hits fall in a documented band. On
  `lband_short`: 35 of 40 families, 89.7% of hits (GSM up/downlink, aircraft
  transponders/DME).
- **Residue** — the rest: 5 families, 10.3% of hits. **This is the point of the
  whole exercise.** A taxonomy of RFI exists to isolate what is left over.
- **Candidate** — residue that is *also* beam-confined (≤4 beams). On
  `lband_short` this is **zero**, which is the pipeline working: L-band is
  saturated with interference and everything unexplained there is still in all
  64 beams.

`bluse-cluster --match` writes `<tag>_families.csv` and `<tag>_candidates.csv`;
`bluse-explore stamps --rows <tag>_candidates.csv --each-family` plots them.

## RFI score

**A per-hit number saying how much a hit looks like interference**, learned
from the spatial filter and computed without ever seeing a beam count.
`bluse-score` writes it as the `rfi_score` column on all 2,014,055 hits.

Precisely: it is the model's estimate of **P(this hit was detected in ≥32
beams | its 12 stamp-morphology features)**. The spatial filter's verdicts are
free labels — ≥32 beams is interference, ≤2 beams is confined — so a classifier
can learn to reproduce them from the stamp alone. It reaches **0.9887 ROC-AUC**
where Track A's entire hand-built flag set reaches 0.9373.

The reason to want it when `n_beams` is already in the catalogue: **the spatial
filter is blind on single-beam hits**, and a technosignature is a single-beam
hit. So is a weak local emitter that only clears threshold at boresight. The
filter cannot separate them; a score that never looks at beams can be asked.

### Three things the score is not

**It is not a probability that a hit is RFI.** The labels are
*positive-unlabelled*: `weak_label == 0` means seen in ≤2 beams, **not verified
clean**. Most single-beam hits are still interference. So a high score is
strong evidence of RFI — the hit is morphologically indistinguishable from
signals independently known to be terrestrial — while a low score says only
that the hit does not resemble this survey's multi-beam RFI.

**A low score is not a candidate.** Measured, on the 524 hits the shortlist
actually contains: they collapse to about **49 distinct (file, 0.1 MHz)
groups, the top ten holding 73%**. The shortlist is a handful of persistent
narrow emitters seen many times over, not 524 independent oddities — and
looking at their stamps shows two morphological populations, neither of them
astrophysical. It is an outlier ranking that works, which is a different and
more useful thing than a detection list.

**It does not transfer to a differently selected hit list.** Pointed at
`mk_sample_hits` — pre-filtered before delivery, ~25 hits per observation
against ~2,931 elsewhere — a model trained on the other six files scores 94.2%
of its nominally-confined hits as RFI, and is *right* to: the same hits appear
in `lband_long` counted in a mean of 29.71 beams against 1.87 there. Beam
multiplicity counted on a sparse hit list undercounts, so that file's labels
are artefacts. Re-check before applying the score to new data — and do not use
`mk_sample_hits` for anything beam-derived at all.

## Verdict — pruned / uncertain / shortlist

What `bluse-score` calls each **Track A survivor**, by where its score falls:

- **pruned** — score > 0.9. On the survey: **3,003 of 4,565 (66%)**. Hits the
  classical filter passed that look exactly like multi-beam RFI. The
  actionable output: two thirds of the vetting list, removed with a reason.
- **uncertain** — 1,038 (23%). No verdict either way.
- **shortlist** — score < 0.1. 524 (11%). Read the caveat above before
  quoting this number.

The thresholds are conventions, not detection criteria; `--shortlist-below`
and `--pruned-above` move them.

## The metrics, in one line each

Quoted constantly, so worth pinning down.

| term | what it measures | watch out for |
|---|---|---|
| **family ARI** | do two runs at different seeds put the same hits together? 0 = chance, 1 = identical | granularity-relative — 0.67 at 36 families and 0.77 at 16 for the *same* configuration. Only comparable at equal family count |
| **narrow share** | fraction of hits in families spanning under 1 MHz | at its floor once families are merged: 6.8% at natural granularity, 0.19% at 36 families. Do not rank on it there |
| **median family span** | typical frequency width of a family | full span is max−min, which one outlier dominates. The interquartile range is the honest number: **111.6 MHz vs 9.4 MHz** on the same 40 families |
| **distance share** | how much one feature contributes to the distance the clusterer sees | the *k-NN* share is the one that matters, not the global one — HDBSCAN responds to local density |
| **weak label** | provisional RFI / beam-confined tag used for enrichment checks | not ground truth. There is no ground truth in this data yet |
| **RFI score** | how much one hit looks like multi-beam interference, from stamp morphology alone | positive-unlabelled: high means RFI, low means *unlike this survey's RFI* — never "candidate" |

## If you read one paragraph

A **hit** is one detection. A **cluster** is a batch-local grouping whose
identity is an artefact. A **family** is what makes those identities mean
something across the whole file, and it is our unit of "a kind of signal" — but
it is a *band of spectrum with characteristic behaviour*, not a transmitter,
its count is a choice we make rather than a number we discover, and one
physical emitter can be split across several. The **residue** — families
matching nothing documented — is what the pipeline exists to produce, and the
**candidates** are the residue that is also confined to a few beams.

Track E adds a second, independent axis to that chain. Where a family says
*what kind of signal this is*, the **RFI score** says *how much this one hit
looks like interference* — learned from the spatial filter, computed without a
beam count, and therefore still available on the single-beam hits where the
filter has nothing to say. High is strong evidence of RFI; low is not evidence
of anything except unusual morphology.
