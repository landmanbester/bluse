# Myburgh et al. 2026 (VLA High-Frequency SETI) Technical Reference

**Purpose:** dense, agent-oriented reference for the seven-filter COSMIC post-processing chain, its numbers, its disagreements with Tremblay et al. 2026, and what we adopted, rejected or changed as a result.

**Primary source:** Myburgh, T., Stiegler, N., Tremblay, C. D., Bright, J. S., Donnachie, R. A. 2026, *"The VLA and High-Frequency SETI: Expanding the Search for Life"*, IAC–25–A4,1,6,x95796, 76th International Astronautical Congress, Sydney, 29 Sep – 3 Oct 2025; arXiv:2608.18275v1, 18 August 2026, 15 pp. Local copy: `papers/Myburgh_2026.pdf`.

Section and figure numbers are the paper's. Numbers attributed to "our measurement" were computed on our catalogues and are reproducible.

---

## 1. Why this paper matters despite being about a different telescope

It is VLA/COSMIC, 25–50 GHz, four coherent beams. None of that is our instrument. Two things make it the most useful cross-check we have:

1. **Same lineage, blind targets.** Chenoa Tremblay is an author of both this and the K2-18b paper, and both run `seticore`. But K2-18b is a *known* system with a computable orbit, while these are ordinary Gaia catalogue stars with unknown companions — **our situation**. Where the two papers differ, this one is usually the one that applies to us.
2. **Three filters the K2-18b paper does not have**, one of which we adopted and one of which we measured to be a no-op on our data (which is itself a finding).

It also **contradicts the K2-18b paper on the coherent/incoherent test**, in a direction that exposed a real defect in our code. See §5.

---

## 2. Data and instrument

| | |
|---|---|
| Instrument | VLA + COSMIC, 21 compute nodes × 2 pipelines |
| Band | 25–50 GHz (K 18–26.5, Ka 26.5–40, Q 40–50) |
| Period | 2024-02-01 to 2025-04-30 |
| Hits | **1,967,056** |
| Observations | 696 unique, 56 s segments treated as independent |
| Resolution | ~2 Hz frequency, 0.524288 s time |
| Sub-bands | ~32 MHz, up to ~1.344 GHz processed simultaneously |
| Beams | 4 coherent + incoherent sum; beam at phase centre when no catalogue target in FOV |
| Targets | 45 Gaia sources (~59.25 h) + 6 SIMBAD sources at phase centre (~116.5 h) |
| Drift search | **±50 Hz s⁻¹, blind** |
| Detection | `seticore` TTDD at SNR > 8σ |

**False-positive baseline:** ~7,000 hits at the 8σ threshold per COSMIC observation on average. This dataset averaged 2,826 hits per observation (range 46 to 40,856) — lower, as expected for a quiet band.

**Calibration grading (§2.1).** Each hit inherits a phase-stability grade in [0,1] from the calibration scan immediately preceding it. Above 0.6 ⇒ less than 40% reduction in coherent sensitivity, typically one or two bad antennas. Distribution: ~116,000 below 0.6, ~825,000 at or above, ~1,000,000 with no preceding calibration to grade.

> **They record the grade and refuse to cut on it**, noting only that poor phase will already have weakened the signal and may cost it the SNR cut. Same discipline as our `flag_*` columns.

**Known RFI, 25–50 GHz (Table 1):** Ka 29.5–30 GHz (VSAT satellite internet), Ka 34.875 GHz, Ka 36.286 GHz (Ka-band tracking beacons). **None of their candidates fell in any of them** — a third independent instance of published RFI tables removing nothing (cf. MeerKAT UHF and S4 in Tremblay et al.).

---

## 3. The seven filters (§3)

### Filter 1 — shared frequency at different sky positions
Reject hits at the same frequency appearing at different sky positions. Our multi-beam coincidence cut. **Removes >85% of all hits on its own** (§5, Fig. 8).

### Filter 2 — exactly zero drift
Same as Tremblay §3.2 and our `flag_zero_drift`. Stated cost, explicitly: *"our resulting limits apply only to signals with measurable drift rates > 0.04 Hz s⁻¹"*, and they acknowledge a real signal could show zero drift over a short integration if the net line-of-sight acceleration were small or the drift unresolved.

### Filter 3 — duration-conditioned SNR floor ⭐ **adopted**

```
timesteps < 16  (< 16 x 0.524288 s ~ 8 s)   =>  require SNR > 15
timesteps >= 16                             =>  require SNR > 10
```

Rationale: with only basic geodesic calibration and no noise reduction in the real-time pipeline, `seticore` reports many false positives, and a short observation gives it too few samples for a reliable noise estimate — "randomly aligned bright pixels in a dynamic spectra". They call the specific thresholds "arbitrary", chosen to make the problem tractable.

**Three papers say this independently:**

| Source | Statement |
|---|---|
| Myburgh §3.3 | SNR > 15 below 16 timesteps |
| Tremblay §3.3 | ~80% of 8σ detections with **fewer than 64 time samples** were false positives |
| Czech et al. 2026 §6 | beams shorter than 150 s are not viable for technosignature searching |

**Our measurement.** `numTimesteps` in our files:

| file | timesteps | hits |
|---|---|---|
| `uhf_short` | **14 and 15** | 208,774 (all) |
| `lband_short_clean` | 24 | 463,625 |
| `sband_short` | 24 | 38,576 |
| `uhf_long` | 36 | 299,878 |
| `lband_long` | 56, 57 | 557,690 |
| `mk_sample_hits` | 57 | 15,119 |
| `sband_long` | 59 | 36,132 |

`uhf_short` is entirely inside the warned regime, and **456 of its 1,193 survivors had SNR ≤ 15**. Adopting filter 3 takes it to 786 survivors. Every other file is untouched. Note that by Tremblay's "fewer than 64 samples" phrasing *all* our files qualify; we use Myburgh's threshold of 16 because it is the one with a stated number attached.

### Filter 4 — SNR ceiling of 100
As Tremblay §3.3. They are explicit about the cost: it forces EIRP_min rejection limits of < 9.76×10¹⁴ W and > 7.06×10¹⁷ W, i.e. surviving transmitters must be **500 to 35,000 times the Arecibo planetary radar** (2×10¹³ W). We keep our looser 1e6 ceiling for the reasons in `papers/Tremblay-technical-reference.md` §6.4.

### Filter 5 — the "lonely hit" test ❌ **measured to be a no-op; not adopted**

Assume every hit within 10 Hz of another hit in the same observation is a modulation of it; drop the hit and its modulations, keeping only hits with no neighbour within 10 Hz. Ten hertz = five 2 Hz fine channels, which they concede "was likely an overly safe precaution".

**Our measurement, and the reason we did not implement it:**

| interpretation | effect on our data |
|---|---|
| per **observation** (across all 64 beams) | **98.5–99.9% of hits dropped** |
| per **(observation, beam)** | **10 hits of 1,619,794** (0.0006%) |

The first reading is catastrophic and wrong for us: our RFI appears in all 64 beams at one frequency, so nearly every hit has 63 neighbours and the filter degenerates into the multi-beam cut with no survivors. The second is the intended reading, and it does essentially nothing — because `seticore` already reports only the top hit per frequency window. **The smallest gap between two hits in one beam anywhere in our survey is 631 Hz**; the median per-beam minimum gap is 3.5–6.8 kHz.

> The corollary is a real limitation, and it is the same one Margot et al. 2021 raised for turboSETI (cited in Tremblay §5): **we cannot see two signals closer than a few hundred hertz in one beam.** Filter 5 is redundant here precisely because that blindness is built in upstream.

### Filter 6 — continuity along the predicted drift trajectory ⚠️ **not implemented; recipe recorded**

For each hit, examine subsequent observations of the same position and check whether the signal persists **along the frequency trajectory the earlier drift rate predicts**. Discard hits that do not — "isolated events". Removes only **4% of what reaches it**, and they flag it for refinement, noting that anything reaching filter 6 is already drifting and reasonably powerful, so this class should be rare.

**This is the constructive complement of our cut 7, and we only had the destructive half.**

| | criterion | verdict |
|---|---|---|
| Tremblay §3.6 / our `cut_repeat` | recurs at the **same** frequency and drift | RFI |
| Myburgh filter 6 | recurs at the **drift-predicted** frequency | strongest candidate |

Not a contradiction once the distinction is seen: an *identical* frequency on another day means a terrestrial emitter; a frequency that has *moved by the predicted amount* means the same source, still drifting. Implementing it needs trajectory propagation across observations of one target, which our per-observation binning does not do. Worth building.

### Filter 7 — coherent vs incoherent ⚠️ **contradicts Tremblay; see §5**

```
coherent_S/N <= sqrt(N_antennas) x incoherent_S/N   =>  "most likely RFI"
```

---

## 4. Results (§4, Fig. 8)

**Zero hits survived all seven filters.** Filter 1 alone removed >85%; filter 6 removed 4% of what reached it.

Their own reading, quoted because it is the most useful sentence in the paper:

> "With no remaining hits, this suggests either an incorrect assessment of what a technosignature looks like or the frequency range has more RFI than suggested in simple 'RFI' scans provided by NRAO. At this time, none of these steps are clear indicators of either scenario, so are not adopted into the real-time pipeline."

**A zero-survivor pipeline is not self-evidently a working pipeline**, and they decline to promote the filters until they can tell over-filtering from a genuine null. Remember this when our own survivor count falls.

**Sensitivity.** S_limit = SEFD / (B_e √(n_pol · n · t_int · Δν)) × SNR, with B_e = 0.9, n_pol = 2, n = 25 antennas, t_int = 56 s, Δν = 2 Hz, SEFD 500 Jy (K) and 1300 Jy (Q). Gives 74.24 Jy/beam at 25 GHz and 193.02 Jy/beam at 50 GHz. EIRP_min spans 2.24×10¹⁵ to 3.31×10²⁰ W over sources at 234 to 90,090 pc — 112 to 1.655×10⁷ Arecibo units.

**Imaging (§4.3).** COSMIC stamp files hold **per-antenna calibrated voltages**, so a candidate can be correlated (BLRI → UVH5 → `pyuvdata` → CASA Measurement Set) and imaged with `tclean`, optionally de-drifting the visibilities first to concentrate power. A sky source is an unresolved point source at the beam position; RFI is structured and unlocalised. Demonstrated on **Voyager 1** at X band (2023-04-26, 8420.4191 MHz, five 2 Hz channels): predicted RA 17:15:42, Dec +12:20:53.3; recovered at 17:15:42.8, +12:20:49.

> **Not available to us.** Our stamps are beamformed intensity cubes, not per-antenna voltages. The single most decisive discriminator in this literature is out of reach unless the BLUSE team can supply voltages.

---

## 5. The Filter 7 contradiction — and the defect it exposed in our code

The two papers state the same inequality with opposite meanings:

| Source | Statement |
|---|---|
| **Tremblay §3.7** | "a signal of the same origin in the incoherent beam will have a reduced sensitivity determined by: SNR_coh ≤ √N × SNR_incoh" — an inequality a **real** signal obeys |
| **Myburgh filter 7** | "if coherent_S/N ≤ √N × incoherent_S/N then the signal is most likely **RFI**" |

Read literally these cannot both be right. As a strict test Myburgh's would reject everything, since the bound is essentially always satisfied.

**The physics settles it.** Coherent summation of N antennas gains signal power as N² against noise as N, so SNR ∝ N. The incoherent sum gains signal as N against noise as √N, so SNR ∝ √N. A source **at the beam centre therefore sits at a ratio of about √N**. Interference, which never phases up, falls well short of it. Both papers' own supporting text agrees:

- Tremblay §4.1 looks for hits showing "an expected power **ratio (4.69)**" — that is √22 for 22 antennas, i.e. they test for the ratio *equalling* √N, not merely respecting a bound.
- Myburgh Fig. 5 caption: the example hit was cut "for having a measured coherent power **not sufficiently more powerful** than the measured incoherent power, marking it as non-localized interference."

So the discriminant is **ratio ≈ √N**, and *both* tails are interference. Tremblay's phrasing is a true but non-discriminating upper bound; Myburgh's is a correct criterion sloppily written as an inequality.

**What this exposed.** Our `cut_incoherent` flagged only `ratio > sqrt(n_ants)` — the physically impossible tail — and passed everything below. That is the less useful half: the dominant RFI signature is a ratio *far below* √N, and we were not testing for it at all. Fixed 2026-09 to a two-sided test with a slack factor `--coh-ratio-tol` (default 2, i.e. accept √N/2 to 2√N). The tolerance is ours; neither paper gives one.

The cut remains inert — `incoherentPower` was never measured for our data, which Tremblay's Table 3 independently confirms for MeerKAT — so this changes no current number. It matters for correctness if the incoherent beam ever arrives.

**A second, unresolved caveat.** Both papers state the relation in **SNR**. Our columns are `power` and `incoherentPower`, so we apply an SNR relation to a **power ratio**. That substitution is not obviously valid and has never been checked, because we have no data to check it against. Verify before trusting this cut.

---

## 6. Effect on our pipeline

Three changes, all measured:

| Change | Source | Effect |
|---|---|---|
| Duration-conditioned SNR floor added, on by default (`--snr-min-short 15`, `--short-timesteps 16`) | filter 3 | `uhf_short` 1,193 → **786** survivors; no other file affected |
| Maximum drift-rate cut switched **off** by default | §3 blind ±50 Hz/s | `lband_short_clean` 670 → **740** survivors; no other file affected |
| `cut_incoherent` made two-sided | filter 7 vs Tremblay §3.7 | none now (cut inert); correctness for later |

Total survivors 5,692 → **5,355**. Clustering is bit-identical: feature *values* did not change, only Track A's provenance flags.

**Why the drift cut went off.** We had added it a day earlier from Tremblay §3.2. Myburgh searches ±50 Hz/s deliberately "as many of our targets are toward unknown planetary systems" — which is our case, not K2-18's, and the coefficient we were using encodes K2-18b's specific orbital acceleration. It was also biting inside the range `seticore` actually searched: on `lband_short_clean` the limit lands at 0.358–0.402 Hz/s against an observed maximum of 0.4203, with 4,257 hits sitting at the extreme `driftSteps` — precisely where a fast-drifting genuine signal would appear. In a blind survey a false negative costs more than one more waterfall to inspect. `--max-drift-coeff 4.18e-4` restores the Tremblay behaviour.

**Not adopted:** filter 5 (measured no-op, §3), filter 6 (needs cross-observation trajectory propagation — worth building, recipe in §3), imaging (needs per-antenna voltages we do not have).

---

## 7. Inconsistencies and errata

1. **Filter 7 contradicts Tremblay §3.7.** The headline. See §5. Neither paper acknowledges the other's phrasing.
2. **Hit accounting.** §2.1's calibration-grade buckets sum to ~1,941,000 (~116,000 + ~825,000 + ~1,000,000) against §4's exact 1,967,056. All three are stated as approximate, so this is probably rounding, but it does not close.
3. **Date range.** The abstract says "reviewing data from February 2024 to the present"; §2 says COSMIC began recording in October 2023; §4 gives 2024-02-01 to 2025-04-30. The last is the operative one.
4. **§3.3 phrasing.** "we only keep hits with timesteps < 16 = 16 × 0.524288s ≈ 8s and SNR > 15" reads as a conjunction to keep, when the intent — clear from the following sentence — is a floor applied conditionally. Also "≈ 8s" is 8.39 s.
5. **§4.1 typo:** "Accounting for filters 3 and 4 3, this study has imposed SNR rejection thresholds of ≤ 10 and ≥ 100" — stray "3".
6. **"Taylor tree de-dispersion"** in the acronym list and §2.3. The algorithm originates in de-dispersion but here performs de-*drifting* (Doppler). Tremblay et al. use the same shorthand, so it is community usage rather than an error — but do not read dispersion into it.

---

## 8. Quick-reference constants

```
Citation        Myburgh et al. 2026, IAC-25-A4,1,6,x95796, arXiv:2608.18275v1
Instrument      VLA + COSMIC, 25-50 GHz, 4 coherent beams + incoherent sum
Period          2024-02-01 to 2025-04-30, 696 observations, 56 s segments
Hits            1,967,056        (2,826/obs avg, range 46-40,856)
Baseline FP     ~7,000 hits at 8 sigma per COSMIC observation
Resolution      ~2 Hz, 0.524288 s
Drift search    +/-50 Hz/s, BLIND (targets are unknown systems)
Targets         45 Gaia (~59.25 h) + 6 SIMBAD at phase centre (~116.5 h)
Cal grades      ~116k < 0.6, ~825k >= 0.6, ~1M ungraded; NOT used to reject

Filter 1  same frequency, different sky positions      removes >85%
Filter 2  drift == 0                                   limits apply >0.04 Hz/s
Filter 3  timesteps < 16 -> SNR > 15; else SNR > 10
Filter 4  SNR <= 100                                   => 500-35,000 x Arecibo
Filter 5  drop hits with a neighbour within 10 Hz      (5 fine channels)
Filter 6  must persist along predicted drift track     removes 4% of remainder
Filter 7  coherent vs incoherent, sqrt(N)              see section 5

Result          ZERO hits survived all seven filters
Known RFI       Ka 29.5-30 GHz (VSAT), 34.875, 36.286 GHz -- caught nothing
Sensitivity     74.24 Jy/beam at 25 GHz, 193.02 Jy/beam at 50 GHz
                B_e 0.9, n_pol 2, n 25, t_int 56 s, dnu 2 Hz
                SEFD 500 Jy (K), 1300 Jy (Q)
EIRP_min        2.24e15 - 3.31e20 W over 234-90,090 pc
                = 112 - 1.655e7 Arecibo units (L_A = 2e13 W)
Imaging         BLRI -> UVH5 -> pyuvdata -> CASA MS -> tclean
                Voyager 1 recovered as a point source at X band

OUR MEASUREMENTS
Filter 3 bites  uhf_short is 14-15 timesteps for all 208,774 hits;
                456 of its 1,193 survivors had SNR <= 15
Filter 5 no-op  10 of 1,619,794 hits have an in-beam neighbour within 10 Hz;
                smallest in-beam gap anywhere is 631 Hz
                (per-observation instead: 98.5-99.9% of hits dropped)
```

---

## 9. Caveats for downstream reasoning

1. **This is a different telescope and a very different band.** 25–50 GHz is quiet; MeerKAT's UHF/L/S are not. Do not transfer hit densities or RFI conclusions, only method.
2. **Their targets are blind, like ours.** Where this paper and the K2-18b paper differ, this is usually the one that applies to us. The drift-range question is the clearest case.
3. **Zero survivors is a warning, not a triumph.** They say so and act on it by withholding the filters from production.
4. **Filter 5 is a no-op here, and that is a statement about our detector, not our data.** `seticore` reports one hit per frequency window, so close pairs are invisible upstream.
5. **Filter 6 is the constructive counterpart to our persistence cut and is not implemented.** Until it is, we can reject repeating signals but cannot *promote* a signal for repeating on its predicted track.
6. **The coherent/incoherent test is two-sided and ours was not.** Fixed, still inert, and the power-versus-SNR substitution remains unverified.
7. **Imaging is the decisive test and we cannot do it.** It needs per-antenna voltages. Worth asking the BLUSE team whether any exist for our observations.
