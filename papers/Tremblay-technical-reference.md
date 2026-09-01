# Tremblay et al. 2026 (K2-18b) Technical Reference

**Purpose:** dense, agent-oriented reference for the post-processing framework that Track A implements — every cut, its stated parameters, the numbers it produced, the paper's internal inconsistencies, and an explicit map onto `src/bluse/track_a_filter.py`.

**Primary source:** Tremblay, C. D., Chaudhary, S., Li, Megan G., Sheikh, Sofia Z., Myburgh, T., Czech, D., MacMahon, D. E., Demorest, P. B., Donnachie, R. A., Siemion, A. P. V., Gajjar, V., Lebofsky, M., Wandia, K., Perez, K. I., Madhusudhan, N. 2026, *"A Narrowband Technosignature Search Toward the Hycean Candidate K2-18b Using the VLA and MeerKAT"*, arXiv:2602.09553v1, draft 11 February 2026, 18 pp. Local copy: `papers/tremblay_2026.pdf`.

Section, table and figure numbers below are the paper's. Where a claim is ours rather than theirs it says so.

---

## 1. Why this paper is the Track A specification

It is the only published post-processing recipe written for **interferometric, multi-beam, seticore-produced** narrowband hits — that is, for exactly the data we have, by the team that produced it. §3 is an ordered, parameterised filter chain and §5.3 states the intent explicitly: a general framework, "readily scalable to larger target samples".

The framework itself is credited as built from Tusay et al. (2024), "modified to account for the larger number of coherent beams distributed across the telescope FOV".

**The one structural difference that changes everything downstream:** they have a single target of interest (K2-18b) among 4 (VLA) or 64 (MeerKAT) beams, so their spatial test is *"is this hit unique to the target beam?"* We have one catalogued Gaia target per beam, all of them of interest, so ours is *"in how many of the beams formed does this hit appear?"* Same physics, inverted bookkeeping. Any transfer of their numbers must account for it.

---

## 2. Observations and instrument parameters (§2, Tables 1–2)

| Telescope | Backend | Bands | Coherent beams | Fine chan | Integration | Threshold |
|---|---|---|---|---|---|---|
| VLA | COSMIC | S 2.4–3.6, C 5–7.2, X 8.1–9.8 GHz | 4 + incoherent sum | 2 Hz | 56 s | 10σ |
| MeerKAT | **BLUSE** | UHF 544–1015, L 900–1670, S4 2625–3500 MHz | 64 | ~1 Hz | 290 s | 10σ |

- MeerKAT antennas: **64** (Table 2). VLA: 17–25 depending on epoch.
- Both run **seticore**'s Taylor-tree de-drift search.
- 7 VLA S-band epochs, 3 each C and X; **single-epoch** MeerKAT observations in each band.
- BLUSE ran in "semi-commensal" mode on Director's Discretionary Time (DDT-20230920-DC-01), appended 5 minutes ahead of primary observations. BLUSE buffers ~290 s of full bandwidth before it must stop and process.
- Distance 38 pc; the whole K2-18 system fits inside one coherent beam of either telescope.

> **Our data is not this data.** Ours is the autonomous commensal survey, not this DDT campaign. Our fine channels are 1.013 Hz (UHF), 1.594 Hz (L), 1.630 Hz (S) — see §6.2 — and our integrations are ~118 s and ~286 s.

---

## 3. The filter chain (§3)

Figure 3 is the block diagram. Order matters; this is the order.

### 3.1 Known-RFI frequency masks

- Per telescope and band, a `.csv` of start/end frequencies of identified RFI sources, plus an identifier where available.
- **All detections overlapping a masked interval are excluded, independent of pointing direction.**
- Sources: NRAO's per-configuration, per-semester RFI plots from time-averaged spectra in 128 MHz partitions, plus a table of known strong emitters. For MeerKAT, SARAO's equivalent — with the note that "regularly updated reference plots are not yet available".
- VLA S band used an **empirically derived** mask instead, built in Tremblay et al. (2025) from time-averaged calibrator observations with an identical instrumental setup, merged with the NRAO database.

> Corroborates our experience directly. See §5 for the two MeerKAT bands where the published masks removed **zero** hits.

### 3.2 Drift rates

Two separate cuts:

- **Exactly zero drift is removed.** "such detections likely correspond to local RFI, where the transmitter and receiver are either stationary relative to one another or moving too slowly to produce a measurable Doppler shift within our drift-rate resolution."
- **A maximum |drift| bound**, from Li et al. (2022, 2023), covering 99% of plausible signals given K2-18's orbital dynamics plus Earth's rotation:

| Frequency | max \|drift\| |
|---|---|
| 544–1500 MHz | ~0.4 Hz s⁻¹ |
| ~4.5 GHz | 1.879 Hz s⁻¹ |
| ~10 GHz | 4.177 Hz s⁻¹ |

Those three anchors are consistent with a linear scaling in frequency, as Doppler drift requires: 0.4/1000 = 4.0e-4, 1.879/4500 = 4.18e-4, 4.177/10000 = 4.18e-4 Hz s⁻¹ per MHz. **Our derivation, not theirs**, but it reproduces all three of their numbers and is what `track_a_filter.py` uses.

> ⚠️ **See §7.1 — §4 does not apply these limits.**

### 3.3 Strong and weak signal excision

**SNR window 10 ≤ SNR ≤ 100.** Both bounds are empirically justified, and the justifications are worth keeping:

- **Lower.** Synthetic signals injected with `setigen` and searched with `seticore` (Steigler, Tremblay, Myburgh et al., in prep.) showed that **~80% of 8σ detections with fewer than 64 time samples were false positives** — random bright pixels aligning so the de-drifted power spectrum shows a peak. 10σ adopted, matching BLUSE's native detection limit.
- **Upper.** 1,000 stamps inspected at random: **over 90% of high-SNR detections were instrumental** — present in a single antenna, at up to 3000σ, not sky-dependent. Capped at 100, with the explicit note that the original data is retained "when a better algorithm for this type of artifact is developed".

Stated cost: signals from K2-18b brighter than 10¹⁴ W (COSMIC) / 10¹³ W (BLUSE) are not detectable through this cut.

### 3.4 Multi-beam analysis

Cross-match a detection against detections in other beams. Tolerances:

- **Frequency: ±1 fine channel.** Because fine channelisation is an FFT, a signal broader than one channel spans several bins. They quote "±2 Hz for COSMIC" (2 Hz channels) and "1 Hz tolerance" for MeerKAT (1 Hz channels).
  > **The rule is ±1 fine channel, not ±1 Hz.** Their MeerKAT number is 1 Hz *because their channel was 1 Hz*. Ours are 1.013–1.630 Hz. See §6.2.
- **Drift: ±1 drift step.** "The seticore algorithm searches along all possible drift trajectories to identify the drift rate that maximizes the signal intensity... For broad or nonlinearly varying signals, the derived drift rate can differ by approximately one drift step."

### 3.5 Primary and secondary transits

A transmitter on or near the planet must be invisible while the planet is behind the star. Retain only detections unique to the visible phase.

- K2-18b: period 33 d, eccentricity 0.2, transit duration 2.66 hr, occulted ~2.7 hr per orbit — visible **99.6%** of the time.
- **Not applicable to these observations** (no data taken during occultation); retained in the paper as framework.
- **Not applicable to us at all** — our targets are catalogued stars, not transit-timed planets.

### 3.6 Multiple-day comparison

Per Li et al. (2022): a signal from a moving transmitter received by a moving observer traces a sinusoidal frequency evolution, so over multiple days it **cannot** recur at an identical frequency *and* drift rate.

- Same frequency **and** drift across epochs → RFI or instrumental.
- Unique in **both** → strongest candidate.
- N/A for MeerKAT here: single-epoch observations.

### 3.7 Coherent / incoherent beam comparison

A coherent beam points; the incoherent sum does not and is √N less sensitive. For a common-origin signal:

```
SNR_coherent  ≤  sqrt(N_antennas) × SNR_incoherent            (Eq. 1)
```

Signals of different origin that happen to share a frequency and drift will not obey it. Called "a strong delineator of RFI within the field".

- §4.1 worked example: expected ratio **4.69** = √22 for 22 VLA antennas; 663 coherent/incoherent matches, **none** satisfying it.
- **N/A for every MeerKAT band** (Table 3). The MeerKAT observations did not include an incoherent beam.

> This independently confirms what the BLUSE team told us: `incoherentPower` is not available for our data either. Cut 5 in Track A is implemented, tested, and permanently inert.

### 3.8 Visual inspection

Plot the dynamic spectrum of every survivor. A candidate must appear in all online antennas and show a distinct linear drift. Figure 7 shows four survivors, all RFI or artefacts: a bright pixel with a weak drifter, an inconsistent-intensity drifter, a weak signal in a noisy region, and a single-antenna instrument artefact.

---

## 4. Sensitivity and EIRP (Appendix A)

```
S_limit = SEFD / (B_e * sqrt(n_pol * n * t_int * dnu)) * SNR      (A1)
SEFD    = 2 * k_B * T_sys / A_eff                                 (A2)
EIRP    = 4 * pi * F_min * D^2                                    (A3)
```

- Beamformer efficiency B_e: **0.9** for COSMIC (methanol maser, Tremblay et al. 2024), **0.93** for BLUSE (**measured on JWST**).
- Δν is the single channel width — they search for signals confined to one channel.
- F_min = S_limit × channel bandwidth; D in metres; 1 Jy ≡ 1e-26 W m⁻² Hz⁻¹.

**Table 4 — EIRP limits (×10¹² W):** MeerKAT UHF 1.6, L 1.2, S4 1.1; VLA S 23, C 16, X 13. All at or below Arecibo's ~10¹³ W.

---

## 5. Results (§4, Table 3)

| Step | VLA S | VLA C | VLA X | MK UHF | MK L | MK S4 |
|---|---:|---:|---:|---:|---:|---:|
| initial | 10,743,382 | 9,920,694 | 310,961 | 59,030 | 398,867 | 4,712 |
| 3.1 RFI | 6,737,277 | 237,333 | 178,944 | **59,030** | 118,677 | **4,712** |
| 3.2 drift | 589,011 | 31,517 | 18,974 | 5,902 | 58,734 | 1,488 |
| 3.3 SNR | 384,955 | 19,816 | 11,067 | 2,991 | 13,386 | 1,103 |
| toward K2-18b | 141,450 | 4,911 | 242 | 46 | 225 | 180 |
| 3.4 multibeam | 13,847 | 768 | 0 | 0 | 0 | 0 |
| 3.5 transits | N/A | N/A | N/A | N/A | N/A | N/A |
| 3.6 day | 4,316 | 86 | – | N/A | N/A | N/A |
| 3.7 beam | 0 | 0 | – | N/A | N/A | N/A |
| 3.8 visual | 0 | 0 | – | – | – | – |

**Final result: zero technosignature candidates in any band or epoch.**

Observations that transfer:

- **The published RFI masks removed nothing in MeerKAT UHF and S4.** §4.6: "None of these overlapped with frequencies flagged in the catalog of persistent RFI." Direct corroboration of our own finding that the SARAO table flags 3 of 38,576 hits in our S band.
- **The drift filter is the workhorse for MeerKAT.** UHF 59,030 → 5,902 (−90.0%), with "≈83% of detections exhibiting zero drift" and drift rates spanning −10 to +9 Hz s⁻¹. Ours run 22–47% zero-drift.
- **VLA S band: masks removed only 37%** of 10.7M hits, versus 97.6% for C band. RFI mask effectiveness varies enormously by band.
- **Every survivor at the visual-inspection stage was RFI.** For C band, "90% of the signals were within a MHz of each other around 6.6 GHz" and per-antenna spectra were inconsistent.

---

## 6. Map onto our Track A (`src/bluse/track_a_filter.py`)

### 6.1 Cut-by-cut

| Paper | Ours | Status |
|---|---|---|
| 3.1 RFI masks | `cut_rfi_bands` + `rfi_masks.py` | ✅ plus ITU allocations of our own (tagged) and `--derive-mask` |
| 3.2 zero drift | `flag_zero_drift` | ✅ |
| 3.2 max drift | `flag_drift_high` | ✅ **added 2026-09**; see §6.3 |
| 3.3 SNR window | `flag_snr_low` / `flag_snr_high` | ⚠️ floor 10 matches; ceiling defaults to 1e6, not 100 — deliberate, see §6.4 |
| 3.4 multi-beam | `cut_multibeam` | ✅ tolerance now ±1 fine channel, not ±1 Hz; see §6.2 |
| 3.5 transits | — | not applicable: no transit-timed targets |
| 3.6 multi-day | `cut_repeat` | ⚠️ adapted; now drift-aware, see §6.5 |
| 3.7 coherent/incoherent | `cut_incoherent` | ✅ implemented, permanently inert — no `incoherentPower` |
| 3.8 visual inspection | `bluse-explore stamps`, Cluster Bench | ✅ interactive rather than batch |

**Our structural departure: nothing is deleted.** Their chain narrows a dataframe per step. Each of our cuts writes a boolean `flag_*` column and `pass_all` is the AND of their negations, so any cut can be inspected, disputed, or re-decided by re-filtering the parquet. `<name>_cutflow.csv` records both the marginal effect of each cut and its sequential contribution.

### 6.2 Multi-beam tolerance — fixed 2026-09

§3.4 specifies **±1 fine channel**. Their "1 Hz" is a consequence of their 1 Hz channel, not the rule. Our channel widths:

| Band | `foff` | old `--tol-hz` | now |
|---|---|---|---|
| UHF | 1.013 Hz | 1.0 | 1.013 |
| L | **1.594 Hz** | 1.0 | 1.594 |
| S | **1.630 Hz** | 1.0 | 1.630 |

We were matching ~37% too tightly in L and S band, under-counting beam multiplicity and letting multi-beam RFI through. `--tol-hz` now defaults to per-file `|foff|`; pass a number to override.

### 6.3 Maximum drift rate — added 2026-09

We implemented zero-drift but never the upper bound. `--max-drift-coeff` (Hz s⁻¹ per MHz) applies a per-hit limit of `coeff × frequency_MHz`; **4.18e-4** reproduces all three of their anchor values. It defaults to **0, i.e. off** — see `papers/Myburgh-technical-reference.md` §6 for why a blind survey should not adopt a limit derived from one known planetary system.

Two honest caveats, recorded in the code:

1. **The coefficient is K2-18-specific.** It bounds Earth's rotation (~1.1e-4 Hz s⁻¹ per MHz, universal) *plus* K2-18b's orbital acceleration. Our targets are arbitrary Gaia sources with unknown companions, so treat it as a generous envelope rather than a derived limit for each target.
2. It is not inert on our data: it flags 12,756 hits in `lband_short_clean` and 1,134 in `uhf_short`, and zero elsewhere.

### 6.4 SNR ceiling — deliberate divergence

Their 100 is justified by *their* artefact population. Ours has a detached population at 10⁷–10⁸ that we want to see rather than silently drop, so `--snr-max` defaults to 1e6. Pass `--snr-max 100` for the literal recipe. `flag_snr_high` is recorded either way, so the strict version is a filter on the parquet, not a re-run.

### 6.5 Cross-epoch persistence — adapted, and now drift-aware

Their §3.6 is a *repeat detector* for one target across epochs: same frequency **and** drift on two days ⇒ RFI. Ours is closer to a spectral-occupancy filter — a frequency bin recurring across many observations — because we have 100–143 observations per file and "any repeat" would flag nearly everything.

Two changes made 2026-09 to bring it closer to the paper:

- The flag now requires a **drift match as well as a frequency match** (`--repeat-tol-steps`, default 1, mirroring §3.4's one-drift-step tolerance), recorded as `n_obs_at_freq_drift`.
- `n_obs_at_freq` (frequency only) is **kept unchanged** because Track B's provenance columns and weak labels depend on it.

`--min-obs` (default 5) has no analogue in the paper; it is ours, and it is a knob.

### 6.6 Antenna count

`--n-ants` defaults to 62. Table 2 says MeerKAT used **64** for this campaign. Ours is a survey-average of available antennas, not their number. Numerically irrelevant while cut 5 is inert, but set it correctly if `incoherentPower` ever arrives.

---

## 7. Inconsistencies and errata in the paper

Recorded so nobody "fixes" them in our direction later.

### 7.1 §3.2's drift limits are not the limits applied in §4 — **the one that matters**

| Band | §3.2 prescription | §4 applied |
|---|---|---|
| MeerKAT L, 900–1670 MHz | ~0.4 Hz s⁻¹ | **±1.9** (§4.5) |
| MeerKAT S4, 2625–3500 MHz | ~1.1–1.5 Hz s⁻¹ (scaled) | **±1.9** (§4.6) |
| VLA C, 5–7.2 GHz | ~2.1–3.0 Hz s⁻¹ (scaled) | **±1.9** (§4.2) |
| VLA X, 8.1–9.8 GHz | ~3.4–4.1 Hz s⁻¹ | ±4.2 (§4.3) ✅ |

±1.9 Hz s⁻¹ is the **4.5 GHz** value from §3.2 applied as a blanket to three bands spanning 0.9–7.2 GHz — too loose for L and S4, too tight for C. Only X band follows the prescription. Table 3's numbers therefore reflect the blanket value, not the stated method.

**We follow §3.2**, the physically motivated, frequency-scaled version the text argues for. Anyone comparing our cut-flow to their Table 3 must know this.

### 7.2 §3.4 and §4.1 give different matching tolerances

§3.4: ±1 fine channel (±2 Hz for COSMIC) and **±1 drift step**. §4.1: "frequency differences greater than ±2 Hz and drift-rate differences exceeding **±0.1 Hz s⁻¹**". COSMIC's drift step is 2 Hz / 56 s = 0.036 Hz s⁻¹, so ±0.1 Hz s⁻¹ is ~2.8 steps. The frequency tolerances agree; the drift tolerances do not.

### 7.3 Table 3's note contradicts §4.5

The note says the "Toward K2-18b" row is "the number of signals **unique to that sky position**". But §4.5 reads: "225 signals appeared solely in the K2-18b beam. However, no signals toward K2-18b were **unique** when compared to the other 63 coherent beams" — which is self-contradictory if both uses of "unique" mean the same thing. The consistent reading is that the row counts hits **detected in** the target beam, and the following row applies the uniqueness test. Read it that way.

### 7.4 Unit errors in §1

"an estimated mass of 8.63 ± 1.35 M⊙, a radius of 2.61 ± 0.09 R⊙" — those must be **M⊕ and R⊕**; a sub-Neptune is not 8.63 solar masses. §2.2 separately gives the radius as **2.34 R⊕**, disagreeing with §1's 2.61 regardless of unit. Cosmetic for us, but it means the paper has not been closely proofed.

### 7.5 Minor

- §2.1 cites the null stellar-emission result as Wandia et al. **2025a**; §5.1 cites it as **2025b**.
- Abstract: "a limit of 10¹² to 10¹³ W". Table 4 spans 1.1e12 to 2.3e13.
- §3.3 states the SNR ceiling "for the COSMIC data", but §4.4–4.6 and Table 3 apply 10–100 to MeerKAT as well.
- Typo: "Therfore" (§3.5); "Table 2.." (§2.2).

---

## 8. Quick-reference constants

```
Citation           Tremblay et al. 2026, arXiv:2602.09553v1
Targets            K2-18b, 38 pc, P = 33 d, e = 0.2, visible 99.6% of orbit

Detection          seticore, Taylor-tree de-drift, 10 sigma
COSMIC             4 coherent beams + incoherent sum, 2 Hz, 56 s, 17-25 ants
BLUSE              64 coherent beams, ~1 Hz, 290 s, 64 ants

SNR window         10 <= SNR <= 100
  lower rationale  ~80% of 8-sigma detections with <64 time samples were false
  upper rationale  >90% of high-SNR detections were single-antenna artefacts,
                   up to 3000 sigma
Drift              exactly 0 removed; |drift| <= f(frequency):
                     0.4 Hz/s   at 544-1500 MHz
                     1.879 Hz/s at 4.5 GHz
                     4.177 Hz/s at 10 GHz
                   => 4.18e-4 Hz/s per MHz  (our fit to their anchors)
Multi-beam         +/-1 fine channel, +/-1 drift step
Coherent/incoh     SNR_coh <= sqrt(N_ant) * SNR_incoh;  4.69 = sqrt(22)
Beamformer eff.    0.90 COSMIC (methanol maser), 0.93 BLUSE (JWST)

EIRP limits        MeerKAT UHF 1.6e12, L 1.2e12, S4 1.1e12 W
                   VLA     S 2.3e13, C 1.6e13, X 1.3e13 W

MeerKAT result     UHF 59,030 -> 0;  L 398,867 -> 0;  S4 4,712 -> 0
RFI mask effect    removed ZERO hits in MeerKAT UHF and S4
Zero-drift rate    ~83% (their UHF)  vs  22-47% (our files)
```

---

## 9. Caveats for downstream reasoning

1. **Their §4 does not apply their §3.2 drift limits.** §7.1. Do not compare cut-flows without accounting for it.
2. **Their multi-beam question is not ours.** One target among many beams versus one target *per* beam. Their "0 unique to the target beam" is not comparable to our "7,143 survivors".
3. **The incoherent test is N/A on MeerKAT here, and unavailable to us.** The recipe's strongest discriminant is missing from both. Everything rests on multi-beam coincidence.
4. **Published RFI masks can remove literally nothing.** Zero in two of three MeerKAT bands. Empirical masks are not a nicety.
5. **Their SNR ceiling of 100 is calibrated to their artefacts, not ours.** Adopting it would silently drop our 10⁷–10⁸ population.
6. **They are single-epoch on MeerKAT; we are not.** Steps 3.6 and 3.7 are N/A for them on MeerKAT but 3.6 is available to us across 100–143 observations per file.
7. **"No candidates" is not "no signal".** Their own §5.4: conservative masking can exclude regions where genuine signals sit; RFI models are incomplete; only approximately linear drift is searched.
8. **The tolerance rule is ±1 fine channel, not ±1 Hz.** §6.2. This bit us for real.
