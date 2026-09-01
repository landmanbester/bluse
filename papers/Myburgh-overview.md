# High-Frequency SETI on the VLA: Seven Filters and Zero Survivors

**A high-level overview for human readers**

Source: Myburgh, T., Stiegler, N., Tremblay, C. D., Bright, J. S., Donnachie, R. A. — *"The VLA and High-Frequency SETI: Expanding the Search for Life"*, IAC–25–A4,1,6,x95796, 76th International Astronautical Congress, Sydney, 29 Sep – 3 Oct 2025; arXiv:2608.18275v1, 18 August 2026. Local copy: `papers/Myburgh_2026.pdf`.

---

## The one-paragraph version

COSMIC, the VLA's commensal SETI backend, recorded 1,967,056 narrowband hits between 25 and 50 GHz over fifteen months — a band almost nobody has searched, and one that is nearly free of terrestrial interference. The authors put those hits through **seven bespoke filters** and **nothing survived**. Not "one candidate that turned out to be RFI": zero. They say plainly what that means — either their model of a technosignature is wrong, or the band has far more interference than NRAO's scans suggest — and they decline to promote the filters into the real-time pipeline until they know which. It is the same group as the K2-18b paper, one telescope over, and the value to us is entirely in §3: seven filters, three of which we had never considered.

---

## Why this matters for the BLUSE workshop

It is the closest thing we have to a **second opinion on Track A**. Tremblay et al. 2026 gave us the recipe; this paper takes the same lineage of code, points it at *blind* Gaia targets instead of one known planet, and adds filters the K2-18b paper did not have.

That "blind targets" part is the crux. K2-18b is a system whose mass, orbit and period are known, so its plausible Doppler drift can be computed. Myburgh et al. are looking at ordinary catalogue stars whose planets — if any — are unknown. **That is our situation exactly.** Where they differ from the K2-18b paper, they differ in the direction that applies to us.

They also disagree with the K2-18b paper on one filter outright, in a way that turned out to matter for our code. See below.

---

## The seven filters

| | Filter | What it does |
|---|---|---|
| 1 | Shared frequency, different sky positions | The multi-beam test. Removed **over 85%** of all hits on its own |
| 2 | Zero drift rate | Stationary means local |
| 3 | **Duration-conditioned SNR floor** | SNR > 15 for hits with fewer than 16 time samples, SNR > 10 above |
| 4 | SNR ceiling of 100 | Aggressively cuts artefacts |
| 5 | **"Lonely hit" test** | Drop any hit with another detection within 10 Hz in the same observation |
| 6 | **Continuity across observations** | Keep signals that *persist along their predicted drift trajectory*; discard one-offs |
| 7 | Coherent vs incoherent power | A sky signal phases up; interference does not |

Filters 3, 5 and 6 are new relative to the K2-18b paper. Filters 1, 2, 4 and 7 are the same cuts we already had.

**Filter 3 is the one that matters most to us.** seticore's noise estimate is computed from the data it has; with few time samples to average over, that estimate is poor and the search reports detections where there is nothing — "randomly aligned bright pixels in a dynamic spectrum". So a short hit has to be brighter to be believed. This is the third independent statement of the same problem: Tremblay et al. measured that ~80% of 8σ detections with few time samples were false positives, and Czech et al. separately call beams shorter than 150 s unviable for searching. Three papers, one warning.

**Filter 6 is philosophically interesting** because it points the opposite way from the cut we already had. Our cross-epoch persistence cut treats a signal that keeps reappearing as interference. Myburgh's filter 6 *requires* reappearance — but at the **frequency the earlier drift rate predicts**, not the same frequency. Those are compatible once you see the distinction: a signal at an *identical* frequency on another day is terrestrial; a signal at the *drift-predicted* frequency is the same source, still drifting, and is the strongest possible candidate. We were only ever implementing the rejecting half.

---

## What they found

Nothing. And the honesty about it is the best part of the paper:

> "With no remaining hits, this suggests either an incorrect assessment of what a technosignature looks like or the frequency range has more RFI than suggested in simple 'RFI' scans provided by NRAO. At this time, none of these steps are clear indicators of either scenario, so are not adopted into the real-time pipeline."

**A pipeline that returns zero candidates is not obviously a working pipeline.** It might be perfectly calibrated, or it might be over-filtered until nothing could possibly get through. They cannot tell, so they hold the filters back. That is the right instinct and it is worth remembering when our own survivor count drops.

They are equally candid about the cost of the cuts. Filter 4's SNR ceiling of 100 means any surviving transmitter would have to be **500 to 35,000 times more powerful than the Arecibo planetary radar**. Filter 2's zero-drift rejection means "our resulting limits apply only to signals with measurable drift rates > 0.04 Hz/s". Filter 5's 10 Hz window they call "likely an overly safe precaution". These are limits stated as limits, not buried.

## Two nice pieces of technique

**Calibration grading.** Every hit inherits a phase-stability grade from the calibration scan before it, between 0 and 1, where above 0.6 means less than a 40% loss of coherent sensitivity. About 116,000 hits scored below 0.6, 825,000 above, and roughly a million had no preceding calibration to grade at all. They deliberately **do not reject** on grade — they record it, and note that poor phase will have weakened the signal already. Recording a quality flag instead of acting on it is exactly the discipline our `flag_*` columns exist for.

**Imaging as the final discriminator.** Because COSMIC saves per-antenna voltages, a candidate can be correlated into visibilities and *imaged*. A real signal appears as a point source at the beam position; interference appears as structured emission that is not localised anywhere. They demonstrate it on **Voyager 1** at X band and recover it as an unresolved point source within a few arcseconds of the JPL ephemeris position. This is the most decisive test in the whole literature — and it is unavailable to us, because our stamps are beamformed intensities, not per-antenna voltages.

---

## What we changed because of this paper

Three things, all verified against our data:

**We added the duration-conditioned SNR floor.** `uhf_short` turns out to be 14–15 time samples for every one of its 208,774 hits — squarely inside the regime all three papers warn about. Of its 1,193 survivors, **456 had SNR ≤ 15**. Applying filter 3 takes it to 786.

**We turned the maximum drift-rate cut off by default.** We had added it a day earlier from the K2-18b paper. Myburgh et al. search ±50 Hz/s deliberately, "as many of our targets are toward unknown planetary systems" — our case, not K2-18's. And our limit was landing *inside* the range seticore actually searched. In a blind survey, throwing away a fast-drifting signal costs more than looking at one more waterfall. The cut is still there; it is now opt-in.

**We fixed the coherent/incoherent test, which was half-implemented.** See below.

**We did not implement filter 5**, and measuring it is why. Applied per observation across all 64 beams it removes 98.5–99.9% of our data — it degenerates into the multi-beam cut with the volume turned up. Applied per beam, the way it is meant, it removes **10 hits out of 1,619,794**. seticore already reports only one hit per frequency window, so the "lonely hit" property holds by construction: the smallest gap between two hits in one beam anywhere in our survey is 631 Hz. The filter is a no-op for us — which is itself worth knowing, because it also means **we are blind to two signals closer than a few hundred hertz**.

---

## Glossary

- **COSMIC** — the VLA's commensal SETI backend; BLUSE's sibling, same `seticore` search.
- **K / Ka / Q band** — 18–26.5, 26.5–40, 40–50 GHz. Quiet, expensive, barely searched.
- **Calibration grade** — 0 to 1, phase stability of the calibration scan preceding a hit.
- **EIRP** — how powerful an isotropic transmitter would need to be for us to have seen it. Scales as distance squared, which is why 90 kpc sources need 10²⁰ W.
- **Arecibo power unit** — ≈2×10¹³ W, the planetary radar. The field's yardstick.
- **Stamp file** — here, per-antenna calibrated voltages, which is what makes imaging possible. Ours are beamformed intensities instead.
- **BLRI** — Breakthrough Listen Interferometry package, correlates stamp voltages into visibilities.
