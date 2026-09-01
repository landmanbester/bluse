# The K2-18b Search: Where Track A Comes From

**A high-level overview for human readers**

Source: Tremblay, C. D., Chaudhary, S., Li, Megan G., Sheikh, Sofia Z., Myburgh, T., Czech, D., MacMahon, D. E., Demorest, P. B., Donnachie, R. A., Siemion, A. P. V., Gajjar, V., Lebofsky, M., Wandia, K., Perez, K. I., Madhusudhan, N. — *"A Narrowband Technosignature Search Toward the Hycean Candidate K2-18b Using the VLA and MeerKAT"*, draft 11 February 2026, arXiv:2602.09553v1 (18 pp.). Local copy: `papers/tremblay_2026.pdf`.

---

## The one-paragraph version

K2-18b is a sub-Neptune in its star's habitable zone, and JWST spectra suggest a hydrogen-rich atmosphere over a possible ocean — a "Hycean world". This paper points two interferometers at it: the VLA with the COSMIC backend and MeerKAT with **BLUSE, the same backend that produced our data**. Across 544 MHz to 9.8 GHz and multiple epochs covering a full orbit, they find nothing, and set upper limits of roughly 10¹²–10¹³ W on any persistent narrowband transmitter in the system. The null result is not why this paper matters to us. What matters is §3: a **complete, ordered, written-down post-processing recipe** for turning millions of interferometric hits into a candidate list. That recipe is our Track A.

---

## Why this matters for the BLUSE workshop

It is the only published description of how to filter BLUSE hits, written by the people who built BLUSE, using BLUSE data.

Every other paper we have describes either the instrument (Czech et al. 2026) or a method developed for a single dish (GLOBULAR, on GBT). This one describes the *interferometric* post-processing chain, and it is explicit that the framework is meant to be reused: "these techniques are not specific to K2–18b... readily scalable to larger target samples."

There is one structural difference we have to keep in mind throughout. **They have one target; we have 64 per pointing.** Their multi-beam test asks "does this signal appear *only* in the K2-18b beam?" Ours asks "in how many of the beams formed does it appear?" — because every one of our beams is somebody's target. Same physics, different bookkeeping.

---

## The recipe, in order

```
hits from seticore (all epochs, one receiver)
   │
   1. remove known RFI bands          observatory tables + empirical masks
   2. drift-rate filter               drop exactly zero; drop beyond ±max
   3. SNR window                      keep 10 ≤ SNR ≤ 100
   │
   ├── split: target beam / other beams / incoherent sum
   │
   4. multi-beam coincidence          ±1 fine channel, ±1 drift step
   5. primary & secondary transits    signal must vanish behind the star
   6. multi-day comparison            same freq AND drift on two days = RFI
   7. coherent/incoherent ratio       SNR_coh ≤ √N_ant × SNR_incoh
   8. visual inspection               look at the dynamic spectrum
```

Steps 1–4 and 6–8 are what we implement. Step 5 needs a transiting planet, which we do not have.

**Why each cut is there:**

1. **Known RFI.** Observatory-published tables of persistent interference — NRAO's for the VLA, SARAO's for MeerKAT. They note SARAO's is not regularly updated, which is exactly the gap we hit in S band.
2. **Drift.** A real signal from another world must Doppler-drift, because the transmitter and the telescope are both accelerating. Exactly-zero drift means the emitter is stationary relative to us, i.e. local. There is also an *upper* limit: only so much drift is physically plausible given the planet's orbit and Earth's rotation. Both bounds are cuts.
3. **SNR.** They injected synthetic signals and found that **nearly 80% of detections at 8σ with few time samples were false positives** — random bright pixels aligning in the de-drifted spectrum. At the other end, they inspected 1,000 stamps and found **over 90% of high-SNR detections were instrumental**, appearing in a single antenna at up to 3000σ. Hence the window, not just a floor.
4. **Multi-beam.** A signal on the sky lands in one beam. RFI floods the field.
5. **Transits.** A transmitter on the planet must go quiet when the planet goes behind the star. A beautiful test, unavailable to us.
6. **Multi-day.** A signal from a moving transmitter observed from a moving Earth traces a sinusoid in frequency over time, so it *cannot* appear at the same frequency and drift on two different days. If it does, it is terrestrial.
7. **Coherent/incoherent.** A coherent beam sums N antennas in phase; the incoherent sum does not. A sky signal in both must obey SNR_coh ≤ √N × SNR_incoh. Anything that violates it is not coming from where the beam points.
8. **Look at it.** They plot the dynamic spectrum of every survivor. All four examples they show turned out to be RFI or instrument artefacts.

---

## What they actually found

| Step | MeerKAT UHF | MeerKAT L | MeerKAT S4 |
|---|---:|---:|---:|
| initial hits | 59,030 | 398,867 | 4,712 |
| after RFI masks | 59,030 | 118,677 | 4,712 |
| after drift filter | 5,902 | 58,734 | 1,488 |
| after SNR window | 2,991 | 13,386 | 1,103 |
| in the K2-18b beam | 46 | 225 | 180 |
| **unique to that beam** | **0** | **0** | **0** |

Two things in that table are worth pausing on.

**The RFI masks removed nothing at all in UHF and S4.** Not a small amount — zero. The published SARAO tables simply did not cover the frequencies where their hits were. We independently ran into the same wall: SARAO's table flags 3 hits out of 38,576 in our S band, which is why we added ITU allocations of our own and built `--derive-mask`.

**The drift filter is doing most of the work.** UHF drops by 90% at that step, with ≈83% of hits at exactly zero drift. Our files run 22–47% zero-drift. Theirs is a five-minute snapshot of one field; ours is a survey. Different exposure to the same interference.

---

## Three things worth knowing about the paper

**The incoherent beam was not available on MeerKAT.** Every MeerKAT column marks step 3.7 "N/A". This independently confirms what the BLUSE team told us directly: `incoherentPower` was never measured for our data either. The single strongest discriminant in the recipe — the one that separates "in the beam" from "on the sky" — is unavailable to us, which puts the whole weight on multi-beam coincidence.

**The drift limits in §3 and the drift limits actually applied in §4 do not match.** §3.2 prescribes a frequency-dependent bound: about 0.4 Hz/s below 1.5 GHz, 1.879 Hz/s at 4.5 GHz, 4.177 Hz/s at 10 GHz. But §4 applies **±1.9 Hz/s to L band, S band and C band alike** — three bands where the prescription gives values from 0.4 to 3.0 — and only X band uses its own number. Anyone reproducing this has to choose. We follow §3.2, because it is the physically motivated version and the one the text argues for.

**They are candid about the limits.** Conservative masking can hide real signals. RFI models are necessarily incomplete. The search only covers approximately linear drift. And a null result over a handful of orbital phases rules out a specific class of transmitter, not habitability.

---

## What we do differently, and why

| | Tremblay et al. | Our Track A |
|---|---|---|
| Targets | one (K2-18b) | ~3,400–4,000 Gaia sources per file |
| Multi-beam test | "unique to the target beam?" | "seen in ≤4 of the beams formed?" |
| SNR ceiling | 100 | 10⁶ by default — our data has a detached population at 10⁷–10⁸ that we want to *see*, not silently drop |
| Transit filter | yes | not applicable |
| Multi-day | one target across epochs | frequency bins recurring across ≥5 observations |
| Incoherent | used on VLA, N/A on MeerKAT | implemented, permanently inert |
| Nothing deleted | filtered dataframes | every cut writes a `flag_*` column; `pass_all` is their AND |

That last row is our main departure and we think it is an improvement. Their pipeline narrows a dataframe at each step. Ours records what each cut *would* remove and leaves the data intact, so you can inspect any cut, disagree with it, and re-decide by re-filtering a parquet rather than re-running.

---

## Glossary

- **Hit** — one narrowband detection in one beam.
- **Coherent beam** — antenna signals summed in phase toward a specific direction. Sensitive, and localised.
- **Incoherent sum** — antenna powers added without phase. Covers the whole field of view, √N less sensitive, and cannot localise.
- **Drift rate** — how fast a signal's frequency slides, in Hz/s, from relative acceleration between transmitter and receiver.
- **Drift step** — the drift resolution of the search, Δν/T_obs. Detections of the same signal can differ by about one step.
- **Primary / secondary transit** — planet in front of the star / behind it. During secondary transit a planetary transmitter must be invisible.
- **EIRP** — equivalent isotropic radiated power: how strong a transmitter radiating equally in all directions would have to be for us to have seen it.
- **COSMIC** — the VLA's commensal SETI backend. BLUSE's sibling system.
- **seticore** — the de-drifting search software both backends run. It produces our hits and stamps.
