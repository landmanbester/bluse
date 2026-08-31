# BLUSE: Breakthrough Listen's Automated Commensal Technosignature Survey with MeerKAT

**A high-level overview for human readers**

Source: Czech, D. J., MacMahon, D. H. E., Heywood, I., Tremblay, C. D., Lebofsky, M., Lacker, K., Ng, C., Horn, D., Buchner, S., Lacki, B., Andersson, A., Bright, J. S., Croft, S., DeBoer, D. R., Drew, J., Gajjar, V., Ma, P., Pollak, A. W., Price, D., Ruzindana, M., Siemion, A. P. V., Worden, S. P., Camilo, F. — *"Breakthrough Listen's Automated Commensal Technosignature Survey with MeerKAT"*, MNRAS (preprint, 28 July 2026), arXiv:2607.23651v1.

---

## The one-paragraph version

BLUSE (Breakthrough Listen User Supplied Equipment) is a large, fully automated computing system installed alongside the MeerKAT radio telescope in South Africa. It quietly taps into the telescope's internal data streams while other astronomers do their own science, and — without needing any telescope time of its own — searches the sky for narrowband radio signals of the kind that a technological civilisation might produce. It has been running autonomously since mid-2022, has processed roughly 1.5 million synthesized beams (~1.2 million usable for searching), and now covers about **29,000 stars never previously observed by BLUSE, every month**.

---

## Why this matters

Traditional SETI (search for extraterrestrial intelligence) surveys used single large dishes pointed at one target at a time. That approach has three problems:

1. **It's slow.** One target per pointing means surveys top out in the thousands to tens of thousands of objects.
2. **It's expensive.** Sensitive dishes are in high demand, and SETI competes with mainstream astronomy for time.
3. **It's vulnerable to interference.** A single dish can't easily tell a genuine sky signal from a mobile phone tower, so observers have to burn extra time on "on-source / off-source" cadences to check.

An interferometric **array** like MeerKAT fixes all three at once. Its 64 dishes are individually small, so each has a wide field of view containing many potential targets; the array can be electronically re-pointed (beamformed) at dozens of targets simultaneously *within* that field; and because the signal must appear coherently across widely separated antennas, local interference is naturally suppressed.

Crucially, MeerKAT was designed with a **multicast Ethernet architecture**: the digitised data from every antenna is broadcast on the internal network, and third-party "user supplied equipment" is allowed to subscribe to those streams. BLUSE takes advantage of this to run in parallel with — and completely invisibly to — whatever the primary observer is doing.

---

## How it works, in plain terms

```
MeerKAT antennas → digitisers → F-engines (channelisers)
                                     │
                    multicast Ethernet (the "corner turner")
                                     │
                   ┌─────────────────┴─────────────────┐
             MeerKAT correlator                      BLUSE
             (the primary observer's                 (commensal
              science)                                technosignature search)
```

1. **Listen in.** BLUSE subscribes to the multicast groups carrying the telescope's channelised antenna voltages. At L-band the full array produces about **1.75 Tbps**; BLUSE's ~96 processing servers each take a 1/64th slice of the band from *all* antennas.

2. **Buffer to disk.** Incoming packets are written straight to NVMe solid-state drives at very high rates (~31 Gbps per processing instance), giving BLUSE a much deeper recording buffer than RAM alone would allow.

3. **Zoom in on frequency.** The telescope's own channels are far too coarse for SETI. BLUSE re-channelises ("upchannelises") the recorded data down to roughly **1 Hz resolution**. Signals that narrow cannot be produced by any known natural astrophysical process — the narrowest known natural line, an OH maser, is ~550 Hz wide — so extreme narrowness is itself the technosignature.

4. **Point at many stars at once.** Using calibration solutions borrowed from the primary observer, BLUSE synthesizes **64 coherent beams** (plus one incoherent beam) pointed at catalogued objects that happen to fall inside the telescope's field of view, for each ~290-second pointing.

5. **Search each beam.** A Taylor-tree "de-Doppler" search sweeps each beam for narrowband signals that drift in frequency — the signature of relative acceleration between a transmitter and Earth. BLUSE currently searches drift rates of ±10 Hz/s at a signal-to-noise threshold of 6.

6. **Save the interesting bits.** Detections are recorded as compact "hits", and the most promising ones trigger "stamps" — small slices of raw per-antenna data around the detection, allowing the observation to be re-beamformed and re-analysed later without storing petabytes of raw voltages.

7. **Choose the next targets.** A scheduler continuously re-ranks a catalogue of ~32 million stars (from Gaia DR2) plus ~2 million "exotic" objects, preferring targets that have been observed least in the current band, then least in other bands, then nearest to Earth.

All of this happens with no human in the loop. A central coordinator process tracks each of MeerKAT's independent subarrays with a pair of state machines, allocating processing nodes from a shared pool as subarrays come and go, and recovering its state from a Redis database if it is interrupted mid-observation. Health and progress are reported to Grafana, Prometheus, Slack and a daily email summary.

---

## Does it actually work? The JWST test

Validating a SETI system is awkward: the one thing you'd like to detect is the one thing you don't have. Earlier surveys used the *Voyager 1* spacecraft as a stand-in transmitter, but MeerKAT has no X-band receiver to hear it.

Instead the team pointed MeerKAT at the **James Webb Space Telescope's S-band telemetry downlink** — a genuine, extremely narrowband artificial transmitter, moving slowly enough to remain in the field of view for a full 290-second recording. BLUSE detected it end-to-end and automatically saved a stamp file; the signal is clearly visible in the data from each of the 62 participating antennas.

Because the raw data were preserved, the team could also synthesize a grid of beams offline and watch JWST's transit move across them, confirming the beam shapes and positions. Comparing coherent to incoherent beam power gave a **beamforming efficiency of ~85%** (~0.92 relative to the best efficiency reported elsewhere at L-band) — slightly below theoretical maximum, as expected from small phase calibration errors.

---

## What it has done so far

| Period | Coherent beams processed |
|---|---|
| mid-2022 → mid-2023 | ~73,500 |
| mid-2023 → mid-2024 | ~386,200 (a five-fold increase) |
| mid-2024 → mid-2025 | ~436,700 |
| **2023 → 2026 total** | **~1.5 million** |

Of the ~1.5 million beams, ~1.2 million were long enough (≥150 s) to be searched, covering roughly **360,000 unique objects** — many observed repeatedly, which is scientifically valuable in its own right. Sky coverage is broad and southern-weighted, with dense coverage along the Galactic plane and clusters of repeat visits wherever the primary observers concentrated (the Euclid Deep Field South, the Virgo Cluster, and so on).

Results are being written up in a series of forthcoming papers, including initial survey results toward the K2-18 system.

---

## The scale of the hardware

BLUSE occupies **16 racks** in the Karoo Array Processor Building next to the telescope:

- **~96 active processing servers** (32 AMD EPYC nodes with 4× RTX A4000 GPUs each; 64 Intel Xeon nodes with an RTX 2080Ti each), plus hot spares
- **8 storage servers** holding 36 large hard drives apiece, in RAID 6, presented as Gluster volumes
- NVMe arrays on every processing node for high-speed capture, later drained to long-term storage between subarray activity

---

## Why this is the future of SETI

The central argument of the paper is one of **rate and cost**. A commensal array survey doesn't compete for telescope time — it rides along. Every hour any observer uses MeerKAT is an hour of SETI coverage. That turns a survey of thousands of stars into a survey of hundreds of thousands, with repeat visits, for the marginal cost of the computing hardware.

BLUSE, along with COSMIC at the VLA, represents a step change in how fast the "cosmic haystack" of possible signals can be searched. Planned improvements include wider drift-rate coverage, additional detection algorithms beyond the classic Taylor-tree search, interferometric imaging-based detection, and GPUDirect RDMA to bypass CPUs and possibly the NVMe buffers entirely.

---

## Glossary

| Term | Meaning |
|---|---|
| **Technosignature** | Observable evidence of technology, as opposed to a biosignature (evidence of life's chemistry). |
| **Commensal** | Running in parallel with, and without disturbing, another observation. |
| **F-engine** | The first stage of the correlator; splits each antenna's signal into frequency channels. |
| **Upchannelisation** | Re-processing coarse channels into much finer ones (here, ~1 Hz). |
| **Coherent beam** | A sensitive, narrow beam formed by adding antenna signals with the correct phase delays for one sky direction. |
| **Incoherent beam** | The simple summed power of all antennas — less sensitive, but covering the whole field of view. |
| **Drift rate** | The rate at which a signal's frequency changes due to relative acceleration (Doppler drift). |
| **Subarray** | A subset of MeerKAT antennas operating as an independent telescope. |
| **Hit / Stamp** | A compact record of a candidate detection / a saved slice of raw per-antenna data around it. |
