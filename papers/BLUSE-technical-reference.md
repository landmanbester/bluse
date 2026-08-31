# BLUSE Technical Reference

**Purpose:** dense, agent-oriented context on the architecture, data flow, software components, parameters, and operating characteristics of the BLUSE (Breakthrough Listen User Supplied Equipment) commensal technosignature survey at MeerKAT.

**Primary source:** Czech et al. 2026, *"Breakthrough Listen's Automated Commensal Technosignature Survey with MeerKAT"*, MNRAS preprint, arXiv:2607.23651v1 (11 pp., 28 July 2026). All figures/tables cited below are from that paper. Where this document says "current", it means as of that preprint.

---

## 1. System identity and scope

- **What it is:** an autonomous, always-on, commensal beamformer + narrowband technosignature search system co-located with MeerKAT.
- **Commensal means:** BLUSE subscribes to MeerKAT's internal multicast F-engine data streams. It consumes no telescope time and does not affect the primary observer. It inherits whatever the primary observer chooses: receiver band, F-engine mode, pointing, antenna allocation, and calibration.
- **Operational since:** mid-2022, autonomous.
- **Owner/affiliation:** Breakthrough Listen (Berkeley SETI Research Center, University of Oxford, SETI Institute), hosted by SARAO.
- **Sibling system:** COSMIC at the VLA (Tremblay et al. 2023) — the same commensal-array paradigm.

### Core operating mode (one sentence)
Ingest F-engine coarse channels → buffer to NVMe → upchannelise to ~1 Hz → synthesize 64 coherent beams + 1 incoherent beam on catalogued targets in the primary field of view → Taylor-tree de-Doppler drift search on each beam → write hits/stamps/filterbanks.

---

## 2. MeerKAT context

- 64 antennas, 13.5 m offset Gregorian, Karoo desert, South Africa (Jonas & MeerKAT Team 2016).
- Three receivers spanning 544 MHz – 3.5 GHz.
- **FX correlator architecture:** channelisation in the F-engine, cross-correlation in the X-engine.
- **Packetised Ethernet cornerturn:** like channels from all antennas are grouped into a packet destined for one X-engine. IPv4 multicast allows multiple digital backends to subscribe to the F-engine streams simultaneously.
- **USE (User Supplied Equipment):** third-party backends subscribe to the same multicast groups and run alternative signal processing in parallel with the correlator (cf. Sanidas et al. 2017; Bailes et al. 2020).
- Access protocol: **SPEAD2** (Streaming Protocol for Exchanging Astronomical Data), an implementation of SPEAD with Python and C++ bindings — `https://github.com/ska-sa/spead2` (Manley et al. 2010).
- MeerKAT supports **simultaneous independent subarrays**, each a subset of antennas with its own band, F-engine mode, pointings and calibrators. BLUSE must handle all subarray configurations concurrently.

---

## 3. Receivers, modes, and data rates (Table 1)

BLUSE ingests the **coarse channels** emitted by the F-engines. Wide-band modes always available: **1k / 4k / 32k** coarse channels (chosen by the primary observer). A narrowband **"zoom" mode** exists but BLUSE does **not** use it (BLUSE must upchannelise across the entire band).

S-band receivers offer more bandwidth than the F-engines can process, so the S band is split into **five overlapping 875 MHz subbands (S0–S4)** covering 1750–3500 MHz.

| Receiver | Digitised BW (MHz) | Digitised range (MHz) | F-engine mode | Coarse chan BW (kHz) | Fine chan BW (Hz) | Gbps/instance |
|---|---|---|---|---|---|---|
| UHF | 544 | 544–1088 | 1k | 531.25 | 1.01 | 17.408 |
| UHF | 544 | 544–1088 | 4k | 132.81 | 1.01 | 17.408 |
| UHF | 544 | 544–1088 | 32k | 16.6 | 1.01 | 17.408 |
| L | 856 | 856–1712 | 1k | 835.94 | 1.59 | 27.392 |
| L | 856 | 856–1712 | 4k | 208.98 | 1.59 | 27.392 |
| L | 856 | 856–1712 | 32k | 26.12 | 1.59 | 27.392 |
| S | 875 | 1750–3500 (5 subbands) | 1k | 854.49 | 1.62 | 28.0 |
| S | 875 | 1750–3500 (5 subbands) | 4k | 213.62 | 1.62 | 28.0 |
| S | 875 | 1750–3500 (5 subbands) | 32k | 26.7 | 1.62 | 28.0 |

### Data-rate arithmetic (L-band worked example)
- Single antenna: `856 MHz × 8 bits × 2 (complex) × 2 (polarisations) = 27.392 Gbps`.
- Full 64-antenna array: `≈ 1.753 Tbps`.
- Network overhead: per 1024 bytes of payload, Ethernet/IP/UDP headers add **42 bytes**, SPEAD2 header adds **96 bytes**.
- **Actual received rate per processing instance at L-band with 64 antennas: ≈ 31.08 Gbps.**

### Sharding model (important)
Each processing pipeline instance receives an amount of data **equivalent to one MeerKAT antenna**, but it is not one antenna's data — it is **1/64th of the bandwidth from every antenna**. Each instance is therefore responsible for upchannelising, beamforming and searching 1/64th of the full band across all antennas.

**Network constraint:** an instance connected to a particular leaf switch may only subscribe to consecutive multicast groups corresponding to **1/4 of the full band**. This constrains how the coordinator may allocate instances.

---

## 4. Processing pipeline (Fig. 1)

```
Telescope ──packetised coarse channels──► packet ingestion ──► NVMe buffer
                                               │
                                               ▼
                          upchannelisation ──► beamforming ──► technosignature search ──► postprocessing
                                                                                              │
control plane (dotted):  telstate/katcp/katportal interfaces ──► coordinator ◄──► target selector ◄──► target DB
                                                     │                    └──► "recipe" generator
                                                     └── processing results
```

### 4.1 Ingestion
- SPEAD2 packets received by **`hpguppi_daq`** instances — `https://github.com/UCBerkeleySETI/hpguppi_daq`.
- Payload voltages assembled into **GUPPI raw format** (Ford & Ray 2010; Lebofsky et al. 2019).
- Written to **NVMe in RAID 0** for buffer depth far beyond what RAM allows.
- Drive endurance was validated with purpose-built software **`disk_hammer`** (`https://github.com/david-macmahon/disk_hammer`): sequential write-to-capacity, read back, delete. Observed longevity far exceeded stated specifications for this linear write/read/delete pattern.

### 4.2 Calibration
- Coherent beamforming needs phasing solutions. BLUSE **retrieves and relies on the calibration solutions produced for the primary observer** by MeerKAT's Science Data Processor.
- Latest solutions fetched from **TelState** (`https://github.com/ska-sa/katsdptelstate`) at the start of a recording and supplied to the beamformer.
- MeerKAT calibration pipelines documented in the MeerKAT External Service Desk Knowledge Base.

### 4.3 Upchannelisation and beamforming
- Upchannelise coarse channels to **≈1 Hz** resolution (exact per-band values in Table 1: 1.01 / 1.59 / 1.62 Hz).
- Form **64 coherent beams + 1 incoherent beam per 290 s primary pointing**.
- Beam pointings and calibration solutions are packaged into an **HDF5 "beamformer recipe file" (BFR5)** by the **`bfr5_generator`** process.
- `bfr5_generator` also computes **delays and delay rates for each beam** relative to the boresight pointing (the F-engines track boresight in delay and phase), at **~1-second intervals** over the recording duration, and writes them into the recipe file for the beamformer.
- Both upchannelisation and beamforming are performed by **`seticore`** — `https://github.com/lacker/seticore`.

### 4.4 Technosignature search
- Algorithm: **Taylor-tree narrowband de-Doppler drift search** (Taylor 1974; Sheikh et al. 2019; Siemion et al. 2013), executed by `seticore` on the beamformed data.
- `seticore` includes a high-performance GPU implementation of the search algorithm found in **turboSETI** (Enriquez et al. 2017).
- Rationale for Taylor-tree: highest throughput among widely used algorithms. Alternatives noted but not deployed: fast-folding for pulsed signals (Suresh et al. 2023), variational autoencoders (Ma et al. 2023).
- **Drift-rate range: ±10 Hz/s**, with stated intent to widen. Context: Li et al. (2023) find **±44 Hz/s** would cover 99% of potential signals arising from transmitters on exoplanet surfaces. Comparable surveys: ±2 (Enriquez 2017), ±4 (Price 2020; Choza 2023), ±8.86 (Margot 2023), ±50 Hz/s (Tremblay 2023). Wider drift search costs compute throughput.
- **SNR threshold: 6** (initial value). This threshold directly sets the rate at which "hits" are recorded and hence storage consumption; it is expected to be tuned during operations.
- **Why ~1 Hz matters:** extremely narrowband signals cannot be produced by known natural astrophysical processes, and their frequency drifts under acceleration (Li et al. 2022). The narrowest known natural line is an OH maser at ~550 Hz (Cohen et al. 1987). Narrowband is also energy-efficient for interstellar transmission.

### 4.5 Data products
| Product | Description |
|---|---|
| **Hits** | Compact records of candidate detections above the SNR threshold. |
| **Stamps** | When a signal is detected in a coherent beam, a stamp file saves an **upchannelised raw voltage swatch of the time-frequency region around the detection, from every participating antenna**. Enables offline re-beamforming anywhere in the primary FoV and repeated further analysis. Example: Fig. 11 (62 antennas, JWST downlink). |
| **Filterbank HDF5** | At a user-specified interval, filterbank-formatted HDF5 files are saved for the incoherent sum of all antennas and for each coherent beam. Used for system validation and diagnostics. |
| **Raw F-engine voltages** | Saved in full **only when BLUSE is used for direct primary observing**, for later re-analysis. |

Stamp volume is controlled by a configurable "dial" for how many stamp files to save per recording (in addition to the SNR threshold), plus a **maximum SNR threshold** to keep RFI-dominated regions of the band manageable.

**Design rationale:** at array scale it is infeasible to store all raw F-engine voltages in perpetuity (single-dish surveys sometimes can, having far fewer receiver pixels — Lebofsky et al. 2019). Stamps are the middle ground: full per-antenna information, but only around detections.

### 4.6 Storage and archiving
- Survey products are stored and processed **in situ** on the BLUSE cluster.
- During a subarray's lifetime, products are written to **temporary volumes on each processing node**: magnetic disks in **RAID 0** for write speed, minimising time spent transferring data so as not to delay subsequent recording cycles.
- **Between active subarrays** (when no multicast subscriptions exist), data are copied to storage nodes for long-term storage in **Gluster volumes backed by RAID 6** disk arrays (survives two concurrent physical disk failures).

---

## 5. Automation and control (Section 3)

### 5.1 State machines (Fig. 2)
**Two state machines per subarray.** Purpose: correct resource allocation, correct metadata delivery, and synchronised recording across participating processing instances.

**(a) FreeSubscribed** — allocates/de-allocates processing instances, returning them to the free pool:
```
Free ──Configuring──► Configuring ──Configured──► Subscribed
  ▲          │                                        │
  └──────────┴────────────Deconfigure──────────────────┘
```

**(b) RecProc** — drives instances already allocated and subscribed to a subarray's multicast streams:
```
Ready ──Tracking──► Record ──Recording end──► Process ──Proc. end (not tracking)──► Ready
                       │                          │
                       │                          └──Proc. end (tracking)──► Record
                       └──Tracking end / recording end (primary time)──► Waiting ──Manual reset──► Ready
                                                    Process ──Processing error──► Error
```

**Instance-set semantics (Fig. 3):** there is exactly one shared `free` set across all `freesubscribed` machines. Each subarray's `freesubscribed` and `recproc` machines share a `subscribed` set. State changes are resolved **in sequence**, so each machine takes its turn moving instances between sets.

### 5.2 State preservation
Each subarray's `recproc` and `freesubscribed` states are persisted to **Redis** as **JSON**-formatted dictionaries on **every state change**. On startup the coordinator checks for saved state and re-initialises each machine with the correct allocation, recreating the last known configuration. This allows recovery from interruptions and mid-observation modification of an active subarray.

### 5.3 Pipeline control across hosts (Fig. 4)
```
headnode:   coordinator
                │ zmq
hosts:      blpn0 ... blpn64 ... blpnN
                │ circusctl
instances:  bluse_analyzer_0 [, bluse_analyzer_1]
                └──────────────── redis ────────────────► coordinator
```
- Central **`coordinator`** controls each processing pipeline instance via **circus** and **ZeroMQ**.
- **circus** also controls all ancillary processes and manages log rotation on every host in the cluster.
- Note: some processing nodes host a **single** pipeline instance, others host **two** (see Table 3 / §7).

### 5.4 Automation processes (Table 2)
| Process | Role | Repo |
|---|---|---|
| `katcp_interface` | BLUSE proxy; responds to MeerKAT CAM (Control and Monitoring) requests when BLUSE is included in a subarray, signalling readiness to receive data. | — |
| `katportal_interface` | Metadata retrieval; connects to CAM via websockets for current pointing, centre frequency, on-target status. Continuous/instantaneous updates. | — |
| `coordinator` | Automates commensal observing; owns the per-subarray state machines; allocates and controls recording and processing across the cluster. | `https://github.com/UCBerkeleySETI/commensal-automator` |
| `bfr5_generator` | Creates BFR5 beamformer recipe files per observation (delays, delay rates, calibration, beam coordinates). | `https://github.com/david-macmahon/BluseBeamformerRecipes.jl` |
| `slack_proxy` | Delivers Slack messages (head node). Other processes bridge to it via Redis. | — |
| `targets_minimal` | Selects targets for observation and analysis (head node). | `https://github.com/danielczech/targets-minimal/` |
| `bluse_raw_watcher` | Maintains awareness of NVMe buffer contents. One per processing node. | `https://github.com/david-macmahon/BluseRawWatch.jl` |
| `bluse_analyzer` | Controls processing. One per instance, on processing nodes. | — |
| `bluse_gateway` | Hashpipe–Redis gateway process. | `https://github.com/david-macmahon/rb-hashpipe` |

**Metadata routing constraint:** metadata associated with a particular subarray must be confined to, and delivered to, only the processing nodes participating in that subarray.

### 5.5 Target selection (Section 3.5)
Handled by **`targets_minimal`** on the head node. The coordinator sends requests **via Redis** with the current observing band and the coordinates of the current primary pointing. `targets_minimal` estimates the **primary beam width at half maximum power**, retrieves all catalogue targets falling within it, and ranks them.

**Score update** — after a target is observed and successfully processed in a band, its score for that band is updated additively:

```
S'_band = S_band + t × b × n            (Eq. 1)
```
where `S_band` = previous score, `t` = duration of the current observation, `b` = number of subband segments, `n` = number of antennas.

**Ranking (ascending — lowest score first):**
1. Targets observed the least in the **current band**.
2. Targets observed the least in **all other bands combined**.
3. **Distance** (nearer preferred).

**Catalogue:** ~**32 million stars** (all-sky), derived from **Gaia DR2** and other sources — selection described in Czech et al. (2021). Plus a provisional extended **"semi-Exotica" sample of ~2 million** other objects (galaxies, AGNs, stars), an expanded counterpart to the Breakthrough Listen exotica catalogue (Lacki et al. 2021), integrated into BLUSE's target list.

### 5.6 Monitoring
- Status/updates via **Slack**.
- **Grafana + Prometheus** for system health: NVMe write rates, CPU usage, etc.
- Automation processes transmit **annotations** to Grafana, displayed on plots (e.g. recording-start time for a given observation).
- **Daily observing summaries emailed** to system maintainers.

---

## 6. Validation (Section 5)

**Test source problem:** *Voyager* has been the canonical SETI test transmitter (Welch et al. 2009; Tremblay et al. 2023) but MeerKAT has no X-band (8–12 GHz) receiver.

**Solution:** observe the **James Webb Space Telescope's S-band telemetry downlink** with MeerKAT's S-band receivers. JWST is a good narrowband technosignature analogue: it moves slowly in RA/Dec and stays well within the primary FoV over a 290 s recording.

**Setup:** fixed pointing at fixed RA/Dec in JWST's path; **S0 subband, 4k mode**; 290 s recording; coarse-channel raw voltages preserved **in addition to** the standard data products, allowing arbitrary offline beam placement after the fact.

**Results:**
- Fig. 7: tiled coherent beams at fixed sky coordinates, synthesized offline, showing JWST's transit from "Start" to "End" markers at t = 9.8 s and t = 285.1 s.
- Fig. 8: power vs time for several fixed-coordinate beams positioned at JWST's location at t = 0, 120, 300 s — each peaks as the source passes through.
- Fig. 9: corroborating 8 s snapshot images from the MeerKAT **correlator** (0.3 × 0.3 deg² field; 1.8 × 1.8 arcmin² zoom), confirming JWST's position at the start and end of the 290 s segment.
- Fig. 11: the automatically saved **stamp file** — time-frequency panels for each of the **62 participating antennas** around 2270.511230 MHz, both polarisations summed, with the drifting downlink clearly visible in every antenna.

**Beamforming efficiency** (procedure of Tremblay et al. 2023):
```
η = (P_coherent / P_incoherent) × (1 / N_ants)      (Eq. 2)
```
with powers normalised as in Tremblay et al. (2022).
- Measured `P_coherent / P_incoherent ≈ 52.63`
- `N_ants = 62` → **η ≈ 84.9%**
- Shortfall from theoretical maximum attributed to small phase calibration errors and other effects.
- Relative to Rajwade et al. (2022), who report 0.92–0.96 at L-band by a different method, BLUSE's efficiency is **≈ 0.92**.

---

## 7. Hardware (Section 4, Table 3)

Hosted in **16 racks** in the **Karoo Array Processor Building (KAPB)**, operated by SARAO.

| Node type | Qty | Specification |
|---|---|---|
| Processing (AMD) | 37 | 32 active, 4 hot spares, 1 experimental. 2× AMD EPYC 7413, 512 GB RAM, 4× RTX A4000 (active nodes) / 2× RTX 3090 (spare nodes), 16× 1 TB HDDs, 8× 1 TB NVMe, 2× 100 GbE ConnectX-5 NIC. |
| Processing (Intel) | 68 | 64 active, 4 hot spares. 2× Intel Xeon 4208, 96 GB RAM, 1× RTX 2080Ti, 6× 8 TB HDDs, 4× 512 GB NVMe, 1× 40 GbE ConnectX-5 NIC each. |
| Storage (version a) | 4 | 2× Xeon 4208, 96 GB RAM, 36× 8 TB HDDs, 50 GbE ConnectX-5 NIC. |
| Storage (version b) | 4 | 2× Xeon 4309Y, 192 GB RAM, 36× 16 TB HDDs, 50 GbE ConnectX-5 NIC. |
| Head node | 1 | 2× Intel Xeon E5-2620, 128 GB RAM, 4× 960 GB SSDs, 2× 600 GB SSDs. |

- **96 active processing nodes** (32 AMD + 64 Intel) — consistent with the 64-way band sharding plus multi-instance hosts.
- Fig. 5: AMD node internals — 4× RTX A4000 GPUs, 2 NVMe carrier cards, NIC.
- Fig. 6: 8 of the 16 racks in situ. Servers physically closer to the 40 GbE switches use **copper** interconnect; those further away use **fibre**.

---

## 8. Observing progress (Section 6)

| Period | Coherent beams processed |
|---|---|
| mid-2022 → mid-2023 | ~73,500 |
| mid-2023 → mid-2024 | ~386,200 (≈5× increase, from automation software improvements) |
| mid-2024 → mid-2025 | ~436,700 |
| 2023 → 2026 (total) | **~1.5 million** |

- Of ~1.5 M coherent beams, **~1.2 M were viable** for technosignature searching; the remainder were **too short (< 150 s)**.
- Of those ~1.2 M, **~360,000 unique objects** were observed — many objects observed multiple times where the same primary pointing coordinates were revisited.
- Current rate: **~29,000 globally unique objects per month** (i.e. never previously observed by BLUSE), at **290 s per pointing**.
- The paper's abstract/conclusions phrase the same ~1.2 M figure as "individual pointings, nearly all of which were 290 s in duration."
- **Sky coverage (Fig. 10):** primary FoVs to scale for every processed pointing in UHF, L and S. Dense coverage along the Galactic plane; visible tiling campaigns over the **Euclid Deep Field (South)** and the **Virgo Cluster**; strongly southern-weighted.
- Data products are under analysis for a series of forthcoming publications, including the distribution of detected signals vs observing frequency, and initial survey results toward **K2-18** (Tremblay et al. 2026, ApJ 171, 210).

---

## 9. Non-commensal and adjacent uses

- BLUSE has also been used **for and alongside primary time observations**, including a campaign on **K2-18** and observations of interstellar comet **3I/ATLAS** (in prep.).
- Student projects are supported as **pipeline components controlled via the `analyzer` processes** — e.g. beam-tiling observations of galaxy clusters when they fall within the primary FoV.

---

## 10. Roadmap / future work (Section 7)

- **Widen the drift-rate range** beyond ±10 Hz/s (target informed by the ±44 Hz/s / 99% figure of Li et al. 2023).
- **More sophisticated search algorithms** in tandem with `seticore`, explicitly including **`bliss`** (`https://github.com/n-west/bliss`) and **interferometric imaging-based** detection approaches.
- **Hardware upgrades**, some of which would be accompanied by architectural changes to the pipeline.
- **GPUDirect RDMA** to bypass the CPUs and perhaps the NVMe recording buffers entirely, increasing performance (cf. Allen Telescope Array work, Ma et al. 2025).
- Tuning of the **SNR threshold** and the stamp-saving "dial" during operations, as these directly govern storage consumption.

---

## 11. Quick-reference constants

```
Antennas                      64 (13.5 m offset Gregorian)
Bands                         UHF 544–1088 | L 856–1712 | S 1750–3500 MHz (5×875 MHz subbands S0–S4)
F-engine modes used           1k, 4k, 32k coarse channels (NOT zoom mode)
Fine channel resolution       ~1 Hz (UHF 1.01 | L 1.59 | S 1.62 Hz)
Recording length              290 s per pointing
Beams per pointing            64 coherent + 1 incoherent
Beamformer efficiency         ~84.9% (η, Eq. 2); ~0.92 relative to Rajwade et al. 2022
Search algorithm              Taylor-tree de-Doppler (seticore; turboSETI-equivalent)
Drift-rate range              ±10 Hz/s
SNR threshold                 6
Band sharding                 each instance = 1/64th of band, ALL antennas
Per-instance ingest (L, 64 ant)  ~31.08 Gbps (incl. overhead); 27.392 Gbps payload
Full-array raw rate (L)       ~1.753 Tbps
Delay/phase update interval   ~1 s (written into BFR5)
Target catalogue              ~32 M stars (Gaia DR2, Czech et al. 2021) + ~2 M semi-Exotica
Active processing nodes       96 (32 AMD EPYC + 64 Intel Xeon); 8 storage; 1 head
Racks                         16, in the KAPB
Node-local scratch            RAID 0; long-term Gluster on RAID 6
Autonomous since              mid-2022
Cumulative beams (2023–2026)  ~1.5 M (~1.2 M viable, ≥150 s); ~360k unique objects
Current unique-object rate    ~29,000 / month
```

## 12. Key file/format vocabulary

| Name | Meaning |
|---|---|
| **SPEAD2 packet** | Transport format on the MeerKAT multicast network. |
| **GUPPI raw** | On-disk format for buffered voltage data written by `hpguppi_daq`. |
| **BFR5** | HDF5 beamformer recipe file: beam coordinates, calibration solutions, per-beam delays and delay rates at ~1 s cadence. Produced by `bfr5_generator`, consumed by `seticore`. |
| **Hit file** | Candidate narrowband detection record from the drift search. |
| **Stamp file** | Per-antenna upchannelised time-frequency swatch around a detection; supports offline re-beamforming. |
| **Filterbank HDF5** | Periodic diagnostic/validation dumps for the incoherent sum and each coherent beam. |
| **TelState** | MeerKAT telescope state store from which calibration solutions are retrieved. |

## 13. Component index (repos)

```
spead2                  https://github.com/ska-sa/spead2
hpguppi_daq             https://github.com/UCBerkeleySETI/hpguppi_daq
seticore                https://github.com/lacker/seticore
commensal-automator     https://github.com/UCBerkeleySETI/commensal-automator
BluseBeamformerRecipes  https://github.com/david-macmahon/BluseBeamformerRecipes.jl
targets-minimal         https://github.com/danielczech/targets-minimal/
BluseRawWatch           https://github.com/david-macmahon/BluseRawWatch.jl
rb-hashpipe             https://github.com/david-macmahon/rb-hashpipe
disk_hammer             https://github.com/david-macmahon/disk_hammer
katsdptelstate          https://github.com/ska-sa/katsdptelstate
bliss (future)          https://github.com/n-west/bliss
```

## 14. Caveats for downstream reasoning

- All figures are **as of the July 2026 preprint**; drift range, SNR threshold, node counts and stamp policy are explicitly described as tunable/evolving.
- BLUSE **does not control the telescope** in commensal mode: band, mode, pointing, antenna count and calibration quality are all inherited, and vary observation to observation. Search sensitivity and beam efficiency therefore vary per observation.
- The ~360,000 "unique objects" figure counts distinct catalogue entries; the ~1.2 M figure counts beam-pointings including repeats.
- Beams shorter than **150 s** are discarded as non-viable for searching.
- Data underlying the paper are available **on request to the corresponding author** (daniel.czech@physics.ox.ac.uk); no public archive is stated.
