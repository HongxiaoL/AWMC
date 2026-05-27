# Investigating Related Work in AI for Materials

**Date:** Today
**Subject:** Related-work investigation for `~/AWMC/result/final_report.md` (Causal Failure Report for Universal MLIPs)

***

## User

> Investigate related work in AI for materials field. Find research evidences on and against the conclusions in the report `~/AWMC/result/final_report.md`.

## Assistant — Initial action

Listed `~/AWMC/result/` and read `final_report.md` (471 lines).

### Summary of report conclusions investigated

The report is a causal failure analysis across 9 universal MLIPs (equflash, equiformer, esen, grace, matris, nequip, pet, sevennet, tace) on WBM/Matbench-Discovery. Key conclusions:

1. **`unit_cell_density_estimate`** is the universal #1 failure driver (9/9 models). Mechanism: heavy elements + steep PES + sparse training data.
2. **`average_bond_ionicity_index`** is the #2 failure driver (8/9 models). Mechanism: local message-passing cannot capture long-range Coulomb / Madelung.
3. **`local_strain_misfit_parameter`** raises error for equivariant models (equflash, equiformer, esen, nequip, sevennet) but is neutral/beneficial for tace, pet, matris.
4. **`crystal_system_homogeneity`** and **`lattice_anisotropy_ratio`** mostly REDUCE error — claimed mechanism: highly symmetric/isotropic WBM structures are unrelaxed and OOD; anisotropic structures have simpler relaxation pathways.
5. **`point_group_order`** (equflash) and **`space_group_number`** (matris) cause massive failures (>1 eV/atom). Mechanism: unrelaxed high-symmetry prototypes.
6. **`polyhedral_volume_distortion_index`** REDUCES error in 5/9 models — distorted = closer to relaxed ground state.
7. **PET** is the most robust model overall.
8. Methodology: DML ATE + paired-difference matching.

## User

> You can NOT modify any file in the directory. As for output, just tell me in this chat.

## Assistant

Dispatched 6 parallel research agents covering:

1. Long-range electrostatics / ionicity claim
2. Heavy-element / density failure claim
3. WBM unrelaxed prototypes / symmetry / distortion claims
4. PET model robustness vs. TACE/GRACE/MatRIS
5. Symmetry-breaking distortions (Jahn-Teller, off-centering)
6. DML / causal-inference methodology

***

# Consolidated Literature Evidence

## Conclusion 1 — "Unit cell density is the universal #1 failure driver (9/9 models): heavy elements + steep PES"

**Supports (mechanism well-attested):**

- **Deng et al.,** ***npj Comput. Mater.*** **2024 ([arXiv:2405.07105](https://arxiv.org/abs/2405.07105))** — "Systematic softening" in M3GNet, CHGNet, MACE-MP-0, SevenNet-0: foundation MLIPs trained on near-equilibrium MPtrj systematically under-predict PES curvature, exactly the steep-PES regime the report flags.
- **Loew et al., "Universal MLIPs under Pressure" ([arXiv:2508.17792](https://arxiv.org/abs/2508.17792))** — Direct empirical link: uMLIP error grows monotonically with pressure (and thus density) from 0–150 GPa.
- **Bochkarev/Drautz et al., "Expanding Universal MLIPs to 97 Elements" ([arXiv:2603.03223](https://arxiv.org/abs/2603.03223))** — Ra, At, Cf show largest residuals; mechanism is DFT-convergence + sparse training data for heavy elements.
- **Po et al.,** ***Digital Discovery*** **2026 (D5DD00260E)** — MACE/Equiformer/eSEN errors track force-norm distance from MPtrj distribution; high-force/steep-PES regions are OOD.
- ZBL hybridization (DimezblNet, MACE+ZBL, NEP+ZBL) is widely deployed precisely *because* uMLIPs extrapolate poorly into the short-range repulsive wall — empirically confirms the report's mechanism.

**Refutes / qualifies:**

- **"Better without U" ([arXiv:2601.21056](https://arxiv.org/abs/2601.21056))** — OMAT-trained MACE actually *worsens* on Hubbard-U-corrected transition-metal oxides; failure is label-inconsistency, not density. Larger/denser sampling does not monotonically fix the problem.
- **Loew et al., "Universal MLIPs are Ready for Phonons" ([arXiv:2412.16551](https://arxiv.org/abs/2412.16551))** — MACE-MPA, SevenNet, GRACE achieve near-DFT phonons across \~10k heavy/dense chemistries; density alone is not a blocker once non-equilibrium data is in the training set.
- **OMat24 / EquiformerV2 ([arXiv:2410.12771](https://arxiv.org/html/2410.12771v1))** — Switching MPtrj → OMat24 (rattled/strained sampling) gives \~5× force-MAE improvement at fixed architecture, implying the latent axis is *force-norm coverage*, not density per se. Density is a correlate, not a clean cause.

**Net:** mechanism is real but density is best read as a *proxy* for force-norm/PES-curvature OOD; the singular-driver framing is overstated.

***

## Conclusion 2 — "Average bond ionicity is the #2 failure driver (8/9 models): local message-passing misses Madelung"

**Supports:**

- **Ko, Finkler, Goedecker, Behler,** ***Nat. Commun.*** **2021 (4G-HDNNP)** — foundational: 2G/3G HDNNPs fail on non-local charge transfer; charge equilibration is the documented fix.
- **Anstine & Isayev,** ***JPCA*** **2023 — review** — local MLIPs systematically blind to long-range Coulomb; ionic/polar = primary failure mode.
- **Grisafi & Ceriotti (LODE) / Huguenin-Dumittan et al.,** ***JPCL*** **2023** — explicit motivation: SOAP-type local features fail on ionic systems.
- **Zhang/E et al.,** ***JCP*** **2022 (Deep Potential + LR)**; **Yue et al.,** ***JCP*** **2021** ("When do short-range NN potentials fail?").
- The architectural fact that NequIP/MACE/Equiformer/SevenNet/PET/GRACE/eSEN all use 4–6 Å cutoffs without explicit Ewald confirms the report's mechanism is plausible *a priori*.

**Refutes / qualifies:**

- **Kim & Cheng,** ***JCP*** **2026 ([arXiv:2512.18029](https://pubs.aip.org/aip/jcp/article/164/6/060901/3379367/Long-range-electrostatics-for-machine-learning))** — "Long-range electrostatics is easier than we thought": deep MPNNs implicitly recover most Madelung energy through stacked message passing.
- **Cheng group's CACE-LR / LES ([arXiv:2408.15165](https://arxiv.org/abs/2408.15165),** **[arXiv:2412.15455](https://arxiv.org/abs/2412.15455))** — gains over short-range CACE on bulk ionic *solids* are often modest (matters most for interfaces, defects, dissociation).
- **Matbench Discovery leaderboard** — EquiformerV2+DeNS, eSEN, SevenNet, MACE achieve near-DFT F1 on ionic oxides/halides without explicit long-range terms; eSEN's APE drops 14.7% → 2.9% just by changing training set (data > architecture).

**Mitigation architectures (with claimed gains):**

- Ewald MP (Kosmala et al., ICML 2023, [arXiv:2303.04791](https://arxiv.org/abs/2303.04791)) — \~10% MAE reduction on OC20.
- LODE / Range-Separated LODE (Grisafi, Huguenin-Dumittan, Ceriotti).
- 4G-HDNNP and iterative 4G-HDNNP ([arXiv:2502.07907](https://arxiv.org/abs/2502.07907)).
- CACE-LR / LES (Cheng group).
- PET-MAD with polarizable LR (Loche, Mazitov et al., 2025).

**Net:** mechanism is canonical and motivates an entire sub-field, but for *bulk crystalline stability* (the WBM task) ionicity is plausibly a smaller axis than the report suggests; the effect is genuinely large for charged defects/interfaces/molecules.

***

## Conclusion 3 — "WBM dataset contains unrelaxed high-symmetry prototypes → high point-group/space-group order causes catastrophic failure; polyhedral distortion REDUCES error"

**Supports:**

- **Riebesell et al.,** ***Matbench Discovery*** **(Nature Mach. Intell. 2025;** **[arXiv:2308.14920](https://arxiv.org/abs/2308.14920))** — confirms WBM is built by 5 iterative prototype-substitution generations and explicitly notes "WBM structures have overall higher symmetry" than MP. The IS2RE task uses unrelaxed input. Direct structural support for the mechanism.
- **Deng et al. (softening,** **[arXiv:2405.07105](https://arxiv.org/abs/2405.07105))** — uMLIPs under-predict curvature near symmetric saddles, exactly the relaxation-energy gap the report invokes for point-group/space-group failures.
- **Loew et al. phonons ([arXiv:2412.16551](https://arxiv.org/abs/2412.16551))** — ORB and OMat24 mislabel >80% of dynamically unstable high-symmetry structures as stable; congruent with the report's `matris` >1 eV/atom failure on `space_group_number`.
- **Falletta et al.,** ***JCP*** **164:174703 (2026)** — Jahn-Teller octahedra carry larger MLIP errors than non-JT systems.

**Refutes / qualifies:**

- **WBM labels are DFT-relaxed**, not unrelaxed prototypes. The standard Matbench Discovery RS2RE task uses the relaxed geometry; the "unrelaxed prototype" narrative only cleanly applies to IS2RE. The report doesn't distinguish.
- **Matbench Discovery** notes that symmetry-aware equivariant models (NequIP/MACE/Equiformer) typically achieve *lower* error on high-symmetry cells (fewer DOFs to learn). The "high symmetry → high error" claim conflicts with this trend except for the unrelaxed-input case.
- High-symmetry prototypes are *over-represented* in MP training (OMat24 paper [arXiv:2410.12771](https://arxiv.org/html/2410.12771v1)); naïvely this predicts *lower*, not higher, test error in that region.
- **Confounding:** large-prototype-order cells (Fm-3m, Pm-3m) are dominated by oxides/perovskites with magnetism/+U corrections — the report's "symmetry" signal could be chemistry under another name.
- **Anisotropy claim contradicted:** Focassio et al. ([arXiv:2403.04217](https://arxiv.org/abs/2403.04217)) and the 2D/vdW literature consistently show *layered/anisotropic* materials are **harder** for uMLIPs (MP lacks dispersion data). The report's "anisotropy reduces error" finding runs *against* this body of evidence.

**Net:** the unrelaxed-prototype mechanism is real for IS2RE; the polyhedral-distortion finding aligns with the softening literature; but the anisotropy direction in the report disagrees with published vdW benchmarks, suggesting either an unusual WBM substructure or confounding.

***

## Conclusion 4 — "Equivariant models fail on symmetry-breaking distortions (Jahn-Teller, off-centering, ferroelectric)"

**Supports:**

- **Smidt, Geiger, Miller,** ***PRR*** **3 L012002 (2021),** **[arXiv:2007.02005](https://arxiv.org/abs/2007.02005)** — proves Curie's principle: a strictly equivariant net cannot lower the output's symmetry below its input's. Perovskite octahedral tilts is their canonical example.
- **Smidt et al. ([arXiv:2312.09016](https://arxiv.org/html/2312.09016v1), 2023)** — formalizes that equivariant NNs "are unable to model symmetry-breaking phenomena at the sample level"; recommends noise injection.
- Deng softening + Loew phonons (above) provide the empirical complement.

**Refutes / qualifies:**

- **Mao et al.,** ***JCP*** **164:174703 (2026)** — NequIP-based equivariant MLIP *does* reproduce the LaMnO₃ J-T order-disorder transition when DFT+U is in the training data. Bottleneck is electronic structure, not equivariance.
- **Yang et al. (BaTiO₃, IOP 2025)**, **Wu et al. (KNbO₃, 2022)** — NN potentials successfully reproduce ferroelectric double wells, off-centering, domain walls.
- **Falletta et al.** ***npj Comput. Mater.*** **2025 (LES)** — captures PbTiO₃ polarization and hysteresis.
- **NNQMD for PbTiO₃ (*Nano Lett.*** **2023)** — equivariant-style models capture flux-closure domain switching.
- **Curie's principle bite is conditional**: real numerical inputs are never perfectly symmetric, and any non-zero displacement breaks degeneracy. The "stuck at symmetric minimum" failure manifests only with idealized parent inputs.

**Net:** firm theoretical backing, but empirically the symmetry-breaking failure is usually *training-data-driven* rather than architecture-driven. The report's massive `atomic_off_centering` finding for eSEN (n=7) could plausibly be novel evidence but is too low-N to attribute confidently.

***

## Conclusion 5 — "PET is the most robust model (and TACE/GRACE/MatRIS handle strain better than equivariant)"

**Supports:**

- **Pozdnyakov & Ceriotti (PET, NeurIPS 2023,** **[arXiv:2305.19302](https://arxiv.org/abs/2305.19302))** + **Langer et al. ([arXiv:2406.17747](https://arxiv.org/abs/2406.17747))** — theoretical argument that non-equivariant models learn symmetry from data and may benefit on distorted/low-symmetry environments where rigid equivariance offers less leverage.
- **PET-MAD (Mazitov et al.,** ***Nat. Commun.*** **2025,** **[arXiv:2503.14118](https://arxiv.org/abs/2503.14118))** — MAD training set was explicitly designed to include high-energy / off-equilibrium / distorted snapshots; authors claim "no catastrophic failure across diverse chemistries". Consistent with the report's "fewest error-increasing features for PET" finding.
- **Deng softening literature** — congruent with equivariant models worsening on strain/anisotropy, leaving room for PET-MAD to outperform on those axes.

**Refutes / qualifies:**

- **Matbench Discovery leaderboard** — top stability F1 = 0.931 belongs to **EquiformerV3+DeNS-OAM** (equivariant), beating PET-MAD. GRACE-2L and MatRIS sit below leading equivariant models on F1.
- **MLIPAudit ([arXiv:2511.20487](https://arxiv.org/pdf/2511.20487))** and **Po et al. (D5DD00260E)** — *all* uMLIPs (PET included) degrade out of distribution; PET is not exempt.
- **TACE identity error:** Report calls TACE "Transformer-ACE". The actual TACE paper ([arXiv:2509.14961](https://arxiv.org/html/2509.14961), Cheng et al. 2025) is **Cartesian Tensor ACE**, fully symmetry-consistent in Cartesian form. So TACE is closer to an equivariant ACE than an "invariant transformer", undercutting the report's grouping of TACE with PET.
- **GRACE** is ACE+graph (symmetry built in via ACE basis), not invariant. **MatRIS** is invariant with attention three-body; no published claim of strain robustness.

**Net:** PET-MAD's training-set design supports the "few error-increasing features" finding, but the strong "most robust overall" claim doesn't hold on stability F1. The mechanistic grouping (PET-class invariant ≠ equivariant) contains a factual misclassification (TACE).

***

## Conclusion 6 — Methodology (DML + paired-difference for causal attribution)

**Supports:**

- **Chernozhukov et al. ([arXiv:1608.00060](https://arxiv.org/abs/1608.00060))** — foundational DML; orthogonalization + cross-fitting with high-dimensional nuisance learners.
- **Kallus,** ***JMLR*** **2020**; **Stuart 2010 ([arXiv:1010.5586](https://arxiv.org/pdf/1010.5586))** — formal grounding for paired-matching.
- **Suresh et al. (OpenReview rcmKO31Js6)** — applies DML to interpret ML model behavior, parallel to this use case.
- **Bach et al., PMLR 2024** — hyperparameter tuning for DML; defends nuisance-learner choice.

**Refutes / qualifies (significant):**

- **Lundberg (SHAP causal docs)** — direct warning that correlational attributions and naïvely-deconfounded DML mislead when unmeasured confounders dominate. Materials features here (density, ionicity, point-group order) are jointly determined by composition/chemistry — classic backdoor structure not closed by the report's covariates.
- **Hunermund & Bareinboim 2025**; **Byrnes,** ***Ecology Letters*** **2025 (doi:10.1111/ele.70023)** — argue that bias magnitude is unknowable without enumerating confounders; chemistry has pervasive unmeasured confounders (valence, magnetic order, +U treatment).
- **PMC12096307 (2024)** — DML overlap/positivity assumption frequently violated in observational scientific data.
- **[arXiv:2508.20366](https://arxiv.org/html/2508.20366v1)** **(2025)** — unmeasured-confounding diagnostic tests; the report does not report these.

**Competing approaches in MLIP literature:**

- **MLIP Arena ([arXiv:2509.20630](https://arxiv.org/pdf/2509.20630))** — stratified multi-task benchmarking; argues aggregate ATEs hide error cancellation.
- **Bigi et al. ([arXiv:2502.07104](https://arxiv.org/html/2502.07104))** and **[arXiv:2508.03405](https://arxiv.org/html/2508.03405)** — UQ + clustering of atomic environments as the dominant alternative paradigm.
- **Matbench Discovery's** chemistry/per-element stratification is the de-facto standard.

**Net:** DML is methodologically valid in principle, but in this domain confounders are pervasive (density↔heavy elements↔magnetism↔+U), the report enumerates no diagnostic for unmeasured confounding, and several effect signs disagree between DML and paired estimators (the report itself flags "heterogeneous" cases for `grace`, `matris`, `tace`, `sevennet`). The findings should be read as **risk-correlates**, not causal effects.

***

# Overall Assessment

| Report claim                                 | Mechanism support        | Empirical support                         | Counter-evidence weight                                                      |
| -------------------------------------------- | ------------------------ | ----------------------------------------- | ---------------------------------------------------------------------------- |
| Density as universal #1 driver               | Strong                   | Moderate                                  | Moderate (force-norm/+U are confounders)                                     |
| Ionicity / long-range driver (8/9)           | Very strong (canonical)  | Mixed                                     | Moderate (recent "easier than thought" results)                              |
| WBM unrelaxed prototypes → symmetry failures | Strong for IS2RE         | Strong for matris/equflash                | High for RS2RE; **anisotropy direction conflicts with vdW benchmarks**       |
| Equivariant models fail symmetry breaking    | Strong theory (Curie)    | Mixed (succeeds with right training data) | Moderate (LaMnO₃, BaTiO₃ successes)                                          |
| PET = most robust                            | Plausible (MAD training) | Partial                                   | Strong (EquiformerV3 leads Matbench Discovery; TACE misclassified)           |
| DML/paired methodology                       | Sound in principle       | —                                         | Significant (confounders unaddressed, sign disagreements internal to report) |

## Recommendations to strengthen the report

1. Reframe density and ionicity as *correlates* / risk markers, not isolated causes; cite the force-norm-coverage and Hubbard-U-inconsistency confounders.
2. Distinguish IS2RE vs. RS2RE WBM tasks when invoking the "unrelaxed prototype" mechanism.
3. Reconcile the anisotropy claim with the vdW/2D literature — the disagreement is the most surprising part of the report and deserves explicit treatment.
4. Correct the TACE characterization (it is Cartesian-Tensor ACE, not Transformer-ACE) and tighten the "PET most robust" claim against Matbench Discovery rankings.
5. Add unmeasured-confounding diagnostics (e.g., E-values, [arXiv:2508.20366](https://arxiv.org/html/2508.20366v1)) and address the DML-vs-paired sign disagreements explicitly.
6. The eSEN `atomic_off_centering` n=7 finding is intriguing but underpowered; flag as exploratory.

***

# Full Reference List

## Universal MLIP benchmarks & foundation models

- [Matbench Discovery (Riebesell et al., Nature Mach. Intell. 2025)](https://www.nature.com/articles/s42256-025-01055-1) — [arXiv:2308.14920](https://arxiv.org/abs/2308.14920)
- [WBM dataset (Wang/Botti/Marques, npj Comput. Mater. 2021)](https://doi.org/10.1038/s41524-020-00481-6)
- [OMat24 / EquiformerV2 (arXiv:2410.12771)](https://arxiv.org/html/2410.12771v1)
- [MACE-MP-0 (arXiv:2401.00096)](https://arxiv.org/abs/2401.00096)
- [eSEN (arXiv:2502.12147)](https://arxiv.org/html/2502.12147v1)
- [Orb-v3 (arXiv:2410.22570)](https://arxiv.org/html/2410.22570v1)
- [UMA — Meta FAIR (arXiv:2506.23971)](https://arxiv.org/abs/2506.23971)

## Density / heavy-element / steep-PES

- [Deng et al., Systematic softening (npj Comput. Mater. 2024)](https://www.nature.com/articles/s41524-024-01500-6) — [arXiv:2405.07105](https://arxiv.org/abs/2405.07105)
- [Loew et al., MLIPs under Pressure (arXiv:2508.17792)](https://arxiv.org/abs/2508.17792)
- [Loew et al., Ready for Phonons (arXiv:2412.16551)](https://arxiv.org/abs/2412.16551)
- [Bochkarev et al., 97 Elements (arXiv:2603.03223)](https://arxiv.org/abs/2603.03223)
- [Po et al., Distribution shifts in uMLIPs (Digital Discovery 2026)](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00260e)
- [Focassio et al., Surfaces (arXiv:2403.04217)](https://arxiv.org/abs/2403.04217)
- [Better without U (arXiv:2601.21056)](https://arxiv.org/abs/2601.21056)
- [MLIP Arena (arXiv:2509.20630)](https://arxiv.org/abs/2509.20630)

## Long-range electrostatics

- [Ko et al., 4G-HDNNP (Nat. Commun. 2021)](https://www.nature.com/articles/s41467-020-20427-2)
- [Iterative 4G-HDNNP (arXiv:2502.07907)](https://arxiv.org/abs/2502.07907)
- [Anstine & Isayev review (JPCA 2023)](https://pubs.acs.org/doi/10.1021/acs.jpca.2c06778)
- [LODE / Physics-Inspired Equivariant (JPCL 2023)](https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02375)
- [Deep Potential + LR (JCP 2022)](https://pubs.aip.org/aip/jcp/article/156/12/124107/2841008)
- [Ewald MP (Kosmala et al., arXiv:2303.04791)](https://arxiv.org/abs/2303.04791)
- [Cheng, Latent Ewald (arXiv:2408.15165)](https://www.nature.com/articles/s41524-025-01577-7)
- [Kim/Loche/Cheng, Learning charges from energies/forces (Nat. Commun. 2025)](https://www.nature.com/articles/s41467-025-63852-x)
- [Kim & Cheng, "Easier than thought" (JCP 2026)](https://pubs.aip.org/aip/jcp/article/164/6/060901/3379367/)
- [Foundation MLIP with polarizable LR (Nat. Commun. 2025)](https://www.nature.com/articles/s41467-025-65496-3)

## WBM / symmetry / prototype substitution

- [LaMnO3 J-T MLIP (JCP 2026)](https://pubs.aip.org/aip/jcp/article/164/17/174703/3388705/)
- [BaTiO3 MLIP (IOP 2025)](https://iopscience.iop.org/article/10.1088/1361-648X/ae2727)
- [Benchmarking uMLIPs for structural relaxation (OpenReview)](https://openreview.net/forum?id=fNyXCCZ0g6)

## Equivariance / symmetry breaking

- [Smidt et al., Finding Symmetry Breaking (PRR 2021)](https://arxiv.org/abs/2007.02005)
- [Symmetry Breaking and Equivariant NNs (arXiv:2312.09016)](https://arxiv.org/html/2312.09016v1)
- [FAENet (arXiv:2305.05577)](https://ar5iv.labs.arxiv.org/html/2305.05577)
- [To Augment or Not to Augment (arXiv:2510.01349)](https://arxiv.org/html/2510.01349)
- [NequIP (Nat. Commun. 2022)](https://www.nature.com/articles/s41467-022-29939-5)
- [MLIP can infer electrical response (npj Comput. Mater. 2025)](https://www.nature.com/articles/s41524-025-01911-z)
- [NNQMD PbTiO3 (Nano Lett. 2023)](https://pubs.acs.org/doi/10.1021/acs.nanolett.3c01885)

## PET / TACE / GRACE / MatRIS

- [PET — Pozdnyakov & Ceriotti (NeurIPS 2023, arXiv:2305.19302)](https://arxiv.org/abs/2305.19302)
- [PET-MAD (arXiv:2503.14118)](https://arxiv.org/abs/2503.14118)
- [PET-MAD Nature Communications](https://www.nature.com/articles/s41467-025-65662-7)
- [Langer et al., Probing broken symmetries (arXiv:2406.17747)](https://arxiv.org/abs/2406.17747)
- [GRACE foundational (arXiv:2508.17936)](https://arxiv.org/abs/2508.17936)
- [GRACE semilocal PRX](https://link.aps.org/doi/10.1103/PhysRevX.14.021036)
- [TACE — Cartesian Tensor ACE (arXiv:2509.14961)](https://arxiv.org/html/2509.14961)
- [MatRIS (arXiv:2603.02002)](https://arxiv.org/html/2603.02002v3)
- [MLIPAudit (arXiv:2511.20487)](https://arxiv.org/pdf/2511.20487)

## Causal methodology (DML / paired matching)

- [Chernozhukov DML (arXiv:1608.00060)](https://arxiv.org/abs/1608.00060)
- [Bach et al. PMLR 2024](https://proceedings.mlr.press/v236/bach24a/bach24a.pdf)
- [Stuart matching review (arXiv:1010.5586)](https://arxiv.org/pdf/1010.5586)
- [Kallus, Generalized Optimal Matching (JMLR 2020)](https://jmlr.org/papers/volume21/19-120/19-120.pdf)
- [SHAP causal caveat](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html)
- [Challenges in valid causal estimates with ML (PMC12096307)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12096307/)
- [Test for unmeasured confounding (arXiv:2508.20366)](https://arxiv.org/html/2508.20366v1)
- [Byrnes, Ecology Letters 2025](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.70023)
- [UQ for misspecified MLIPs (arXiv:2502.07104)](https://arxiv.org/html/2502.07104)
- [Model accuracy + heterogeneity UQ (arXiv:2508.03405)](https://arxiv.org/html/2508.03405)
- [From Prediction to Causal Interpretation DML (OpenReview)](https://openreview.net/pdf?id=rcmKO31Js6)

*End of transcript.*
