# Act-0 Report: Reproducing EEG-VJEPA (EEE 402 proposal)

*Last updated: 2026-09-26 23:10 (+06). All Act-0 runs are finished; no new experiments (user decision 19:04).

> **Licensing.** The TUAB copy used here comes from Kaggle and is mislicensed; the real TUAB needs the NEDC data-use agreement. Every TUAB number below is **internal only** until that agreement is in place.

## Scope

The proposal has two parts:

- **Part 1.** Reproduce the paper's frozen-encoder TUAB normal/abnormal results using the **released** EEG-VJEPA checkpoints (ViT-M 4×30×4, ViT-B 4×30×2).
- **Part 2.** Healthy vs stroke on Zenodo 19599466, using frozen encoders plus the Brain Symmetry Index (BSI), evaluated leave-one-subject-out.

FEI+C is a comparison arm in both parts. It uses plain FEI (never Ada-FEI) plus Branch-C h20, with the NMT seed-0 fold encoders.

**Joint pretraining (added 2026-09-26 on user request):** FEI+C encoders are also pretrained on NMT + TUAB-train. Held out from all pretraining: each NMT test fold, the TUAB eval set, all stroke data. Results are reported next to the NMT-only encoders, not instead of them.

**Out of scope, by decision on 2026-09-26:** our own EEG-VJEPA pretraining and the ablations that depend on it; Liu2024 (motor imagery / paralysis side), stopped mid-run.

**Added 2026-09-26 (user):** fine-tuning the released ViT-M, and a supervised from-scratch ViT-M (status row FT).

> **Results Analysis PDF (done 2026-09-27, internal):** `docs/act0_results/Act0_Results_Analysis.pdf` — 8 experiments for newcomers; every number, table and figure is generated from the saved runs by `make_results.py` + `build_pdf.py` (see `docs/act0_results/STATUS.md`).

## Status

| Step | What | Status |
|---|---|---|
| P1.1 | TUAB → paper §4.1 preprocessing (19 ch, 100 Hz, 1–40 Hz, 5 s / 50 % windows) | ✅ gate passed: 1371 N / 1346 A train, 150 / 126 eval, 2329 subjects, no train/eval subject overlap |
| P1.2 | Evaluation subsets: 5 seeded, stratified, subject-disjoint draws of 546 recordings (276 N + 270 A); test = official eval set (276) | ✅ |
| P1.3 | Band-power sanity check (logistic regression) | ✅ ran; the pre-declared gate failed, but a diagnostic shows the prep is sound (see below) |
| P1.4 | Released checkpoints, frozen, the authors' eval protocol (500 epochs, batch 2, lr 1e-3), both heads, plus random-init controls | ✅ complete |
| P1.4b | Collapse metrics on TUAB | ✅ |
| P1.4c | Linear probe on frozen ViT-M features (the same probe as FEI+C) | ✅ done, including paired tests vs FEI+C |
| A0 | FEI+C load-path anchor on NMT | ✅ gate passed |
| FEI+C-TUAB | FEI+C on TUAB (prep → embed → probe) | ✅ done 10:05 |
| B | Phase B stroke: BSI gate, 9 leave-one-subject-out arms, permutation tests, CIs | ✅ done 11:41 (NMT-only encoders), both cohorts |
| J | **Joint FEI+C** (user, 10:15): FEI + C re-pretrained on NMT train-fold + TUAB **train** (2717), same recipe, 5 fold encoders → NMT CV, TUAB probe, ViT-M pairing, Phase B all re-run | ✅ done 12:03 (NMT CV, TUAB, ViT-M pairing, Phase B) |
| L | **Liu2024** (user, 11:05): frozen joint / NMT-only / random FEI+C on 50 stroke patients, left-vs-right MI + paralysis side | ⏹ **stopped 12:15 by user: out of scope for Act-0.** MI finished at chance (see Findings); side task killed after its ASYM baseline |
| FT | **ViT-M fine-tune (released ckpt) + supervised from-scratch**, 5 subsets each, authors' recipe (`code/tuab_finetune_fast.py`). Gate vs upstream `run_one_epoch`: fp32 eager bit-exact (0.0 over 50 steps); compile = fp noise only. Bench s/epoch: upstream 9.2, fp32 8.2, fp32+compile 8.0, **TF32 6.3 (1.45×)**, CUDA-graph variants slower (10.0–11.4), bf16+graphs 8.6; follow-up: bf16 eager 4.4 (not used: real precision deviation), TF32+compile 7.0 (no gain), eval ≈10% of an epoch; profile: fwd+bwd scales linearly with batch (17.8/34.1/67.7 ms at 2/4/8 → GPU saturated, run-fusion useless), AdamW foreach = 4.8 ms of ~23 ms/step → **fused AdamW from scratch_vitm_s0 on** (rounding-level: 0 at step 1, 3e-8 at step 10; ft_vitm_s0 used foreach) | ✅ **All 10 bf16 runs done 19:48:** fine-tune 75.5 ± 6.0 vs scratch 76.8 ± 12.8 final AUROC (paired p 0.81, see the FT section). ▶ **ft_vitm_s0 (TF32) DONE 14:00: final AUROC 66.0 / acc 59.4 / F1 59.2** (best-epoch* 78.8 @ ep 47; train acc never above ~62% → the authors' lr 1e-3 full-encoder recipe does not fit even the train set; paper FT 88.5). **ft_vitm_s0_bf16 DONE 14:37: final AUROC 81.5 / acc 73.2 / F1 72.9 / BAcc 72.8** (best-epoch* 86.1 @ 242; mean AUROC over epochs 101–500 = 81.3 vs TF32 66.1; train acc ~72–76 vs ~60): far better than TF32 on the same subset/seed, one run each, so the cause (precision vs a lucky trajectory at lr 1e-3 / batch 2) is not yet separated. **All remaining fine-tune + from-scratch runs switched to bf16** (lane 10, tags `*_bf16`, ~40 min each, ~21:00). Deviation from upstream `use_bfloat16: false`, labelled as such. Lane 10 step 1 benches bf16 eager vs +compile vs +max-autotune (never measured with bf16; bf16 steps are ~14 ms = likely launch-bound); the queue auto-adopts the fastest compile mode if ≥5 % faster. **Result: bf16 eager 3.4 s/epoch, +compile 4.2, +max-autotune 4.7 → compile is slower, queue stays eager bf16.** Live runs do 4.7–4.9 s/epoch vs 3.4 benched: CPU thermal throttling of the launch-bound loop |
| C60 | **60–360 s crop test** (user, 18:20): the braindecode/Gemein convention (drop the first 60 s) instead of our 0–300 s; everything else is identical (same subsets, seeds, recipes). Arms: prep gate → collapse check → band power → frozen ViT-M s0–s4 (cached, paired vs 67.1). **Trimmed 18:57 (user):** the single-seed bf16 fine-tune + scratch s0 were dropped (n = 1 cannot be read against the 70–81.5 seed spread). One recording (338 s, in no subset) keeps its last 300 s. Opt-in via `TUAB_CROP=60` (`*_c60` dirs; the defaults are verified bit-identical) | ⏹ **cancelled 19:03 (user: keep only what is necessary, finish by 21:00).** Never started; the crop stays an open, untested choice (see the literature check) |
| B-CV | **Phase B CV robustness** (user, 18:30, post-hoc): the same embeddings + classifier, re-scored with 100× stratified 5-fold, leave-pair-out and 5-fold permutation p (`code/stroke_phaseB_cv.py`) | ✅ done 19:06, **primary cohort only** (trimmed by user 19:04; stopped before sensitivity). Table in the Phase B section. Sensitivity keeps its preregistered LOSO + the audit's 10× 5-fold |
| B-clean | **Clean FEI+C re-prep** (user, 18:55, post-hoc): stroke FEI+C input rebuilt to match NMT's actual spectrum (no band-pass; one label-free zero-phase per-channel filter per cohort maps the pooled mean log-PSD onto NMT-train's), then gates (PSD, Branch-C BN z, C collapse) → CPU re-embed → LOSO + perm + 5-fold + LPO (`code/stroke_phaseB_clean.py`; design fixed in `runs/act0/phaseB_clean/POSTHOC.md` before any run). **Trimmed 19:04 (user): primary cohort only; 1000-perm p for FEI+C and rand-FEI+C only** | ✅ done 19:16 (gates PASS; FEI+C LOSO 0.833 → 0.926, still ties rand-FEI+C; see "Phase B clean") |


## Part 1 results (TUAB)

### P1.4: released checkpoints, frozen probe (final-epoch AUROC, mean ± sd)

The training set is subset *s* with probe seed *s*. Every run is scored on the 276 official eval recordings. The paper's targets are ViT-M Acc 83.3 / F1 82.4 / **AUROC 87.7**, and ViT-B 81.2 / 81.0 / **87.9**.

| Encoder | Head | n | Acc | F1 | AUROC | Best-epoch AUROC* |
|---|---|---|---|---|---|---|
| Released ViT-M | §4.2 attention (upstream code) | 5 | 62.8 ± 2.4 | 62.6 ± 2.3 | **67.1 ± 2.0** | 73.7 |
| Released ViT-M | Fig-3 attentive | 5 | 66.4 ± 3.5 | 66.0 ± 3.3 | **71.3 ± 2.5** | 78.4 |
| Random-init ViT-M | §4.2 attention | 3 | 70.9 ± 1.0 | 70.4 ± 0.6 | **78.7 ± 0.6** | 84.4 |
| Random-init ViT-M | Fig-3 attentive | 3 | 70.7 ± 2.3 | 70.5 ± 2.3 | **76.0 ± 1.6** | 83.9 |
| Released ViT-B | §4.2 attention | 5 | 59.5 ± 2.1 | 59.4 ± 2.1 | **66.2 ± 3.0** | 71.5 |
| Released ViT-B | Fig-3 attentive | 5 | 69.4 ± 3.3 | 68.7 ± 3.0 | **76.2 ± 3.1** | 81.4 |

\* Optimistic: the epoch is picked on the eval set.

Two protocol tests on subset 0: batch 256 instead of 2 gives AUROC 59.7 (under-trains, so we keep batch 2); an fp16 token cache gives 64.3.

**Caching is exact.** The encoder is frozen and the eval transform is the identity, so encoder tokens are cached per (recording, clip position). The upstream head, optimiser and schedule are then trained on the cache. We verified this against the live encoder: max |cached − live| is 7.6e-6.

**Reading:**
- The released checkpoints land **16–21 AUROC points below the paper**, and **below random-init weights** of the same architecture.
- Even the optimistic best-epoch number does not reach 87.7.

### Collapse (TUAB eval, 276 recordings)

| | Token std | Cross-recording cosine |
|---|---|---|
| Released ViT-M | 0.006 | 1.0000 |
| Random-init ViT-M | 0.654 | — |

The released ViT-M's representation is collapsed: nearly the same tokens for every recording. This matches what we found earlier on NMT.

### P1.4c: linear probe on frozen ViT-M features (StandardScaler + LogReg, same as FEI+C)

Each recording's feature is the token mean concatenated with the token std, averaged over all clip positions.

| Encoder | Subsets | AUROC | BAcc |
|---|---|---|---|
| Released | s0–s4 | 0.813 / 0.811 / 0.800 / 0.781 / 0.818 → **0.805 ± 0.013** | 0.731 |
| Random-init | s0–s2 | 0.813 / 0.756 / 0.784 → **0.784 ± 0.023** | 0.709 |
| Released, full train (2717) | — | **0.838** (AUC-PR 0.846, Acc 0.764, F1 0.739, Sens 0.730, Spec 0.793) | 0.762 |

**Reading:**
- **Released ≈ random** even with a head that can rescale the features, so pretraining adds nothing measurable.
- With the full training set (2717 recordings) the linear probe reaches 0.838. That is the best released-checkpoint number we have, and it is still 4 points short of 87.7.
- The paper's heads pull the released checkpoint down to 67–71, while random features stay at 76–79. The near-constant (collapsed) tokens hurt those heads, and standardizing the features removes that problem.

### P1.3: band-power sanity check (logistic regression, §4.1 prep)

| Train set | AUROC |
|---|---|
| Subsets s0–s4 | 0.730 ± 0.014 |
| Full train | 0.757 |

The pre-declared gate (≥ 0.80) failed. A diagnostic on an independent prep (CBraMod-style) gives the same picture:

| Features | AUROC |
|---|---|
| Relative band power, 5 min | 0.765 (≈ ours) |
| Relative band power, 20 min | 0.773 |
| Relative + log-absolute band power, 20 min | 0.808 |

So our prep is sound. The bar was set too high for relative-only band power: per-channel z-scoring removes amplitude, which is where the missing signal is.

### A0: FEI+C anchor on NMT (seed-0 5-fold CV, the same fold encoders as the Act-1 records)

**Gate: passed.** Plain FEI at L = 1000 reproduces every recorded fold AUROC to within 3e-4:

| Fold | Reproduced | Recorded |
|---|---|---|
| f0 | 0.8580 | 0.8583 |
| f1 | 0.7954 | 0.7953 |
| f2 | 0.8471 | 0.8472 |
| f3 | 0.7870 | 0.7872 |
| f4 | 0.8040 | 0.8040 |

The Act-0 configuration on NMT:

| Arm | AUROC | BAcc | AUC-PR |
|---|---|---|---|
| FEI (L = 4000) | 0.819 ± 0.029 | 0.729 | 0.563 |
| C (Branch-C h20) | 0.836 ± 0.021 | 0.752 | 0.572 |
| **FEI+C** | **0.865 ± 0.024** | 0.778 | 0.657 |

The earlier Act-1 figure of 0.870 was for Ada-FEI+C, which is not used here.

### Joint FEI+C (NMT + TUAB-train pretraining): NMT seed-0 5-fold CV

This is the A0 protocol with the same folds. Each fold's encoders were pretrained on that fold's NMT training recordings plus TUAB train; the NMT test fold was never seen. The comparison with the NMT-only encoders is paired, recording by recording.

| Arm | Joint AUROC | NMT-only AUROC | Δ (folds won) |
|---|---|---|---|
| FEI@4000 | 0.829 ± 0.023 | 0.819 ± 0.029 | +0.010 (4 / 5) |
| C | 0.822 ± 0.016 | 0.836 ± 0.021 | −0.014 (0 / 5) |
| FEI+C | 0.855 ± 0.025 | 0.865 ± 0.024 | −0.010 (1 / 5) |

C's embedding spread on the NMT test folds falls from 0.38 (NMT-only) to **0.19–0.30** (joint). FEI's stays healthy at 0.43–0.58.

**Reading:**
- **Adding TUAB helps FEI a little and hurts C on every fold.** C's past-to-future loss is solved within about 10 epochs, and the extra data compresses its embeddings further (partial collapse) instead of teaching it more.
- **Net effect: joint FEI+C is slightly below NMT-only FEI+C on NMT.**

### Joint FEI+C on TUAB (the same probe and splits as the NMT-only run below)

TUAB eval AUROC. The per-encoder mean is the fair comparison with random-init (5 pretrained members vs 3 random):

| Arm | Subsets s0–s4: joint | Subsets s0–s4: NMT-only | Full train: joint | Full train: NMT-only | Full train: random-init |
|---|---|---|---|---|---|
| FEI | 0.833 ± 0.008 | 0.835 ± 0.015 | 0.859 (members 0.844) | **0.872** (0.846) | 0.813 (0.786) |
| C | 0.811 ± 0.012 | 0.805 ± 0.005 | 0.858 (0.834) | 0.840 (0.818) | **0.863** (0.838) |
| FEI+C | 0.821 ± 0.012 | 0.821 ± 0.006 | 0.862 (0.841) | 0.857 (0.834) | 0.861 (0.837) |

Collapse on the TUAB eval set (embedding std / cross-recording cosine):

| Encoder | Joint | NMT-only | Random-init |
|---|---|---|---|
| FEI | 0.83–1.07 / 0.64–0.71 | 0.46–0.58 / 0.73–0.78 | 0.0006 / 0.9997 |
| C | **0.36–0.39 / 0.59–0.69** | 0.04–0.07 / 0.96–0.99 | 0.11 / 0.96 |

**Paired, joint FEI+C − released ViT-M (same linear probe):**
- Subsets: +0.016 ± 0.014.
- Full train: **+0.024**, 95 % CI [−0.022, +0.074], p = 0.15. Still **not significant**.

**Reading:**
- **Pretraining on TUAB itself buys essentially nothing on TUAB.**
  - FEI+C moves 0.857 → 0.862.
  - FEI alone moves 0.872 → 0.859, slightly down.
  - C improves 0.840 → 0.858, but only up to the level of its random-init control (0.863).
- **C's "collapse" is domain-specific, not global.**
  - With TUAB in its pretraining, C spreads TUAB recordings out well (std 0.37 vs 0.05 NMT-only, cosine 0.6 vs 0.97).
  - At the same time it compresses NMT (0.38 → 0.19–0.30).
  - So C reallocates its capacity toward the domains it has seen. Better spread does not turn into accuracy above the random spectrogram features.
- **Overall:** the random-init spectrogram features are a strong baseline, and neither pretraining recipe beats them on TUAB.

### FEI+C on TUAB (NMT-pretrained encoders, zero TUAB pretraining)

- **Prep:** 19 channels, first 20 min, 0.5–40 Hz, 200 Hz, per-channel z-score. ⚠️ The 0.5–40 Hz filter does **not** reproduce NMT's spectrum (Phase B audit below), so the NMT-only C encoder sees out-of-distribution spectrogram bins here. The joint encoders were pretrained on this prep, so they are the in-distribution comparison.
- **Encoders:** the 5 NMT seed-0 fold encoders (their predictions are averaged). Random-init controls: 3 seeds.
- **Probe:** StandardScaler + LogReg, the same as P1.4c. Scored on the 276 official eval recordings.

AUROC:

| Arm | Subsets s0–s4 (546 recordings) | Full train (2717) | Full train, per-encoder mean |
|---|---|---|---|
| FEI | **0.835 ± 0.015** | **0.872** | 0.846 ± 0.011 |
| rand-FEI | 0.790 ± 0.020 | 0.813 | 0.786 ± 0.016 |
| C | 0.805 ± 0.005 | 0.840 | 0.818 ± 0.011 |
| rand-C | 0.819 ± 0.014 | 0.863 | 0.838 ± 0.010 |
| FEI+C | 0.821 ± 0.006 | 0.857 | 0.834 ± 0.007 |
| rand-FEI+C | 0.834 ± 0.014 | 0.861 | 0.837 ± 0.002 |
| *Released ViT-M, same linear probe* | *0.805 ± 0.013* | *0.838* | — |

Full-train FEI+C also gives BAcc 0.792, AUC-PR 0.852, Sens 0.730, Spec 0.853. Every arm's metrics are in `results.json`.

**Collapse** (per-recording embedding std / mean cross-recording cosine):

| | Pretrained | Random-init |
|---|---|---|
| FEI | 0.46–0.58 / 0.73–0.78 | 0.0006 / 0.9997 |
| C | 0.04–0.07 / 0.96–0.99 | 0.11 / 0.96 |

**Paired test, FEI+C − released ViT-M (same probe, same eval recordings):**
- On the subsets, FEI+C is ahead by +0.016 ± 0.018. The per-subset gaps are 0.000 / +0.011 / +0.018 / +0.050 / 0.000.
- On the full train set, the gap is **+0.019**, with a 95 % CI of [−0.030, +0.066] and p = 0.23, so **not significant**.

**Reading:**
- **FEI transfers from NMT to TUAB; C does not.**
  - Pretrained FEI beats its random control by 4.5 points on the subsets and 6 points on the full train set, with no TUAB pretraining at all.
  - FEI's features are not collapsed.
  - Its 0.872 on the full train set is **at the paper's 87.7**, but that comparison uses a different head and prep.
  - Pretrained C is *worse* than random-init C. So is FEI+C vs rand-FEI+C, once the per-encoder means are compared.
- **Adding C to FEI hurts on TUAB** (0.872 → 0.857). This is the opposite of NMT, where FEI+C beat FEI.
  - Branch C's dynamics features look NMT-specific. The random-init C features stay strong (0.863), so the spectrogram input itself carries the signal.
- **FEI+C vs the released ViT-M is a statistical tie.** FEI alone is the arm that clearly clears ViT-M: 0.872 vs 0.838 on the full train set. That comparison was not a pre-declared test.
- **Caveat:** the headline numbers average 5 pretrained encoders but only 3 random ones. The per-encoder-mean column is the fair comparison, and it tells the same story.

### FT: fine-tune (released ViT-M) vs supervised from scratch, bf16, authors' recipe (final, 10 of 10 runs, done 19:48)

Recording-level metrics on the 276 official eval recordings. The AUROCs come from each run's `final_preds.npz`, and acc / BAcc / F1 from the last row of `probe_r0.csv`.

| Run | s0 | s1 | s2 | s3 | s4 | Mean ± sd |
|---|---|---|---|---|---|---|
| Fine-tune (released), final AUROC | 81.5 | 81.3 | 70.1 | 75.6 | 69.0 | **75.5 ± 6.0** |
| From scratch, final AUROC | 79.6 | 86.7 | 54.7 | 79.2 | 83.9 | **76.8 ± 12.8** |
| Fine-tune, mean AUROC ep 101–500 | 81.3 | 77.6 | 65.5 | 76.3 | 55.9 | 71.3 ± 10.4 |
| From scratch, mean AUROC ep 101–500 | 78.6 | 84.6 | 53.0 | 78.2 | 82.8 | 75.5 ± 12.8 |
| Fine-tune, best-epoch* | 86.1 | 84.5 | 74.9 | 84.2 | 76.4 | 81.2 ± 5.2 |
| From scratch, best-epoch* | 84.8 | 88.2 | 64.0 | 85.4 | 86.8 | 81.8 ± 10.1 |
| Fine-tune, final acc / BAcc / F1 | 73.2 / 72.8 / 72.9 | 74.6 / 74.3 / 74.3 | 64.1 / 64.1 / 64.0 | 65.9 / 65.8 / 65.8 | 65.2 / 64.8 / 64.9 | 68.6 / 68.4 / 68.4 |
| From scratch, final acc / BAcc / F1 | 73.2 / 72.5 / 72.7 | 75.7 / 75.4 / 75.5 | 54.7 / 54.8 / 54.6 | 71.7 / 71.3 / 71.4 | 75.4 / 75.4 / 75.3 | 70.1 / 69.9 / 69.9 |

\* The best epoch is chosen on the test set, so it is an optimistic upper bound and not a result. Upstream has no best-val/patience rule (500 epochs, last epoch kept).

TF32 reference (s0 only): 66.0. Paper FT: 85.8 acc / 85.6 F1 / **88.5 ± 0.2 AUROC**.

| Paired fine-tune − scratch (5 subsets) | s0 | s1 | s2 | s3 | s4 | Mean | ft > scratch | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| Final AUROC | +2.0 | −5.4 | +15.4 | −3.7 | −14.9 | −1.3 | 2/5 | 0.81 | 0.81 |
| Mean AUROC ep 101–500 | +2.7 | −7.0 | +12.5 | −1.9 | −26.9 | −4.1 | 2/5 | 0.56 | 0.81 |
| Best-epoch* | +1.3 | −3.8 | +10.9 | −1.1 | −10.4 | −0.6 | 2/5 | 0.87 | 1.00 |

**Reading:**
- **The released pretraining gives no measurable gain under the authors' fine-tune recipe.**
  - Fine-tune 75.5 ± 6.0 and from scratch 76.8 ± 12.8 are the same within noise: fine-tune wins on 2 of 5 subsets, p = 0.81.
  - This fits the collapse finding: a collapsed start is worth about as much as a random one.
- **Both are far below the paper's 88.5 ± 0.2**, and even the optimistic best-epoch means (81–82) fall short.
- **The recipe is unstable.**
  - On s2 (both runs) and ft s4, the model predicts the majority class (val acc 54.3 % = 150/276 normal) for about 200+ epochs, until the cosine lr has decayed.
  - Scratch s2 never recovers (final 54.7), and ft s4 degrades late (mean AUROC over ep 101–500 is 55.9).
  - This is the signature of lr 1e-3 at batch 2 being too high.
- **The spread across subsets (sd 6.0–12.8) cannot be squared with the paper's ± 0.2.**
- **bf16 is a labelled deviation** (upstream `use_bfloat16: false`). On s0, TF32 gave 66.0 vs bf16 81.5, from one run each.

## Part 2: stroke (Phase B)

**BSI gate: PASS in both cohorts.** Broadband pdBSI (1–30 Hz) is higher in stroke: Mann-Whitney p = 0.003 (primary) and p < 0.001 (sensitivity).

| Feature (stroke vs control) | Primary p | Sensitivity p |
|---|---|---|
| BSI 1–30 Hz ↑ | 0.003 | <0.001 |
| BSI delta / theta / alpha / beta ↑ | 0.005 / 0.113 / 0.012 / 0.003 | 0.001 / 0.021 / 0.002 / 0.001 |
| Relative delta ↑ / theta ↑ | 0.008 / 0.036 | 0.068 / 0.016 |
| Relative alpha ↓ / beta ↓ | 0.050 / 0.026 | 0.173 / 0.068 |
| DAR ↑ / DTABR ↑ | 0.018 / 0.002 | 0.122 / 0.027 |

The directions match the stroke literature (slowing plus asymmetry), so the cohort carries the expected signal.

### Phase B LOSO, primary cohort (9 stroke vs 6 controls, NMT-only FEI+C encoders)

LOSO AUROC with a 95 % bootstrap CI; permutation p comes from 1000 label shuffles.

| Arm | AUROC | 95 % CI | Perm p |
|---|---|---|---|
| BSI | 0.907 | 0.68–1.00 | 0.005 |
| VJEPA (released ViT-M) | **0.981** | 0.89–1.00 | 0.006 |
| VJEPA+BSI | 0.981 | 0.89–1.00 | 0.007 |
| rand-VJEPA | 0.833 | 0.57–1.00 | 0.039 |
| rand-VJEPA+BSI | 0.870 | 0.66–1.00 | 0.021 |
| FEI+C | 0.833 | 0.56–1.00 | 0.034 |
| FEI+C+BSI | 0.926 | 0.74–1.00 | 0.009 |
| rand-FEI+C | 0.889 | 0.68–1.00 | 0.014 |
| rand-FEI+C+BSI | 0.907 | 0.71–1.00 | 0.011 |

Pre-declared paired comparisons (Δ AUROC, 95 % CI, bootstrap p(Δ ≤ 0)):

| Comparison | Δ | 95 % CI | p |
|---|---|---|---|
| VJEPA+BSI vs BSI | +0.074 | −0.07 to +0.30 | 0.37 |
| VJEPA vs rand-VJEPA | +0.148 | 0.00 to +0.40 | 0.09 |
| FEI+C vs VJEPA | −0.148 | −0.39 to 0.00 | 0.996 (FEI+C is not better) |

**Reading (preliminary, n = 15):**
- **No pre-declared comparison is significant.** Every CI is wide.
- **The VJEPA 0.981 needs caution.** On this cohort the released ViT-M is collapsed even harder than on TUAB: recording std 0.0014, cross-recording cosine 0.999994. The probe separates on a vanishing residual that StandardScaler blows back up. With 15 subjects, that residual could just as well be a recording-condition difference as pathology.
- **Random-init encoders score 0.83–0.89**, so the architecture plus the probe does most of the work. BSI alone reaches 0.907.

### Phase B LOSO, sensitivity cohort (10 stroke vs 8 controls, first 300 s)

| Arm | AUROC | 95 % CI | Perm p |
|---|---|---|---|
| **BSI** | **0.938** | 0.78–1.00 | 0.002 |
| VJEPA | 0.863 | 0.64–1.00 | 0.013 |
| VJEPA+BSI | 0.887 | 0.69–1.00 | 0.010 |
| rand-VJEPA | 0.825 | 0.58–0.98 | 0.026 |
| rand-VJEPA+BSI | 0.862 | 0.64–1.00 | 0.014 |
| FEI+C | 0.788 | 0.50–1.00 | 0.037 |
| FEI+C+BSI | 0.800 | 0.53–1.00 | 0.032 |
| rand-FEI+C | 0.900 | 0.72–1.00 | 0.008 |
| rand-FEI+C+BSI | 0.913 | 0.74–1.00 | 0.006 |

Pre-declared comparisons:

| Comparison | Δ | 95 % CI | p |
|---|---|---|---|
| VJEPA+BSI vs BSI | −0.050 | −0.26 to +0.14 | 0.73 |
| VJEPA vs rand-VJEPA | +0.038 | −0.15 to +0.25 | 0.38 |
| FEI+C vs VJEPA | −0.075 | −0.28 to +0.08 | 0.81 |

### Phase B with joint FEI+C encoders

Only the FEI+C arms change. The other 7 arms' inputs were verified identical and reused.

| Cohort | FEI+C: joint | FEI+C: NMT-only | FEI+C+BSI: joint | FEI+C+BSI: NMT-only | rand-FEI+C | BSI |
|---|---|---|---|---|---|---|
| Primary (15) | 0.778 (0.45–1.00), p = 0.068 | 0.833 | 0.778 | 0.926 | 0.889 | 0.907 |
| Sensitivity (18) | 0.825 (0.55–1.00), p = 0.028 | 0.788 | 0.838 | 0.800 | 0.900 | 0.938 |

Joint FEI+C vs VJEPA: primary −0.204 (p = 0.995), sensitivity −0.037 (p = 0.69).

Joint pretraining does not rescue FEI+C on stroke. Its point estimates stay below random-init FEI+C and BSI in both cohorts, but no difference is significant (see the audit below).

**Phase B reading, both cohorts:**
- **Hand-crafted BSI is the most reliable arm** (0.907 / 0.938).
- **No frozen encoder beats BSI or its own random-init control significantly.**
  - VJEPA's 0.981 on the primary cohort drops to 0.863 on the sensitivity cohort, which is consistent with its collapsed features (cosine 0.999994) being unstable.
  - Pretrained FEI+C is **significantly above chance** (perm p = 0.034 / 0.037) but **statistically tied** with random-init FEI+C (post-hoc DeLong p = 0.66 / 0.35, bootstrap CIs include 0). It is lower only as a point estimate, and under repeated 5-fold the primary ranking flips in about half the partitions (audit below).
- **The proposal's Part-2 hypothesis (frozen EEG-VJEPA + BSI beats BSI) is not supported** at n = 15 / 18.


### Phase B audit (2026-09-26, user: "even FEI+C fails? are you completely sure? did we do a 5-fold test?")

Three independent auditors checked the FEI+C stroke path from different angles: inputs vs NMT, encoder and embedding code, and statistics. A separate skeptic then tried to refute each material finding, and none was refuted. Scripts are in the session scratchpad (`audit/`).

**Verified sound:**
- **Encoders and embedding:** the right plain-FEI and h20 checkpoints are loaded (strict, eval mode, L = 4000). The embedding call is identical to the NMT anchor (FEI 0.819 / FEI+C 0.865), and a CPU re-embed matches the stored embeddings to ≤ 1e-3.
- **Inputs:** channel order and renames, reference, sampling rate and z-score all match NMT. NMT is stored average-referenced, so CAR on stroke matches it.
- **LOSO:** no leakage, and the stored AUROCs reproduce exactly.
- **Permutation test:** valid, because the null re-runs the full LOSO.

**Defects found:**
1. **The FEI+C stroke prep is not NMT-matched: the "0.5–40 Hz = NMT device band" assumption is wrong.**
   - NMT has no filter (the EDF prefilter fields are NaN). Its spectrum rolls off by about 35–40 Hz onto a floor near −5.7 (log10 relative PSD).
   - The MNE 0.5–40 Hz FIR passes 40–45 Hz and drops to about −10 above 60 Hz.
   - This pushes Branch C's input BatchNorm (NMT running statistics) out of range: z ≈ +3 to +4 at 36–45 Hz and ≈ −3 above 46 Hz.
   - As a result, pretrained C **collapses on stroke**: embedding std 0.03 vs 0.38 on NMT, and 95 of 256 dims are constant. FEI is unaffected.
   - The same mismatch hits the NMT-only C on TUAB (`tuab_feic.py`).
   - **It does not change the conclusion:**
     - A causal probe that sets the > 40 Hz bins to the BN mean un-collapses C, and the ranking still holds on primary: FEI+C 0.907, rand 0.963, BSI 0.907.
     - The joint encoders were pretrained on 0.5–40 Hz TUAB, so they are in-distribution and not collapsed (std 0.24). They are still not better: 0.778 / 0.825.
2. **The NMT-only FEI+C point estimate is numerically fragile.**
   - The collapsed C dims (about 29 % with std < 1e-5) get blown up by StandardScaler.
   - A GPU vs CPU re-embed moves FEI+C from 0.833 to 0.907 (primary) and from 0.788 to 0.862 (sensitivity), while the random-init arm stays put.
   - Read FEI+C as **≈ 0.83–0.91**, not as a precise 0.833.
3. **The report's "FEI+C below random-init" was an untested point-estimate ordering.** The pre-registered comparisons don't include it. Post-hoc DeLong tests:

   | Comparison (p, primary / sensitivity) | NMT-only encoders | Joint encoders |
   |---|---|---|
   | FEI+C vs rand-FEI+C | 0.66 / 0.35 | 0.32 / 0.48 |
   | FEI+C vs BSI | 0.65 / 0.33 | 0.49 / 0.44 |

   None is significant. Wording corrected above.
4. **Pooled LOSO AUROC is biased low** (permutation-null mean 0.44–0.47). The permutation p-values stay valid. Leave-pair-out scores are about 0.04 higher for every arm, so the ranking is unchanged.
5. **The "+BSI" arms bury BSI.** Early concatenation leaves the 5 BSI features among 517–773 standardized dims, where they hold about 1 % of the variance. So these arms are only a weak test of complementarity.
   - Post-hoc late fusion (averaging probabilities with BSI) ceilings every arm: FEI+C 0.963 / 0.975, rand-FEI+C 1.000 / 0.988, VJEPA 1.000 / 0.950.
   - That means n = 15 / 18 cannot separate them.

**Answer:**
- Phase B used **leave-one-subject-out** (15- and 18-fold), not 5-fold. The "5" in FEI+C is the 5 NMT fold encoders.
- The conclusion **survives** all of the following:
  - every CV scheme;
  - the prep-bug probe;
  - the in-distribution joint encoders;
  - numerical noise.
- The conclusion is that **pretrained FEI+C detects stroke above chance but does not beat its random-init control or BSI.** "Worse than random-init" is **not** supported.
- Repeated 5-fold and leave-pair-out numbers for all arms are in the CV-robustness table below.

### Phase B CV robustness, primary cohort (post-hoc, `code/stroke_phaseB_cv.py`, done 19:06)

The embeddings and classifier are the same as in the pre-registered LOSO; only the CV scheme changes. LOSO is recomputed, and the script asserts it equals the stored pre-registered values.
- **5-fold ×100:** stratified 5-fold, 100 repeats (seeds 0–99), with pooled out-of-fold AUROC per repeat.
- **5-fold perm p:** 200 label permutations × 10 repeats each.
- **LPO:** leave-pair-out (every stroke × control pair held out together, 54 pairs). It is unbiased at small n, unlike pooled LOSO.
- **Trimmed:** primary cohort only (user, 21:00 deadline). The sensitivity cohort keeps its pre-registered LOSO plus the audit's 10× 5-fold.

| Arm | LOSO | 5-fold ×100, mean ± sd [2.5–97.5 %] | LPO | 5-fold perm p |
|---|---|---|---|---|
| BSI | 0.907 | 0.920 ± 0.012 [0.907, 0.944] | 0.944 | 0.005 |
| VJEPA | 0.981 | 0.975 ± 0.028 [0.889, 1.000] | 1.000 | 0.015 |
| VJEPA+BSI | 0.981 | 0.976 ± 0.027 [0.898, 1.000] | 1.000 | — |
| rand-VJEPA | 0.833 | 0.842 ± 0.051 [0.741, 0.917] | 0.889 | 0.030 |
| rand-VJEPA+BSI | 0.870 | 0.857 ± 0.049 [0.750, 0.926] | 0.907 | — |
| FEI+C | 0.833 | 0.874 ± 0.042 [0.815, 0.954] | 0.870 | 0.020 |
| FEI+C+BSI | 0.926 | 0.916 ± 0.031 [0.852, 0.963] | 0.944 | 0.010 |
| rand-FEI+C | 0.889 | 0.856 ± 0.049 [0.768, 0.944] | 0.926 | 0.030 |
| rand-FEI+C+BSI | 0.907 | 0.891 ± 0.044 [0.796, 0.944] | 0.963 | — |
| FEI *(exploratory)* | 0.852 | 0.841 ± 0.057 [0.741, 0.926] | 0.926 | — |
| C *(exploratory)* | 0.870 | 0.875 ± 0.051 [0.787, 0.963] | 0.889 | — |
| rand-FEI *(exploratory)* | 0.741 | 0.728 ± 0.066 [0.611, 0.833] | 0.778 | — |
| rand-C *(exploratory)* | 0.907 | 0.900 ± 0.046 [0.805, 0.963] | 0.963 | — |
| FEI+C (joint) | 0.778 | 0.791 ± 0.035 [0.704, 0.852] | 0.852 | 0.060 |
| FEI+C+BSI (joint) | 0.778 | 0.804 ± 0.030 [0.759, 0.852] | 0.852 | 0.035 |
| FEI (joint) *(exploratory)* | 0.648 | 0.668 ± 0.073 [0.509, 0.778] | 0.741 | — |
| C (joint) *(exploratory)* | 0.815 | 0.846 ± 0.032 [0.778, 0.899] | 0.889 | — |

| Comparison | 5-fold Δ (A better in % of repeats) | LPO Δ [95 % CI], p(Δ ≤ 0) |
|---|---|---|
| VJEPA+BSI vs BSI *(pre-reg)* | +0.056 (93 %) | +0.056 [+0.000, +0.222], 0.36 |
| VJEPA vs rand-VJEPA *(pre-reg)* | +0.133 (100 %) | +0.111 [+0.000, +0.296], 0.07 |
| FEI+C vs VJEPA *(pre-reg)* | −0.101 (2 %) | −0.130 [−0.352, +0.000], 1.00 |
| FEI+C vs rand-FEI+C | +0.018 (57 %) | −0.056 [−0.259, +0.111], 0.82 |
| FEI+C vs BSI | −0.046 (19 %) | −0.074 [−0.315, +0.148], 0.79 |
| FEI+C+BSI vs BSI | −0.004 (57 %) | +0.000 [−0.185, +0.167], 0.57 |
| FEI+C (joint) vs rand-FEI+C | −0.066 (8 %) | −0.074 [−0.222, +0.000], 1.00 |
| FEI vs rand-FEI | +0.113 (91 %) | +0.148 [−0.056, +0.444], 0.15 |
| C vs rand-C | −0.025 (31 %) | −0.074 [−0.222, +0.000], 1.00 |

**Reading:**
- **The CV scheme changes no conclusion.** Every pre-registered arm is above chance under 5-fold permutation (p ≤ 0.03). Joint FEI+C is borderline (p 0.060).
- **Pretrained FEI+C ties its random-init control** (better in 57 % of repeats, LPO CI spans 0). It sits below BSI and VJEPA.
- **The branches split:**
  - **FEI:** pretrained beats random in 91 % of repeats (+0.11), but the LPO CI still spans 0.
  - **C:** pretrained is *not* better than random (31 %). This is the collapsed branch from defect 1, and it is what the clean re-prep tests.
- **The released ViT-M is the strongest single arm** (0.975, 100 % of repeats over random-init). "% of repeats" re-splits the same 15 subjects, so it is not a significance test. The subject-level LPO bootstrap is the honest one: +0.111, CI [0.000, 0.296], p(Δ ≤ 0) = 0.07.

### Phase B clean FEI+C re-prep, primary cohort (post-hoc, `code/stroke_phaseB_clean.py`, done 19:16)

This tests audit defect 1: the pre-registered 0.5–40 Hz, 200 Hz prep did not reproduce NMT's spectrum, which pushed Branch C's BatchNorm out of range and collapsed C.
- **Design** (fixed in `runs/act0/phaseB_clean/POSTHOC.md` before any run; sha in the results):
  - authors' bad-channel interpolation → CAR → 19 ch → 200 Hz, **no band-pass** → per-channel z;
  - then ONE fixed, label-free, zero-phase per-channel filter per cohort that maps the cohort's pooled mean log-PSD onto the NMT-train mean (100 recordings);
  - then re-z → 2.5 s frames.
- **Unchanged:** same 5 NMT-only plain-FEI + h20-C fold encoders and 3 random-init controls, same classifier. The LOSO, 5-fold ×100 (same seeds and partitions) and LPO are as above.
- **Permutations:** 1000-perm p only for FEI+C and rand-FEI+C (trim).
- **References:** BSI, VJEPA and "FEI+C (orig prep)" are the pre-registered predictions.

**Gates: all PASS.**

| Gate | Pre-registered prep | Clean prep | Threshold |
|---|---|---|---|
| PSD max \|dev\| from NMT, 0.5–99 Hz (decades) | 5.811 | **0.184** | ≤ 0.25 |
| Branch-C BN max \|z\| | 4.26–4.31 | **0.36–0.38** | ≤ 1.5 |
| C rec_std, fold encoders (NMT 0.38–0.40) | collapsed | **0.28–0.31** (ratio 0.74) | ratio ≥ 0.5 |
| Constant C dims | — | **0.00** | ≤ 5 % |

Random-init C rec_std is 0.095–0.098.

| Arm (clean prep) | LOSO AUROC | 5-fold ×100 | LPO | perm p (LOSO) | Before (orig prep): LOSO / 5-fold / LPO |
|---|---|---|---|---|---|
| FEI+C | **0.926** | 0.918 ± 0.051 | 0.981 | 0.003 | 0.833 / 0.874 / 0.870 |
| FEI+C+BSI | **0.963** | 0.951 ± 0.039 | 0.981 | — | 0.926 / 0.916 / 0.944 |
| rand-FEI+C | 0.944 | 0.930 ± 0.037 | 1.000 | 0.004 | 0.889 / 0.856 / 0.926 |
| rand-FEI+C+BSI | 0.944 | 0.942 ± 0.033 | 1.000 | — | 0.907 / 0.891 / 0.963 |
| FEI *(exploratory)* | 1.000 | 1.000 ± 0.000 | 1.000 | — | 0.852 / 0.841 / 0.926 |
| C *(exploratory)* | 0.833 | 0.819 ± 0.063 | 0.852 | — | 0.870 / 0.875 / 0.889 |
| rand-FEI *(exploratory)* | 0.907 | 0.903 ± 0.049 | 0.981 | — | 0.741 / 0.728 / 0.778 |
| rand-C *(exploratory)* | 0.852 | 0.876 ± 0.035 | 0.907 | — | 0.907 / 0.900 / 0.963 |
| *ref.* BSI / VJEPA | 0.907 / 0.981 | 0.920 / 0.975 | 0.944 / 1.000 | — | (same prep) |

For FEI+C (clean) the LOSO operating point is Acc 0.933, BAcc 0.944, Sens 0.889, Spec 1.000 (TP 8, FP 0, TN 6, FN 1).

| Comparison (clean prep) | LOSO Δ [95 % CI] | 5-fold Δ (A better in % of repeats) | LPO Δ [95 % CI], p(Δ ≤ 0) |
|---|---|---|---|
| FEI+C vs FEI+C (orig prep) | +0.093 [−0.120, +0.361] | +0.045 (71 %) | +0.111 [0.000, +0.296], 0.06 |
| FEI+C vs rand-FEI+C | −0.019 [−0.182, +0.100] | −0.012 (41 %) | −0.019 [−0.111, 0.000], 1.00 |
| FEI+C vs BSI | +0.019 [−0.240, +0.280] | −0.001 (60 %) | +0.037 [−0.074, +0.185], 0.42 |
| FEI+C+BSI vs BSI | +0.056 [−0.120, +0.296] | +0.031 (82 %) | +0.037 [−0.074, +0.185], 0.42 |
| FEI+C vs VJEPA | −0.056 [−0.273, +0.077] | −0.057 (8 %) | −0.019 [−0.111, 0.000], 1.00 |
| C vs rand-C | −0.019 [−0.120, +0.071] | −0.057 (16 %) | −0.056 [−0.222, +0.111], 0.81 |
| FEI vs rand-FEI | +0.093 [0.000, +0.286] | +0.097 (100 %) | +0.019 [0.000, +0.111], 0.56 |

**Reading:**
- **Defect 1 is fixed and confirmed as the cause of C's collapse.**
  - The input now matches NMT (0.18 vs 5.81 decades).
  - BN stays in range (|z| 0.38 vs 4.3).
  - Pretrained C is no longer collapsed (0.74 × its NMT spread, 3 × random-init).
- **Matching the input helps everything, not pretraining in particular.**
  - FEI+C rises 0.833 → 0.926 LOSO, and LPO 0.870 → 0.981. The subject-level LPO bootstrap puts this at the edge of significance: +0.111, CI [0, 0.296], p 0.06.
  - Its random-init control rises too, 0.889 → 0.944. Pretrained FEI+C still **ties** rand-FEI+C: −0.019, better in 41 % of repeats.
- **FEI+C now ties BSI** (was −0.046). FEI+C+BSI is the best FEI+C arm (0.963), but its gain over BSI alone is inside the CI.
- **The released ViT-M stays the strongest pre-registered arm.** The gap to FEI+C shrinks from −0.101 to −0.057 (5-fold).
- **Branches, exploratory:**
  - **FEI: pretrained beats random-init in 100 % of 5-fold repeats (+0.097).** It reaches 1.000 on all three schemes, but that is a ceiling on 15 subjects in a post-hoc arm, so it is not a claim. The LPO CI touches 0 because rand-FEI is also near the ceiling (0.981).
  - **C: un-collapsed but still no pretraining benefit** (−0.019 LOSO, 16 % of repeats). The 20 s NMT dynamics C learned do not transfer to this task beyond what a random GRU gives.
- **Bottom line:** the clean re-prep turns "FEI+C above chance, below BSI" into "FEI+C on par with BSI". Pretraining's contribution on stroke is carried by FEI, not C, and it is not significant at subject level with n = 15.

- **Primary cohort:** 9 stroke patients (PAC02–10) vs 6 controls (C03, CONTROL04–08), on a fixed 285 s pure-rest block.
- **Sensitivity cohort:** all 18 PRE files, first 300 s.
- **BSI gate:** broadband pdBSI higher in stroke (Mann-Whitney p < 0.05).
- **Arms:** BSI, VJEPA, VJEPA+BSI, rand-VJEPA(+BSI), FEI+C(+BSI), rand-FEI+C(+BSI).
- **Statistics:** 1000 label permutations, bootstrap CIs, and 3 pre-declared paired comparisons.

## Findings / notes

- **The Fig-3 head is not deterministic on the GPU at a fixed seed.** Re-runs diverge from epoch 1; for `attentive_s0`, final AUROC was 72.1 originally and ~64 on the re-run. §4.2-head re-runs are bit-exact.
  - The original runs are kept in `code/runs/tuab_cached/_backup_pre_epochpreds/`, giving a second replicate to quantify this noise.
- **Things the paper leaves unspecified** (our choices, fixed in advance):
  - which 546 training recordings;
  - which epoch is reported (we report the final epoch as primary, best-epoch as optimistic);
  - channel order;
  - whether TUAB-eval was in pretraining (we could not verify).
- **Reproduction audit (2026-09-26): paper + upstream repo vs our pipeline.**
  - **Matches:**
    - The fine-tune recipe equals the authors' `configs/evals/cluster_vitt16_EEG.yaml` exactly (supervised, lr = start_lr 1e-3 → 0, wd 1e-3, batch 2, 500 epochs, fp32, frames 32, step 3, tubelet 4, patch 4×30, vit_small, target_encoder key).
    - The eval transform is the identity (`EEGTransformEval`), and the clip sampling equals `loadvideo_decord`.
    - The checkpoint loads with all keys matched (epoch 353).
    - The subset sizes equal §4.2 (276 N / 270 A train, 150 N / 126 A = official TUAB eval).
    - The fast runner is bit-exact vs upstream `run_one_epoch`.
  - **Unspecified; we chose:**
    - filter at the native rate, then resample (the paper lists downsample then filter);
    - 119 windows per recording (the paper says 118);
    - per-recording channel z-score (the paper's "last step" could be per window).
  - **Tested: the z-score choice does not matter.** Released ViT-M on 120 eval recordings: token-std 0.0059 and cos 1.0000 under both per-recording and per-window z-score (random-init: 0.72 / 0.997). The collapse is a property of the checkpoint, not of our input.
  - **The paper contradicts its own released config:** §5 says full fine-tuning "benefited from lower learning rates and partial unfreezing", so the 88.5 FT number most likely did not come from lr 1e-3 on the whole encoder. The recipe that produced it is unreleased.
- **Code-integrity audit vs upstream GitHub (2026-09-26).** Fresh clone of `github.com/amir-hojjati/eeg-vjepa` at `upstream/eeg-vjepa` (HEAD `c739fad`, the same commit our copy came from; 48 files).
  - **43 of 48 files are byte-identical**, including both configs, the dataset/loader, the ViT, the masks, and the transforms.
  - **5 files differ, and every difference is ours and inert at default settings:**
    - `app/main.py`, `app/vjepa/train.py`: hw_guard only.
    - `attentive_pooler.py`: an added, unused `MeanStdClassifier`.
    - `schedulers.py`: `lr × lr_scale` (default 1.0; set only by `encoder_lr`, which no Act-0 run passes).
    - `eval.py`: seed key (default 0, the same as upstream); head switch (default = upstream `AttentionClassifier`); class weights (default None = upstream unweighted CE); balanced accuracy added to the metrics (AUROC/F1 math unchanged); `train_data_path`/`val_data_path` = upstream's own config keys (we set both to the subset root, so upstream's `mode` → `train/` or `eval/` subfolder logic is unchanged); a random-init option; hw_guard.
  - **Weights:** both released checkpoints' SHA-256 values equal the HF LFS oids.
  - **Imports:** `src` resolves to `code/src`; there is no stale installed copy.
  - **Verdict:** no corruption. Our code behaves exactly like upstream.
  - **The upstream eval has no best-val selection or patience** (500 epochs, `latest.pth.tar` only, per-epoch CSV).
  - **Pretraining saves `jepa-best` = lowest train loss (patience 200)**, and the released file is that one (epoch 353, loss 0.163). Hypothesis, not tested: selecting on the lowest JEPA loss favours the collapsed checkpoint.
- **Preprocessing literature check (2026-09-26).** We compared our pipeline against Schirrmeister 2017, Gemein 2020, Kiessner 2023/24, the braindecode TUH example, ChronoNet, the Lopez 2017 thesis, BIOT, LaBraM, CBraMod, and EEG2Rep, reading primary text and code.
  - **Nothing substantial is missing relative to EEG-VJEPA's own §4.1 spec.**
  - **Open choices the paper leaves unspecified (untested):**
    - *Which 5 minutes.* The braindecode group always drops the first 60 s ("stronger artifacts"). Their official TUH example uses exactly 60–360 s at 100 Hz, a plausible template for "a fixed 5-minute length". We use 0–300 s.
    - *±800 µV clipping.* Used by every braindecode paper, but with no high-pass. Our 1 Hz high-pass already removes most drift, and no paper quantifies the effect.
  - **Already tested:** the z-score scope, per recording vs per window, does not change the collapse.
  - **General TUAB best practice, outside the paper's fixed spec:**
    - keep absolute amplitude, µV or /100 µV, not z-score: the biggest measured lever, band power 0.773 → 0.808 with log-absolute power;
    - use ≥ 20 min: +0.008 for us, larger for deep models per Schirrmeister Fig. 4;
    - use a wider band with a notch: ≈ +0.01 in our CBraMod-prep diagnostic;
    - montage and 100 Hz: negligible.
  - **The P1.3 band-power shortfall (0.73–0.76) is mostly the feature set** (relative power drops amplitude), not a prep bug.
  - **The foundation models' 0.90–0.92 AUROC is window-level on full train**, so it is not comparable to recording-level numbers.
  - **Caveat:** the collapse is a checkpoint property, so no prep change can recover the paper's frozen 87.7 from the released weights.
- **Liu2024 (exploratory, stopped as out of scope).** Frozen encoders on left-vs-right MI are all at chance: joint FEI+C 51.7%, NMT-only 49.5%, random-init 51.3%, band power 53.1%, CSP+LDA 50.8%, FgMDM 52.2% (joint vs random +0.4 pt, p = 0.97). Consistent with the Act-2 finding that frozen SSL features carry no MI information. Paralysis side: only the power-asymmetry baseline ran (AUROC 0.659, perm p = 0.075); encoder arms not run. Partial outputs in `code/runs/act0/liu2024_feic/`.
- **Operational:**
  - The CPU temperature guard aborted two jobs at 97 °C when another application heated the CPU; both were resumed without loss.
  - CPU-heavy prep now runs single-threaded next to GPU jobs.

## Artifacts (for later analysis)

| Output | Path |
|---|---|
| P1.4 per-run epoch CSVs, `final_preds.npz` (with `prob_abnormal_by_epoch`, 500 × 276; ViT-B s0 final epoch only), `head.pt` | `code/runs/tuab_cached/<tag>/` |
| P1.4 summary | `code/runs/tuab_cached/summary.txt` |
| P1.3 band power (predictions, features, diagnostic) | `code/runs/act0/p13_bandpower/` |
| A0 (per-fold predictions + embeddings) | `code/runs/act0/a0_feic_nmt/` |
| ViT-M linear probe (pooled features, predictions, collapse, paired) | `code/runs/act0/tuab_vitm_linear/` |
| FEI+C on TUAB (embeddings + index, per-member predictions, collapse) | `code/runs/act0/tuab_feic/` |
| Phase B (gate, embeddings, collapse, results, per-member out-of-fold predictions + permutation nulls) | `code/runs/act0/phaseB/` |

Live status:

```bash
watch -n 30 /home/mashfiq/eeg_vjepa/code/runs/act0/status.sh
```
