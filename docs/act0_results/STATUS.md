# Act-0 Results Analysis PDF — DONE 2026-09-27

Output: `docs/act0_results/Act0_Results_Analysis.pdf` (23 pp, INTERNAL — TUAB unlicensed; SendUserFile, never Artifact).

Rebuild (CPU only):
    /home/mashfiq/.eeg_vjepa_venv/bin/python docs/act0_results/make_results.py   # numbers.json, tables.json, figs/
    /home/mashfiq/.eeg_vjepa_venv/bin/python docs/act0_results/build_pdf.py      # analysis.md -> PDF

- `analysis.md` = text; every result is a `{{key}}` (numbers.json) or `{{t:name}}` (tables.json) placeholder,
  build_pdf.py raises on any missing key. Figure alt text becomes the caption.
- Self-reviewed: all 12 figures + every page rendered. No agents.
- Framing rules (user-approved): no buggy results, only the one neutral post-hoc-prep sentence (Exp 8);
  FEI+C ties ViT-M on TUAB; FEI > ViT-M borderline + not pre-declared;
  C helps in-domain (NMT) but its pretraining doesn't transfer (rand-C ties); demographic confound in limits.
- Stroke part (Exps 7-8) reports NO EEG-VJEPA arms : Exp 7 = gate + BSI baseline, Exp 8 = FEI+C vs BSI / rand-FEI+C.
  VJEPA stroke data is read only inside make_results.py consistency asserts. Exp 3 (Part A) still shows stroke collapse rows.
- "Visual check: t-SNE and UMAP maps" section (end of Part B, glance row V): TUAB eval, 4 feature sets
  (released/rand0 ViT-M, FEI+C f0 / rand0 from tuab_feic_joint), 10-NN label agreement ≈0.6 everywhere (chance 0.502) = no class
  clusters; random-init maps equally structured. Needs umap-learn 0.5.7 (pinned; 0.5.12 would force sklearn 1.6 — keep sklearn 1.5.2).
