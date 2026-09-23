# The Acoustic Fingerprint of Pipeline Leaks

Official code repository for the paper:

> **The Acoustic Fingerprint of Pipeline Leaks: Spectral–Physical Characterization and Group-Aware Calibrated Fusion for Small-Sample Leak Detection**
> Ziming Yang. *Submitted to ICASSP 2026.*

This repository contains the complete, end-to-end reproducible pipeline behind
the paper: from raw 1-second 8 kHz audio clips, through a 212-dimensional
handcrafted acoustic feature set and a waveform 1D CNN, to a frozen,
group-aware, Platt-calibrated SVM+CNN fusion — plus every post-freeze audit
(LOCO/LORO generalization, SNR degradation, wav2vec 2.0 baseline, calibration /
threshold-fairness / negative-source audits) and the paper figures.

## Data availability

The dataset (1,000 clips: 500 leak, 386 no-leak, 114 noise; each 1 s at 8 kHz)
is publicly available on Zenodo:

**https://doi.org/10.5281/zenodo.18631450**

After downloading and extracting, arrange the clips as:

```
data/
├── leak/     *.wav
├── no_leak/  *.wav
└── noise/    *.wav
```

All scripts locate the repository root relative to their own path
(`Path(__file__).resolve().parents[1]`), so no path configuration is needed.

## Environment

Python 3.11+ is recommended (a conda environment works well):

```bash
conda create -n leak python=3.11
conda activate leak
pip install -r requirements.txt
```

> **⚠️ Version warnings — read before installing:**
>
> - **scipy must be ≤ 1.15.x.** Under scipy 1.18, scikit-learn's
>   `LogisticRegression(lbfgs)` crashes at the process level (not a Python
>   exception) inside the nested Platt calibration loops.
> - **numpy must be 2.2.x.** numpy 2.4.6 crashes inside `einsum` during
>   feature/model evaluation.
> - **torch 2.10** was tested. A GPU is optional: the CPU runs everything, but
>   the deep models (1D CNN retraining in `stability_selection.py`, LOCO/LORO,
>   wav2vec 2.0 fine-tuning) are slow without one.
> - `src/wav2vec2_baseline.py` downloads the torchaudio `WAV2VEC2_BASE`
>   weights on first run — Internet access is required once.

## Reproduce the paper

Run the scripts in this order from the repository root. Each step lists its
main output directory and the paper items it produces.

```bash
# 0. (optional) data sanity check: counts, sample rates, durations
python src/check_audio.py

# 1. 212-d handcrafted acoustic statistics for all 1,000 clips -> features/
python src/extract_features.py

# 2. group-isolated 800/200 train/verification split (seed 42) -> train_test_data/
python src/make_group_split.py

# 3. acoustic-signature / spectral-physics analysis -> results_signature/
#    (Section II, Table 1, Figs. 1-3)
python src/acoustic_signature.py

# 4. 7-model group-aware OOF model selection -> results_model_selection/
#    (Section III)
python src/model_selection_oof.py

# 5. 12 candidate heterogeneous fusion pairs + nested group-aware Platt
#    calibration -> results_fusion_pairs/
python src/fusion_pair_search.py

# 6. repeated (5 partitions x 6 models x 5 folds) stability selection
#    -> results_stability/   (Table 2)
#    WARNING: heavy — full retraining across all repeats; GPU recommended.
python src/stability_selection.py

# 7. frozen verification: fixed 0.55/0.45 fusion, tau = 0.520
#    -> results_frozen_verification/, models_frozen/   (Table 3, main result)
#    This is the ONLY step that reads the 200-clip verification partition.
python src/frozen_verification.py

# 8. leave-one-condition-out and leave-one-region-out generalization
#    -> results_loco/, results_loro/   (Fig. 5(c) and cross-zone transfer)
python src/leave_one_condition_out.py
python src/leave_one_region_out.py

# 9. post-freeze additive-noise degradation (white/pink, +20...-5 dB)
#    -> results_snr_degradation/   (Fig. 5(a,b))
python src/snr_degradation.py

# 10. wav2vec 2.0 baseline -> results_wav2vec2/   (the w2v2 rows of Table 3)
python src/wav2vec2_baseline.py

# 11. Section IV-D audits (calibration, threshold fairness, negative sources)
#     -> results_audit_calibration/, results_audit_threshold_fairness/,
#        results_audit_negative_sources/
python src/audit_calibration.py
python src/audit_threshold_fairness.py
python src/audit_negative_sources.py

# 12. paper figures -> figures/
python src/fig_pipeline.py
python src/fig_robustness.py

# 13. frozen verification of the 5 ablation fusion pairs (P1..P5)
#     -> results_fusion_pairs_verification/, models_fusion_pairs/
#     (Table 3, middle block)
#     requires step 7 (reads models_frozen/ and
#     results_frozen_verification/fixed_test_predictions.csv)
python src/fusion_pairs_verification.py

# 14. paired statistics ours (P0) vs each ablation pair: exact McNemar,
#     5000x group-cluster bootstrap CIs (dF1, dAUC), plus Brier /
#     log-loss / 15-bin ECE for all six pairs
#     -> results_verification_stats/
#     requires steps 7 and 13 (statistics only; no retraining)
python src/verification_comparison_stats.py

# 15. faithful re-run of the recorded wav2vec 2.0 fine-tune (the baseline
#     script does not persist weights), with a consistency gate against
#     the recorded 0.950/0.950/0.9676/5/5; saves weights and per-sample
#     probabilities -> models_w2v2_finetuned/, results_w2v2_stats/
#     requires steps 7 and 10; uses the locally cached WAV2VEC2_BASE
#     backbone (no download)
python src/w2v2_finetune_verify.py

# 16. ours vs wav2vec 2.0 fine-tuned: w2v2 calibration (Brier / log-loss /
#     ECE), exact McNemar + group-bootstrap CIs (produced by step 15),
#     parameter-count comparison and FN-cost sensitivity
#     -> results_w2v2_stats/
#     requires steps 7, 10 and 15
python src/w2v2_verification_stats.py
```

### extras/

`extras/` contains the dynamic-gating experiments that the paper describes as
**"tested and rejected"** — *"Two dynamic gating schemes (SNR-aware and
reliability-aware) were tested and rejected under nested meta-validation"* —
plus the convergence diagnostics for the three selected fusion structures.
They are kept for verifiability. They read artifacts of the main chain
(e.g. `results_stability/`), so run the main chain first. See
[extras/README.md](extras/README.md).

## Protocol constants

The scripts assert the fixed protocol of the Zenodo dataset above:

- **1,000 clips** (500 leak / 386 no-leak / 114 noise), 8 kHz, 1 s (8,000 samples)
- **800 train / 200 verification** clips, group-isolated (seed 42)
- **370 training groups / 91 verification groups** (group id derived from filenames)
- **212-dimensional** handcrafted feature vector
- frozen fusion **p = 0.55·p_SVM + 0.45·p_CNN**, decision threshold **τ = 0.520**

These constants (and seeds such as `BOOTSTRAP_SEED`) are part of the published
protocol, not tuning knobs. If you use a different dataset, you must update
them and re-derive the frozen fusion parameters from scratch — do not reuse
0.55/0.45/0.520 on new data.

## Repository structure

```
acoustic-leak-fingerprint/
├── README.md
├── LICENSE                      MIT (c) 2026 Ziming Yang
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── src/                         main reproduction chain (run in order)
│   ├── check_audio.py           dataset sanity check
│   ├── extract_features.py      212-d handcrafted features -> features/
│   ├── make_group_split.py      800/200 group-isolated split -> train_test_data/
│   ├── acoustic_signature.py    spectral-physical signature analysis (Sec. II)
│   ├── model_selection_oof.py   7-model group-aware OOF selection
│   ├── fusion_pair_search.py    12 fusion pairs + nested Platt calibration
│   ├── stability_selection.py   repeated stability selection (Table 2)
│   ├── frozen_verification.py   frozen 0.55/0.45 @ tau=0.520 (Table 3)
│   ├── leave_one_condition_out.py  LOCO generalization
│   ├── leave_one_region_out.py     LORO (cross-zone) generalization
│   ├── snr_degradation.py       additive-noise robustness (Fig. 5a,b)
│   ├── wav2vec2_baseline.py     pretrained wav2vec 2.0 baseline
│   ├── audit_calibration.py     calibration / reliability audit (Sec. IV-D)
│   ├── audit_threshold_fairness.py  threshold-fairness audit (Sec. IV-D)
│   ├── audit_negative_sources.py    negative-source sensitivity (Sec. IV-D)
│   ├── fusion_pairs_verification.py  5 ablation fusion pairs (Table 3)
│   ├── verification_comparison_stats.py  McNemar/bootstrap + calibration
│   ├── w2v2_finetune_verify.py  w2v2 fine-tune re-run + weight persistence
│   ├── w2v2_verification_stats.py  ours vs w2v2-FT paired stats + cost
│   ├── fig_pipeline.py          pipeline schematic -> figures/
│   └── fig_robustness.py        robustness figure -> figures/
├── extras/                      tested-and-rejected dynamic gating + diagnostics
│   ├── README.md
│   ├── snr_mechanism.py
│   ├── reliability_mechanism.py
│   ├── reliability_gate_meta_cv.py
│   ├── reliability_joint_gate_meta_cv.py
│   └── fusion_convergence_diagnostics.py
├── data/                        (you provide; Zenodo DOI 10.5281/zenodo.18631450)
├── features/  train_test_data/  results_*/  models_frozen/  figures/
                              (all produced by the scripts above; git-ignored)
```

Output directories are likewise renamed (e.g. `results_training_only_selection_11`
→ `results_model_selection`, `results_frozen_svm_cnn1d_14` → `results_frozen_verification`,
`results_leave_one_condition_out_20` → `results_loco`,
`results_all_fusion_pairs_verification_30` → `results_fusion_pairs_verification`,
`results_verification_comparison_stats_31` → `results_verification_stats`,
`results_w2v2_verification_stats_32` → `results_w2v2_stats`,
`models_all_fusion_pairs_30` → `models_fusion_pairs`,
`models_w2v2_finetuned_33` → `models_w2v2_finetuned`); each script prints its
own output directory when run.

## License

This code is released under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code or the dataset, please cite:

```bibtex
@inproceedings{yang2026acoustic,
  author    = {Yang, Ziming},
  title     = {The Acoustic Fingerprint of Pipeline Leaks:
               Spectral--Physical Characterization and Group-Aware Calibrated
               Fusion for Small-Sample Leak Detection},
  booktitle = {Submitted to ICASSP},
  year      = {2026},
  note      = {Code: https://github.com/Steven-Yang05/acoustic-leak-fingerprint}
}
```

Dataset: https://doi.org/10.5281/zenodo.18631450
