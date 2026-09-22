# extras/ — tested-and-rejected dynamic gating + convergence diagnostics

These scripts are **not** part of the main reproduction chain in `src/`.
They document the dynamic-gating ideas that the paper reports as rejected:

> *"Two dynamic gating schemes (SNR-aware and reliability-aware) were tested
> and rejected under nested meta-validation."*

They are preserved here so the rejection claims in the paper can be verified
end-to-end.

## Dependencies on the main chain

All scripts below read artifacts produced by the main pipeline — in
particular `results_stability/repeated_training_oof_probabilities.csv`
(from `src/stability_selection.py`) and, for the gate meta-CV scripts, the
decision JSONs of the earlier extras steps. **Run the main chain (at least
through `src/stability_selection.py`) first**, then run these in the order
listed.

## Contents

| Script | What it does | Paper statement it backs |
|---|---|---|
| `snr_mechanism.py` | Tests whether an acoustic-quality (SNR) proxy has a reproducible, monotone relationship with the relative OOF advantage of the calibrated RBF-SVM over the 1D CNN. Training-only; no gate is tuned. | The SNR-aware gating scheme was rejected because the underlying mechanism did not hold up reproducibly. |
| `reliability_mechanism.py` | Tests whether an expert-reliability signal (`2*|p_svm-0.5| - 2*|p_cnn-0.5|`) predicts which expert is right, per group. Training-only; no gate is tuned. | Same rejection basis for the reliability-aware scheme. |
| `reliability_gate_meta_cv.py` | The reliability-aware dynamic gate itself: `p = w(r)·p_SVM + (1-w(r))·p_CNN` with steepness β, evaluated under nested group-aware meta-CV at the fixed threshold τ = 0.520. | "tested and rejected under nested meta-validation" — the gate does not beat the frozen 0.55/0.45 fusion. |
| `reliability_joint_gate_meta_cv.py` | Stronger variant that jointly tunes β **and** the decision threshold under meta-CV, separating threshold gain from gate gain (methods A/B/C). | Even with a jointly tuned threshold, the dynamic gate is rejected. |
| `fusion_convergence_diagnostics.py` | Convergence diagnostics for the three fusion structures that each won at least one stability repeat (RBF-SVM+1D CNN, RBF-SVM+2D CNN, MLP+2D CNN): SVM solver termination plus epoch-by-epoch clean/OOF weighted-BCE curves. | Supports that the comparison among candidate structures was not confounded by non-convergence. |

## Usage

```bash
# from the repository root, after the main chain has been run
python extras/snr_mechanism.py
python extras/reliability_mechanism.py
python extras/reliability_gate_meta_cv.py
python extras/reliability_joint_gate_meta_cv.py
python extras/fusion_convergence_diagnostics.py
```

Each script supports `--check-only` (verify inputs without computing) and
writes to its own `results_*` directory (`results_snr_mechanism`,
`results_reliability_mechanism`, `results_reliability_gate`,
`results_reliability_joint_gate`, `results_convergence`). None of them reads
the fixed 200-clip verification partition.
