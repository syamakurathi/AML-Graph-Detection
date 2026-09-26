# Recorded HMPNN IBM HI-Small results

Executed and verified locally using a 100,000-transaction uniform sample, seed 42, and a static stratified 60/20/20 account split. The test set has 23,814 accounts and 38 positives. These are IBM adaptation results, not the paper's DNB scores.

| Model | AP (PR-AUC) | ROC-AUC | F1 | Precision | Recall | Recall@1000 |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.015572 | 0.650003 | 0.050000 | 0.500000 | 0.026316 | 0.078947 |
| HMPNN_sum | 0.003810 | 0.672603 | 0.000000 | 0.000000 | 0.000000 | 0.184211 |
| HMPNN_ct | 0.011529 | 0.762060 | 0.000000 | 0.000000 | 0.000000 | 0.394737 |

The no-skill AP reference is 0.001596 (test prevalence). HMPNN-ct has the highest ROC-AUC; logistic regression has the highest AP. Both HMPNN models detect zero test positives at their validation-F1-selected thresholds. HMPNN-ct ranks 15 of the 38 positives in its top 1,000 accounts, but thresholded detection remains poor.

Both HMPNN variants reached the 300-epoch cap with their best validation BCE at epoch 300. Convergence has not been established. The graph uses shared static context, labels are sampled retrospective account proxies, and accounts/entities can span splits. One labeled transaction can mark both endpoints in different splits. These limitations and the small positive count prevent strong generalization claims.

## Verification

All notebook code cells executed without error. Exported AP, ROC-AUC, F1, top-K metrics and validation-selected thresholds were independently recomputed. Reloaded HMPNN checkpoints reproduce exported probabilities within numerical tolerance. The evaluation figure was visually inspected.

The local run was independently checked by recomputing the metrics and reloading the trained checkpoints. Rerun the notebook to recreate local account-level predictions and graph artifacts. See `../HMPNN_README.md` for instructions and `run_manifest.json` for versions, hashes and timings.


Account-level prediction exports, sampled account membership, graph serialization and preprocessing objects are reproducible from the notebook but are kept out of this source repository.
