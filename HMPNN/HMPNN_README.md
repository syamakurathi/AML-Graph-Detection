# HMPNN IBM HI-Small experiment

Open `HMPNN_IBM_HI_Small.ipynb` and select the project's `.venv` Python kernel.
The delivered notebook contains executed outputs. To rerun, use **Run All** from
the project root or `HMPNN` directory. A rerun replaces the files in `hmpnn_results`.

From the project root, the equivalent command is:

```powershell
.venv/Scripts/python.exe HMPNN/run_hmpnn_notebook.py
```

Dependencies for the executed environment are pinned in `requirements-hmpnn.txt`.
The source notebook can be regenerated with `build_hmpnn_notebook.py`, but doing so
clears its execution outputs. Regeneration is only needed when editing the builder.

## Method

The model classes are the original two-layer HMPNN-sum and HMPNN-ct implementations
from https://github.com/fredjo89/heterogeneous-mpnn at commit
`4f56e6a38a7e35fd186393c1849f2711392c908b`. Original source and MIT license are kept in
`vendor/heterogeneous-mpnn`.

The experiment adapts their inputs to local IBM HI-Small data:

- A seeded, uniform sample of 100,000 transactions, selected without consulting labels.
- Account and entity node types; payment-format, reverse-payment and ownership edges.
- An account label indicating involvement in a positive transaction in the sample.
- A static, transductive, stratified 60/20/20 account split, not a chronological split.
- Label-free node and edge features; fitted scalers use training accounts or their links.
- Adam and unweighted BCE; a deep-copied best validation-loss checkpoint.
- A validation-selected F1 threshold, then held-out test evaluation.
- A logistic regression baseline on the same account features and split.

The original IMDB demo uses three layers and a longer epoch budget. This experiment
uses two layers and at most 300 epochs per model to run on CPU. The private DNB data
from the paper is unavailable, so these IBM scores are not a replication of its
reported results. The notebook lists additional limitations, including sampled proxy
labels, entities shared across splits, and the single-seed evaluation.
One labeled transaction can also mark accounts in different splits positive, so
account labels are correlated and this is not an independent transaction holdout.

## Saved results

- `metrics.csv`: validation/test AP, ROC-AUC, F1, precision, recall, accuracy,
  confusion counts, threshold-0.5 F1, and Precision/Recall@100, 500, 1000.
- `predictions.csv`, `account_splits.csv`: generated locally when rerunning; excluded from the repository because they contain account-level records.
- `split_summary.csv`: test and validation counts. Account-level membership and sampled row exports are generated locally but omitted from the repository.
- `HMPNN_*_history.csv`, `HMPNN_*.pt`: learning curves and selected checkpoints.
- `graph.pt`, `preprocessing.pkl`: generated locally when rerunning; excluded as regenerable intermediate artifacts.
- `run_manifest.json`: settings, versions, timings, thresholds and source hashes.
- `evaluation.png`: precision-recall, ROC and validation-loss plots.

`PR_AUC_AP` is sklearn average precision, rather than trapezoidal PR integration.
Report test rows, their positive counts, and the stated sampling/split protocol together.
Check `hit_epoch_cap` before making any claim about convergence.
