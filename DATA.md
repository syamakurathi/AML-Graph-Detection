# Dataset setup

Both experiments expect the HI-Small transaction and account CSVs in `Data/` at the repository root:

- `Data/HI-Small_Trans.csv`
- `Data/HI-Small_accounts.csv`

Get the data through IBM's [AML-Data repository](https://github.com/IBM/AML-Data), which links to the [Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and the [original IBM Box archive](https://ibm.box.com/v/AML-Anti-Money-Laundering-Data). Extract the HI-Small transaction and account files into `Data/`. The optional `HI-Small_Patterns.txt` annotations are not loaded by either experiment.

The data is synthetically generated. IBM states that the data files are licensed separately under [CDLA-Sharing-1.0](https://spdx.org/licenses/CDLA-Sharing-1.0.html); the Apache-2.0 license on IBM's repository applies to repository materials, not the actual data. Review the applicable data license before redistributing datasets or derived data. Raw input files are excluded from this repository, and `/Data/` is ignored by Git.
