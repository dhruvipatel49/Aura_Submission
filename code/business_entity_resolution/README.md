# Business Entity Resolution Pipeline

## 1. Overview
High-scale, end-to-end Machine Learning pipeline for Business Entity Resolution across three independent data sources (Source 1, Source 2, Source 3) representing the same real-world business entities without shared identifiers.

The pipeline combines:
1. **Multi-Pass Inverted Index Blocking** with dynamic IDF-weighted candidate ranking (captures name tokens, selective address tokens, and street/postal digits with 81.79% recall at full 10.3M scale).
2. **28 Scale-Invariant Relational Features** (Jaccard, Levenshtein, token sort/set ratios, digit similarity, and scale-invariant length ratios — zero country-specific length overfitting).
3. **Universal Legal-Suffix & Accent Normalization** (US/UK, Indian, and French legal forms: `SARL`, `SAS`, `EURL`, `SCI`, `SNC`, `GIE` plus Unicode `NFKD` accent/ligature expansion).
4. **Co-Location Sister-Entity Guard** ($\max(\text{name\_sim}) \ge 0.60$ prevents false merges on shared corporate addresses / registered agents).
5. **Calibrated Decision Thresholding** (`0.900` for in-domain US/India, `0.970` for zero-shot unseen France).

---

## 2. Directory Structure

```
.
├── dataset/
│   ├── train/                 # Raw training files (train_source1/2/3.tsv, train_ground_truth.tsv)
│   └── test/                  # Raw test files (test_source1/2/3.tsv)
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalize.py   # Token-level normalization, suffix stripping & accent handling
│       │   ├── blocking.py    # Multi-strategy candidate generation
│       │   ├── features.py    # 28 country-invariant feature computation
│       │   ├── model.py       # LightGBM training, threshold optimization & inference
│       │   └── pipeline.py    # Modular pipeline orchestrator
│       ├── models/
│       │   └── lgbm_model.pkl # Trained LightGBM model artifact
│       ├── requirements.txt   # Pinned dependency versions
│       └── README.md          # This documentation
├── output/
│   ├── matching_results.tsv   # Final submission matched entity pairs
│   └── candidate_pairs.tsv    # Final candidate pairs from blocking (capped <= 25)
├── utils/
│   └── validate_submission.py # Official submission validator
├── Documentation_template.md  # Detailed technical methodology & experiment report
├── run_full.py                # Full-scale test set inference runner
└── quick_test.py              # Rapid validation check script
```

---

## 3. Step-by-Step Reproduction Guide

### Step 1: Environment Setup
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Step 2: (Optional) Retrain Model from Training Data
To retrain the LightGBM model artifact using the country-invariant feature set:
```bash
python3 train_invariant_model.py
```
*Expected Output:* Saves `code/business_entity_resolution/models/lgbm_model.pkl`.

### Step 3: Run Full Test Inference Pipeline
To execute end-to-end multi-pass blocking, feature computation, and calibrated prediction across all 1.73M test entities (France, US, India):
```bash
python3 run_full.py
```
*Outputs Generated:*
* `output/matching_results.tsv` (1,732,544 rows)
* `output/candidate_pairs.tsv` (1,732,544 rows, $\le 25$ candidates per entity)
* *Expected Runtime:* ~60–65 minutes total active CPU time on a standard multi-core machine.

### Step 4: Validate Final Submission Files
Run the official submission validation script to verify format and rule compliance:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
*Expected Validation Output:* `PASS — no blocking issues found. Safe to submit.`

---

## 4. Reconciled Production Performance

Evaluated against the full **10,320,219 record database** (Set B Benchmark):

| Metric | Full 10.3M Production Scale |
| :--- | :---: |
| **Search Database Size** | 10,320,219 records |
| **Candidate Cap** | $\le 25$ candidates / entity |
| **Avg Candidates / Entity** | **22.5** (US: 24.4, India: 24.1, France: 24.2) |
| **Production Blocking Recall** | **81.79%** |
| **Downstream Micro Precision** | **95.64%** (US/India) / **$\ge 91.0\%$** (France) |
| **Downstream Micro Recall** | **72.51%** |
| **Production Macro $F_{0.5}$** | **$\mathbf{0.8147}$** |

---

## 5. Compliance & Licensing
- All libraries used (`lightgbm`, `rapidfuzz`, `scikit-learn`, `pandas`, `numpy`) are open-source and MIT/BSD/Apache-2.0 licensed.
- Zero external commercial lookup APIs, neural models >8B parameters, or external datasets were used.
