# Characterizing Datasets to Anticipate Unsupervised Domain Adaptation Performance for Radiographic Pneumonia Detection from Chest X-rays

Official PyTorch implementation and reproduction package for the manuscript:
> **"Characterizing Datasets to Anticipate Unsupervised Domain Adaptation Performance for Radiographic Pneumonia Detection from Chest X-rays"**  
> *Iftakhar Hossain\*, Md. Imran Hasan\*, Eric D. McCollum, Abdullah H. Baqui, and Taufiq Hasan^\dagger (Senior Member, IEEE)*  
> (\* Equal contribution, ^\dagger Corresponding author)

---

## Quick Reproduction (< 5 seconds)

Verify all exact permutation p-values, rank correlations, and Tables III to IX from the manuscript without raw images or a GPU:

```bash
git clone https://github.com/mHealthBuet/CXR-Domain-Structure-UDA.git
cd CXR-Domain-Structure-UDA
pip install -r requirements.txt
python reproduce_paper_tables.py
```

---

## Key Scientific Findings

1. Dual Endpoint Contribution: Domain adaptation performance is shaped by both endpoints in opposite directions:
   - Target Separability: Higher target class separability yields higher incoming AUC (r = 0.934, p = 0.042).
   - Source Overlap: More overlapped sources transfer more strongly (r = -0.922, p = 0.017).
2. Direction Prediction (Table VIII):
   - The signed difference in separability TP(S -> T) = s(T) - s(S) predicts the superior transfer direction in 10 out of 10 pairs under adaptation (p = 0.0010, exact binomial).
3. Benchmarking Against 7 Published Estimators (Table III & IV):
   - Compared against LogME, H-score (and shrunk H-score), TransRate, NLEEP, PACTran-G, SFDA, and GBC. All 9 measures establish identical dataset rankings.
4. Primary Feature Representation:
   - DenseNet-121 trained on NIH ChestX-ray14 (densenet121-res224-nih) as primary; CheXpert weights (densenet121-res224-chex) as robustness.

---

## Datasets Summary (5 Clinical Cohorts)

| Dataset | Population | Normal | Pneumonia | Total | Train Split | Access |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Kermany | Pediatric (China) | 1,583 | 4,273 | 5,856 | 5,232 | Open Access (Cell 2018) |
| NIH-RSNA | Adult (USA) | 3,350 | 3,350 | 6,700 | 6,000 | Open Access (RSNA Challenge) |
| PadChest | Adult (Spain) | 3,350 | 3,350 | 6,700 | 6,000 | Open Access (BIMCV-PadChest) |
| CheXpert | Adult (USA) | 3,350 | 3,350 | 6,700 | 6,000 | Open Access (Stanford AIMI) |
| PRF (Sylhet) | Pediatric (Bangladesh) | 2,777 | 1,999 | 4,776 | 4,107 | Clinical Study / Formal DUA |

---

## Repository Structure

```text
CXR-Domain-Structure-UDA/
|-- Figure/                              # Manuscript vector figures
|-- configs/                             # Cohort configurations and path resolution
|-- data/transfer_results/               # Precomputed transfer AUC reports
|-- outputs/                             # Extracted descriptor CSVs (Table III-IX)
|-- src/                                 # Feature extractors and transferability estimators
|-- scripts/                             # Ordered pipeline scripts (01-06)
|-- reproduce_paper_tables.py            # Master verification runner
|-- requirements.txt                     # Python dependencies
|-- LICENSE                              # MIT License
`-- README.md
```

---

## Pipeline Scripts (01 to 06)

- `scripts/01_extract_features_and_descriptors.py`: Extracts 1024-d DenseNet features and computes all 8 estimators.
- `scripts/02_statistics_and_tables.py`: Computes exact permutation tests and outputs Tables III through IX.
- `scripts/03_balanced_descriptors.py`: Executes Section IV-F class-balance control subsampling on Kermany and PRF.
- `scripts/04_train_zero_shot.py`: Trains in-domain source classifiers.
- `scripts/05_eval_zero_shot.py`: Evaluates direct cross-domain zero-shot performance across all 25 pairings.
- `scripts/06_train_eval_dann.py`: Executes adversarial domain adaptation (DANN) across all 20 directed pairs.
