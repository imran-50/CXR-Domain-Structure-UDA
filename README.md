# Characterizing Domain Structure to Anticipate Unsupervised Domain Adaptation Performance for Pneumonia Detection from Chest X-ray

Official PyTorch implementation and reproduction package for the manuscript:
> **"Characterizing Domain Structure to Anticipate Unsupervised Domain Adaptation Performance for Pneumonia Detection from Chest X-ray"**  
> *Iftakhar Hossain\*, Md. Imran Hasan\*, and Taufiq Hasan (BUET mHealth Lab)*  
> (\* Equal contribution)

---

## Quick Reproduction (< 5 seconds)

Verify all exact permutation p-values, rank correlations, and Tables III to IX from the manuscript without raw images or a GPU:

```bash
git clone https://github.com/imran-50/CXR-Domain-Structure-UDA.git
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

## Citation

If you find this work, codebase, or methodology useful in your research, please cite:

> **Iftakhar Hossain\*, Md. Imran Hasan\*, and Taufiq Hasan**, *"Characterizing Domain Structure to Anticipate Unsupervised Domain Adaptation Performance for Pneumonia Detection from Chest X-ray"*, 2026 (*In Preparation*).  
> (\* Equal contribution)

```bibtex
@misc{hossain2026characterizing,
  author       = {Iftakhar Hossain and Md. Imran Hasan and Taufiq Hasan},
  title        = {Characterizing Domain Structure to Anticipate Unsupervised Domain Adaptation Performance for Pneumonia Detection from Chest X-ray},
  year         = {2026},
  howpublished = {\url{https://github.com/imran-50/CXR-Domain-Structure-UDA}},
  note         = {In Preparation}
}
```
