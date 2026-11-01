# Characterizing Domain Structure to Anticipate Unsupervised Domain Adaptation Performance for Pneumonia Detection from Chest X-ray

Official PyTorch implementation and reproduction package for the manuscript:
> **"Characterizing Domain Structure to Anticipate Unsupervised Domain Adaptation Performance for Pneumonia Detection from Chest X-ray"**  
> *Iftakhar Hossain\*, Md. Imran Hasan\*, and Taufiq Hasan (BUET mHealth Lab)*  
> (\* Equal contribution)

---

## Quick Reproduction (< 5 seconds)

Verify all exact permutation p-values, rank correlations, and Tables II to VII from the manuscript on any CPU:

```bash
git clone https://github.com/imran-50/CXR-Domain-Structure-UDA.git
cd CXR-Domain-Structure-UDA
pip install -r requirements.txt
python reproduce_paper_tables.py
```

---

## Cohort Summary (5 Clinical Datasets)

- **Kermany**: Pediatric chest radiographs (Cell 2018)
- **PRF (Sylhet)**: Pediatric study in Sylhet, Bangladesh
- **CheXpert**: Adult chest radiographs (Stanford AIMI)
- **NIH-RSNA**: Adult chest radiographs (NIH Clinical Center / RSNA)
- **PadChest**: Adult chest radiographs (BIMCV-PadChest, 8-bit)

---

## 📖 Citation

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
