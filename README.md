# Skin Lesion Cross-Dataset Framework

An open-source, reproducible pipeline for classifying melanoma versus other skin lesions in dermoscopic images. It uses leakage-safe splits and **cross-dataset validation**: a model is trained on HAM10000 and tested on an independent dataset (ISIC 2020), and the pipeline checks whether colour normalisation and colour augmentation reduce the drop in performance.

   **Status:** version 1.0.1. The archived release and its DOI are listed below.

- Archived release (Zenodo): https://doi.org/10.5281/zenodo.24052375
- Paper: in preparation

## Why cross-dataset validation

A model that scores well on its own dataset can fail on new data. In this work, ROC-AUC fell from about 0.90 on the HAM10000 test set to about 0.65 on ISIC 2020 for EfficientNet-B0, and to about 0.71 for ResNet-50. This repository makes that check part of the pipeline.

## What is here

| Path | Content |
|---|---|
| `skinfw/` | The framework: splits, colour normalisation, data, models, training, metrics |
| `scripts/` | Command-line tools: download, splits, train, evaluate, summarise, figures |
| `configs/default.yaml` | All settings (seed, epochs, learning rate, variant, backbone, paths) |
| `splits/` | Reference split files (lesion-level HAM10000, patient-level ISIC 2020) |
| `results/` | Per-run scores for the reported experiments |
| `tests/` | Tests that need no GPU and no data |

## Variants and networks

- `base`: mild augmentation
- `sog`: Shades-of-Gray colour normalisation
- `coloraug`: strong colour jitter and random grayscale
- `both`: `sog` and strong colour augmentation
- Networks: `efficientnet_b0` (default) and `resnet50`, both with ImageNet weights

## Quick start (Google Colab, T4 GPU)

```bash
git clone https://github.com/surekhadhanushkodi-collab/skin-lesion-crossdataset-framework.git
cd skin-lesion-crossdataset-framework
pip install -q -r requirements.txt
```

```python
import os
from google.colab import userdata
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")   # your Kaggle token, never commit it
```

```bash
python scripts/download_data.py                 # downloads both datasets (about 6 GB)
python scripts/make_splits.py --check           # verifies the stored splits can be regenerated
python scripts/train.py --variant both --seed 42                        # EfficientNet-B0
python scripts/train.py --backbone resnet50 --variant both --seed 42    # ResNet-50
python scripts/evaluate.py --run runs/both_seed42 --set isic_final --ref runs/base_seed42
python scripts/summarize_seeds.py
python scripts/make_figures.py --runs runs --out figures
python scripts/make_figures.py --runs runs --out figures_resnet50 --backbone resnet50
python -m pytest -q                             # tests, no GPU or data needed
```

A run saves its model, threshold, predictions and scores to `runs/<variant>_seed<seed>/` (EfficientNet-B0) or `runs/resnet50_<variant>_seed<seed>/` (ResNet-50).

## Method in brief

- Task: melanoma (label 1) versus all other lesion types (label 0).
- HAM10000 is split by lesion and ISIC 2020 by patient, so no lesion or patient is in two sets. Splits are stratified, with seed 42.
- Training uses HAM10000 only. The epoch and the decision threshold (Youden J) are chosen on the HAM10000 validation set.
- ISIC 2020 is split into a 30% design part (used only to choose between variants) and a 70% final part (scored once per model).
- Confidence intervals come from bootstrapping whole lesions or patients.

## Results (seeds 42, 1, 2; mean ± SD)

| Network | Variant | ISIC 2020 final ROC-AUC | ISIC 2020 final PR-AUC | HAM10000 test ROC-AUC |
|---|---|---|---|---|
| EfficientNet-B0 | base | 0.646 ± 0.025 | 0.051 ± 0.006 | 0.897 ± 0.005 |
| EfficientNet-B0 | both | 0.729 ± 0.015 | 0.092 ± 0.005 | 0.884 ± 0.003 |
| ResNet-50 | base | 0.708 ± 0.038 | 0.082 ± 0.008 | 0.896 ± 0.002 |
| ResNet-50 | both | 0.759 ± 0.037 | 0.105 ± 0.010 | 0.889 ± 0.005 |

The colour changes raised the external ROC-AUC in every seed for both networks. The mean gain was 0.083 for EfficientNet-B0 (range 0.042 to 0.117) and 0.052 for ResNet-50 (range 0.046 to 0.056), at a small internal cost of 0.013 and 0.007. On ISIC 2020, melanoma is 1.75% of the images, so chance-level PR-AUC is about 0.018. Sensitivity and specificity at validation-chosen thresholds vary a lot between seeds, so ROC-AUC and PR-AUC are the headline metrics. This is a research baseline, not a diagnostic tool.

Per-run scores are in `results/seed_summary.csv` (EfficientNet-B0) and `results/seed_summary_resnet50.csv` (ResNet-50, rounded to three decimals).

## Reproducibility

The reported runs were made with the scripts in this repository on a Google Colab T4 GPU, and they reproduced our original notebook results to the precision reported. Environment: Python 3.13, torch 2.11.0 (CUDA 13.0), torchvision 0.26.0, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1. Results are seeded but not guaranteed to be bit-for-bit identical on other hardware or library versions. `python scripts/make_splits.py --check` confirms that the stored splits regenerate identically.

## Limitations

- Three seeds, two networks and one patient split.
- The colour variant was chosen with EfficientNet-B0 only.
- Other causes of the dataset shift (camera, framing, lesion mix, label definitions) are not ruled out.
- The ISIC 2020 images are a pre-resized 224 x 224 copy.
- Patient overlap between HAM10000 and ISIC 2020 cannot be checked.
- The PH2 dataset has not been tested.

## Data and licences

The images are **not** included. Download them yourself and respect each licence.

- HAM10000 (CC BY-NC-SA 4.0): Tschandl, P., Rosendahl, C. and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161.
- ISIC 2020 (CC BY-NC 4.0): SIIM-ISIC 2020 Challenge Dataset, International Skin Imaging Collaboration, DOI 10.34970/2020-ds01.

The code is released under the MIT License.

## Citation

See `CITATION.cff`. Authors: D. Surekha and Priti Rishi, Department of Electronics and Communication Engineering, SRM Institute of Science and Technology, Vadapalani Campus, Chennai, India.
