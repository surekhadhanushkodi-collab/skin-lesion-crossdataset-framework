# Skin Lesion Cross-Dataset Framework

An open-source, reproducible pipeline for classifying melanoma versus other skin lesions in dermoscopic images, with leakage-safe splits and **cross-dataset validation**: a model is trained on HAM10000 and tested on an independent dataset (ISIC 2020).

**Status:** work in progress. A DOI will be added here after the first release.

## Why cross-dataset validation

A model that scores well on its own dataset can fail on new data. In our baseline, ROC-AUC fell from 0.90 on the HAM10000 test set to about 0.6 on ISIC 2020. This repository makes that check part of the pipeline, and tests simple colour-based fixes.

## What is here

| Path | Content |
|---|---|
| `skinfw/` | The framework: splits, colour normalisation, data, model, training, metrics |
| `scripts/` | Command-line tools: download, splits, train, evaluate, summarise |
| `configs/default.yaml` | All settings (seed, epochs, learning rate, variant, paths) |
| `splits/` | Reference split files (lesion-level HAM10000, patient-level ISIC 2020) |
| `tests/` | Tests that need no GPU and no data |

## Variants

- `base`: mild augmentation
- `sog`: Shades-of-Gray colour normalisation
- `coloraug`: strong colour jitter and random grayscale
- `both`: `sog` and strong colour augmentation

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
python scripts/train.py --variant both --seed 42
python scripts/evaluate.py --run runs/both_seed42 --set isic_final --ref runs/base_seed42
python scripts/summarize_seeds.py
python -m pytest -q                             # tests, no GPU or data needed
```

A run saves its model, threshold, predictions and scores to `runs/<variant>_seed<seed>/`.

## Method in brief

- Task: melanoma (label 1) versus all other lesion types (label 0).
- HAM10000 is split by lesion and ISIC 2020 by patient, so no lesion or patient is in two sets. Splits are stratified, with seed 42.
- Training uses HAM10000 only. The epoch and the decision threshold (Youden J) are chosen on the HAM10000 validation set.
- ISIC 2020 is split into a 30% design part (used only to choose between variants) and a 70% final part (scored once).
- Confidence intervals come from bootstrapping whole lesions or patients.

## Results so far (seeds 42, 1, 2)

| | ISIC 2020 final ROC-AUC | HAM10000 test ROC-AUC |
|---|---|---|
| base | 0.646 ± 0.025 | 0.897 ± 0.005 |
| both | 0.729 ± 0.015 | 0.884 ± 0.003 |

The colour changes raise external ROC-AUC by about 0.08 on average (range 0.04 to 0.12, positive in 3 of 3 seeds) at a small internal cost of about 0.013. Sensitivity and specificity at validation-chosen thresholds vary a lot between seeds, so ROC-AUC and PR-AUC are the headline metrics. This is a research baseline, not a diagnostic tool.

## Limitations

- Three seeds and one patient split.
- Other causes of the dataset shift (camera, framing, lesion mix) are not ruled out.
- The ISIC 2020 images are a pre-resized 224x224 copy.
- Patient overlap between HAM10000 and ISIC 2020 cannot be checked.
- The PH2 dataset has not been tested yet.
- Runs are seeded but not guaranteed to be bit-for-bit identical across hardware.

## Data and licences

The images are **not** included. Download them yourself and respect each licence.

- HAM10000 (CC BY-NC-SA 4.0): Tschandl, P., Rosendahl, C. and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161.
- ISIC 2020 (CC BY-NC 4.0): SIIM-ISIC 2020 Challenge Dataset, International Skin Imaging Collaboration, DOI 10.34970/2020-ds01.

The code is released under the MIT License.

## Citation

See `CITATION.cff`. A DOI will be added after the first release.
