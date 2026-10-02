# Skin Lesion Cross-Dataset Framework

An open-source, reproducible pipeline for classifying melanoma versus other skin lesions in dermoscopic images, tested across independent public datasets.

**Status:** work in progress.

## What is here so far

- `ham10000_train.csv`, `ham10000_val.csv`, `ham10000_test.csv`: lesion-level splits of HAM10000.

## About the splits

- Task: melanoma (label 1) versus all other lesion types (label 0).
- Images from the same lesion are never placed in different splits, which prevents leakage.
- Splits are stratified by label with a fixed random seed (42), so they can be reproduced exactly.
- Sizes: 6,969 train, 1,517 validation and 1,529 test images.

## Data

The images are not included in this repository. Download HAM10000 from Kaggle (`kmader/skin-cancer-mnist-ham10000`). The dataset is released under CC BY-NC-SA 4.0, so please check its terms before reuse.

Dataset reference: Tschandl, P., Rosendahl, C. and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161.

## License

Code in this repository is released under the MIT License.

## Citation

A DOI will be added here after the first release.
