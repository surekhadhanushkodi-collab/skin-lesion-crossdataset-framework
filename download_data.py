"""Download the public datasets with the Kaggle CLI.

Needs a Kaggle API token in the KAGGLE_API_TOKEN environment variable.
The images are NOT part of this repository. Check each dataset's licence first.
"""
import argparse
import os
import subprocess
import sys

DATASETS = {
    "ham10000": ("kmader/skin-cancer-mnist-ham10000", "data/ham10000",
                 "HAM10000 (CC BY-NC-SA 4.0). Cite Tschandl et al., Scientific Data 2018."),
    "isic2020": ("nischaydnk/isic-2020-jpg-224x224-resized", "data/isic2020",
                 "ISIC 2020 resized copy. Original licence CC BY-NC 4.0; cite the "
                 "SIIM-ISIC 2020 Challenge Dataset, DOI 10.34970/2020-ds01."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = ap.parse_args()
    if not os.environ.get("KAGGLE_API_TOKEN") and not os.path.exists(
            os.path.expanduser("~/.kaggle/kaggle.json")):
        sys.exit("Set KAGGLE_API_TOKEN (or provide ~/.kaggle/kaggle.json) first.")
    for key in args.which:
        ref, dest, note = DATASETS[key]
        print(f"\n{key}: {note}")
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", ref, "-p", dest, "--unzip"],
                       check=True)


if __name__ == "__main__":
    main()
