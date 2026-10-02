# MS-HGIA

This repository is the official PyTorch implementation of  
**"Multi-scale Hypergraph Fusion with Contrastive-guided Distillation for Whole Slide Image Classification"**.

MS-HGIA is designed for multi-scale whole slide image (WSI) classification. It combines intra-scale hypergraph modeling, cross-magnification interaction, and semantic alignment to improve the consistency of multi-scale representations.

## Installation

The code was developed and tested with Python 3.10 and PyTorch 2.1.2.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Data

Camelyon16
The Camelyon16 dataset can be obtained from the official Grand Challenge website:
https://camelyon16.grand-challenge.org/Data/
Please follow the official instructions and data usage policy provided by the challenge website.

TCGA Lung
The TCGA Lung dataset can be obtained from the official Genomic Data Commons (GDC) portal:
https://portal.gdc.cancer.gov/
The experiments in this work use whole slide images from the LUAD and LUSC cohorts.

BreastC-MS
BreastC-MS is an in-house breast cancer dataset used in this study.
Due to privacy and institutional restrictions, the original WSIs and associated clinical data cannot be publicly released at this time.

## Training and Testing

The main training entry is:
main.py

The program supports different datasets through the --dataset argument:
cam      : Camelyon16
lung     : TCGA Lung

The experiments are conducted using five-fold cross-validation.
```bash
for f in 0 1 2 3 4; do
  python main.py \
    --datasetpath path \
    --dataset cam \
    --fold $f
done
```

