# AbFold

Official repository for [AbFold](): an AlphaFold Based Transfer Learning Model for Accurate Antibody Structure Prediction.

## Installation

All Python dependencies are specified in `environment.yml`. 
We recommend using conda environment, to install dependencies, run:
```bash
$ conda env create -f environment.yml
```

To activate the virtual environment, run:

```bash
$ conda activate abfold
```

## Data

Data used as benckmark and pre-trained model weights are available at [Google Drive](https://drive.google.com/drive/folders/1_-IYqj0bQWra_7erb8mwLnrJ1Vm_l1G4?usp=sharing).

## Training

Before training, download data to `data` folder as following structure:

```
data/
└── abfold/
    └── train/
        ├── unsupervised_repr       # precomputed representations
        ├── unsupervised_fasta      # fasta files
        ├── supervised_label        # label files
        ├── embed                   # embedding files generated by point_mae
        └── unsupervised_embed      # embedding files generated by point_mae for non-structure data
```

To train AbFold model, run:

```bash
$ python train_ema.py
```

## Inference

Before inference, download data to `data` folder as following structure:

```
data/
└── abfold/
    └── test/
        ├── repr                    # precomputed representations
        ├── fasta                   # fasta files
        ├── embed                   # embedding files generated by point_mae
        └── pred_structures         # in which to output predicted structures
```

To predict Antibody structure with AbFold model, run:

```bash
$ python predict.py
```

## Citing this work

```bibtex
@article{abfold,
    title = {AbFold -- an AlphaFold Based Transfer Learning Model for Accurate Antibody Structure Prediction},
    author = {Peng, Chao and Wang, Zelong and Zhao, Peize and Ge, Weifeng and Huang, Charles},
    journal = {bioRxiv},
    year= {2023}
}
```