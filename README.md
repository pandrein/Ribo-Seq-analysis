This repository contains the codes for the paper **"Statistical and Machine Learning Methods for the Analysis of Reproducible Ribo–Seq Profiles"**.

*Authors*: 
Giorgia Giacomini, Caterina Graziani, Veronica Lachi, Pietro Bongini, Niccolò Pancino, Monica Bianchini, Davide Chiarugi, Angelo Valleriani, Paolo Andreini

The analysis consists of three main parts: ORF-specific Ribo-seq analysis, statistical analysis on the nucleotide composition and data validation with neural network models.

## ORF-specific Ribo-seq analysis:

The analysis of the ORF-specific Ribo-seq profiles is articulated as follows:

- Set up of the coverage matrix and elaboration of digitalised profiles

- Comparison of the digitalised profiles

- Assessment of the similarity scores

- Identification of the "significantly reproducible Ribo-seq profiles"

core source code related to the identification of reproducible Ribo-seq profiles through the systematic comparison of different Ribo-seq datasets.

## Statistical analysis
[Statistical analysis](https://github.com/pandrein/Ribo-Seq-analysis/tree/main/statistical_analysis) to detect the differences in the nucleotide composition between sub--sequences

## Machine learning analysis:
The generated sub-sequences are analyzed through different machine learning models:

- 1D-CNN
- [MLP applied on nucleotide frequency](https://github.com/pandrein/Ribo-Seq-analysis/tree/main/mlp_model)
- Ensamble of 7 1D-CNN


