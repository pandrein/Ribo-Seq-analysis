This repository contains the codes (Python) used to compare the ORF-specific Ribo-seq profiles coming from different datasets and to assess the reproducibility of them.

Our analysis of the ORF-specific Ribo-seq profiles consist of two phases:

## Upstream phase:
The upstream phase allows us to compute the Ribo-seq profiles starting from the raw Ribo-seq data

## Downstream phase:
The downstream phase is the core of our method and is articulated as follows:

- Set up of the coverage matrix and elaboration of digitalised profiles

- Comparison of the digitalised profiles

- Assessment of the similarity scores

- Identification of the "significantly reproducible Ribo-seq profiles"

## Machine learning analysis:
The generated sub-sequences are analyzed through different machine learning models:

- 1D-CNN
- MLP applied on nucleotide frequency
- Ensamble of 7 1D-CNN

## Statistical analysis
To show the differences in the nucleotide composition between sub--sequences

