# Project AI 2021: Matrix Factorization on Goodreads Dataset

This is the source code repository for the Matrix Factorization (MF) implementation by Andrei Bondarenko and Geert Goemaere for the Project AI course (2021). We performed experiments on the Goodreads datset provided to us.

A small overview of the files in this repository:

- data: contains intermediate data files
- recommenders: contains the implementations of both the baseline popularity recommender, as well as the MF based recommender.
- sources.md: contains (some) of the sources we referenced
- utils.py: utility functions for notebook experiments
- preprocessing.ipynb: preprocessing of the Goodreads dataset
- baseline.ipynb: quantitative evaluation of the baseline popularity recommender
- mf_grid_search_\*.ipynb: grid search experiment for various MF recommender hyper-parameters
- mf_qual_eval.ipynb: qualitative evaluation experiment for MF recommender
- mf_quant_eval.ipynb: quantitative evaluation experiment for MF recommender

## Agreed pre-processing, training and evaluation in optimization phase for the goodreads teams

### preprocessing

- min. sup. users = 5
- min sup items = 1 (should be the case in dataset already, check to be sure)
- keep all the ratings and scores

### training

- For validation we consider doing 5 random 80%/20% train-test splits.
- For each train-test pair, we first perform hyperparameter optimisation on the train part (via cross-validation) and then evaluate on the test part.
- This gives us 5x(recall@10, ndcg@10) for which we compute the mean and stdev.
- --> for the 5 random splits we should all use the same seeds

### evaluation

- recall k = 5, 10
- NDCG = 5, 10

