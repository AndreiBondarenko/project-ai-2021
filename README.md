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

