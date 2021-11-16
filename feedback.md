# Intermediate evaluation

## Data removal

- Removing both users < u\_minsup and items < i\_minsup is wrong approach. Recommedation is about niches. 
- Better to remove most popular items
- Do tests with and without removing users/items and compare.

## Recall@K

- Recall@20 or higher is not good because the higher K the better the result
- Use recall@5 or recall@10

## Evaluation set for grid search

- Need 3 datasets: training, validation (hyperparameter optimization), test

## Quantitative results
- Does it make sens to use as input a combination of items (e.g. frequent pairs)
- Add plots for R@10, NDCG@10, MSE in function of training iterations

## Qualitative results

- Don\'t draw conclusions from this
- Only if it is really bad

## New paper from authors of matrix factorization
- Published October 26th 2021
- Definitely interesting for the final evaluation
- Revisiting the Performance of iALS on Item Recommendation Benchmarks?

