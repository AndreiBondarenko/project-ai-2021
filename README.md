# project-ai-2021

## Evaluatie 16/11/2021

Things to do:

- Kennis abstract sectie van papers in references van de Matrix\_Factorization paper.
- Netflix prize:
  - Wat is deze competitie?
  - Wie heeft de competitie gewonnen en met welke methode?
  - Netflix rating niet zo effectief in praktijk? Wat zegt literatuur, alternatieven?
- SGD:
  - Wat is het?
  - Verschillende alternatieven en/of optimizers
- Verschil SGD en ALS uitleggen
- Details paper goed kennen
- Colaborative filtering: verschil neighborhood methods versus latent factor models
- Paper: Kennis van de context, voorgaand en volgend werk

recall@K:
- per user, predict score for all items
- per user, sort descending items based on score and keep K items with highest score
- recall@K per user: (number of relevant items of top-K list found in future list) / K
- average recall@K = mean(recall@K of all users)

NDCG@k:
- recall@K but relevant item weighted based on position in top-K list
