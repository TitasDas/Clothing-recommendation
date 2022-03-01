## Results 

| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | popular | 10 | nan | nan | 0.2397 | 0.004115 | 0.045793 | 0.049583 | 0.013647 | nan | nan | nan | 0.725257 | 8.942318 | 0.004488 | 3.98002 |
| 1M | popular | 10 | nan | nan | 2.3132 | 0.003371 | 0.016208 | 0.019308 | 0.018529 | nan | nan | nan | 0.877848 | 8.577696 | 0.003416 | 3.61935 |
| all(8M) | popular | 10 | nan | nan | 322.5469 | 0.009730 | 0.020413 | 0.007956 | 0.047232 | nan | nan | nan | 0.968766 | 7.961962 | 0.002481 | 3.409857 |

## Popular algorithm

The popular algorithm recommend the 10 most popular items to users. It has been implemented in order to have a clear baseline.