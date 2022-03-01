## Results 

| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | random | 10 | nan | nan | 2.7093 | 0.000722 | 0.007079 | 0.00625 | 0.001933 | nan | nan | nan | 0.984977 | 13.446246 | 0.741586 | 11.767055 |
| 1M | random | 10 | nan | nan | 24.3165 | 0.000413 | 0.001584 | 0.001469 | 0.001415 | nan | nan | nan | 0.995884 | 14.426557 | 1.000000 | 12.695711 |
| all(8M) | random | 10 | nan | nan | 231.232 | 0.000381 | 0.000715 | 0.000274 | 0.001312 | nan | nan | nan | 0.998323 | 15.23186 | 1.0 | 12.902564 |

## Random algorithm

The random algorithm recommend random items to users. It has been implemented in order to have a clear baseline.