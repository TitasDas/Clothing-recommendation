# Models

The table below lists the recommender algorithms that have been tested on our data. Notebooks are linked under the Environment column when different implementations are available.

| Algorithm | Environment | Type | Description |
| --- | --- | --- | --- |
| Alternating Least Squares (ALS) | [PySpark/Python CPU/GPU](als) | Collaborative Filtering | Matrix factorization algorithm for explicit or implicit feedback in large datasets|
| Bayesian Personalized Ranking (BPR) | [Python CPU](bpr) | Collaborative Filtering | Matrix factorization algorithm for predicting item ranking with implicit feedback |
| Simple Algorithm for Recommendation (SAR)| [Python CPU](sar) | Collaborative Filtering | Similarity-based algorithm for implicit feedback dataset |
| LightFM/Hybrid Matrix Factorization | [Python CPU](lfm) | Hybrid | Hybrid matrix factorization algorithm for both implicit and explicit feedbacks |
| Singular Value Decomposition (SVD) | [Python CPU](svd) | Collaborative Filtering | Matrix factorization algorithm for predicting explicit rating feedback in datasets that are not very large |
| Truncated SVD | [Python CPU](t_svd) | Collaborative Filtering | Matrix factorization algorithm by using the real mathematical SVD |
| Random | [Python CPU](random) | Classic | Random suggestions |
| Popular | [Python CPU](popular) | Classic | Most popular items suggestions |

# Evaluation protocol 

Each algorithm should use the same preprocessed data when we are extracting 100k or 1M data out of the 8M data that we have in total. The goal is to have the same (customer_id,item_id) pair. You can find how the processing was done [here](../data/get_data.ipynb).

Moreover, we need to split the data into train and test set the same way. We have decided to use chronological split which is deterministic by nature (timestamp-dependent) because our goal is predict what the customer will buy next. By doing that, timestamps of train data are all precedent to those in test data.

# Comparison of the different models

I have sorted the models on the column Precision@k in descending order. 
For several models, I wasn't able to get the metrics because of memory issues.

## 100k data 
| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | sar | 10 | 1.869315 | nan | 0.412482 | 0.015832 | 0.120409 | 0.11125 | 0.031310 | nan | nan | nan | 0.760300 | 11.058183 | 0.274308 | 8.97861 |
| 100k | bpr | 10 | 317.5505 | nan | 14.4185 | 0.008045 | 0.069304 | 0.064444 | 0.018639 | nan | nan | nan | 0.809253 | 9.924094 | 0.188482 | 8.515699 |
| 100k | lfm | 10 | 302.0318 | nan | 5604.3420 | 0.00733 | 0.062168 | 0.059444 | 0.017409 | nan | nan | nan | 0.836161 | 10.316668 | 0.259536 | 9.383213 |
| 100k | als | 10 | 3.0065 | nan | 4.2962 | 0.006227 | 0.055221 | 0.051667 | 0.015082 | nan | nan | nan | 0.854077 | 10.217964 | 0.205497 | 9.261189 |
| 100k | popular | 10 | nan | nan | 0.2397 | 0.004115 | 0.045793 | 0.049583 | 0.013647 | nan | nan | nan | 0.725257 | 8.942318 | 0.004488 | 3.98002 |
| 100k | svd | 10 | 4.5891 | 0.2028 | 80.2 | 0.005101 | 0.044841 | 0.033194 | 0.009239 | 5.302307 | 1.317344 | 0.032495 | 0.932874 | 11.655815 | 0.008788 | 4.442072 |
| 100k | t_svd | 10 | 0.2441 | nan | 11.9373 | 0.00075 | 0.008703 | 0.008194 | 0.00232 | nan | nan | nan | 0.956324 | 13.007329 | 0.528609 | 10.999713 |
| 100k | random | 10 | nan | nan | 2.7093 | 0.000722 | 0.007079 | 0.00625 | 0.001933 | nan | nan | nan | 0.984977 | 13.446246 | 0.741586 | 11.767055 |

## 1M data 

| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1M | sar | 10 | 4.480668 | nan | 9.005530 | 0.027378 | 0.074688 | 0.06103 | 0.059482 | nan | nan | nan | 0.904983 | 11.574736 | 0.774543 | 10.55634 |
| 1M | bpr | 10 | 1593.6376 | nan | 2583.3877 | 0.013866 | 0.041259 | 0.033866 | 0.033263 | nan | nan | nan | 0.910324 | 10.176613 | 0.437249 | 9.188603 |
| 1M | als | 10 | 8.6742 | nan | 135.8204 | 0.012856 | 0.038076 | 0.031262 | 0.030650 | nan | nan | nan | 0.920905 | 10.144708 | 0.234517 | 9.150357 |
| 1M | popular | 10 | nan | nan | 2.3132 | 0.003371 | 0.016208 | 0.019308 | 0.018529 | nan | nan | nan | 0.877848 | 8.577696 | 0.003416 | 3.61935 |
| 1M | svd | 10 | 39.7332 | 10.2 | 893.3 | 0.003614 | 0.010738 | 0.006898 | 0.006461 | 2.466905 | 0.816845 | 0.107346 | 0.979418 | 12.466466 | 0.010248 | 4.600128 |
| 1M | t_svd | 10 | 22.8542 | nan | 873.4765 | 0.00053 | 0.002037 | 0.001889 | 0.001787 | nan | nan | nan | 0.984564 | 14.124596 | 0.982771 | 11.913509 |
| 1M | random | 10 | nan | nan | 24.3165 | 0.000413 | 0.001584 | 0.001469 | 0.001415 | nan | nan | nan | 0.995884 | 14.426557 | 1.000000 | 12.695711 |
| 1M | lfm | 10 | X | X | X | X | X | X | X | X | X | X | X | X | X | X |
## all data ( 8M ) 

| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all(8M) | sar | 10 | 18.09361 | nan | 211.60146 | 0.100136 | 0.138245 | 0.034234 | 0.21693 | nan | nan | nan | 0.957637 | 11.336318 | 0.957958 | 11.040913 |
| all(8M) | als | 10 | 326.4773 | nan | 8666.2828 | 0.070028 | 0.098556 | 0.024141 | 0.160735 | nan | nan | nan | 0.962114 | 10.042972 | 0.158767 | 9.149739 |
| all(8M) | popular | 10 | nan | nan | 322.5469 | 0.009730 | 0.020413 | 0.007956 | 0.047232 | nan | nan | nan | 0.968766 | 7.961962 | 0.002481 | 3.409857 |
| all(8M) | svd | 10 | 2205.5820 | 20.6284 | 5173.3098 | 0.007453 | 0.010208 | 0.002405 | 0.014684 | 2.107780 | 0.736401 | -0.058925 | 0.988950 | 11.154896 | 0.031075 | 5.524555 |
| all(8M) | random | 10 | nan | nan | 231.232 | 0.000381 | 0.000715 | 0.000274 | 0.001312 | nan | nan | nan | 0.998323 | 15.23186 | 1.0 | 12.902564 |
| all(8M) | t_svd | 10 | X | X | X | X | X | X | X | X | X | X | X | X | X | X |
| all(8M) | bpr | 10 | X | X | X | X | X | X | X | X | X | X | X | X | X | X |
| all(8M) | lfm | 10 | X | X | X | X | X | X | X | X | X | X | X | X | X | X |