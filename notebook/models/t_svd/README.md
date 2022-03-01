# 1 Results 

| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | t_svd | 10 | 0.2441 | nan | 11.9373 | 0.00075 | 0.008703 | 0.008194 | 0.00232 | nan | nan | nan | 0.956324 | 13.007329 | 0.528609 | 10.999713 |
| 1M | t_svd | 10 | 22.8542 | nan | 873.4765 | 0.00053 | 0.002037 | 0.001889 | 0.001787 | nan | nan | nan | 0.984564 | 14.124596 | 0.982771 | 11.913509 |
| all(8M) | t_svd | 10 | X | X | X | X | X | X | X | X | X | X | X | X | X | X |

## 2. Production 

This method is the one used in production. 

I have changed a little bit the actual product recommendation engine based on t_svd because :
- The actual product recommendation engine return the same recommendation bc the score are not sorted. We only tahe the first 10 results with a confidence score over 0.95. In many cases, we have the exact recommendation because a lot of products have a score over 0.95. 
- The already seen product were not removed

So I have :
- sorted the score, removed the already seen products and return the top_k items 
- the recommendation is only based on the most bought item by a customer ( in the actual engine, we used the first two ones most bought ) 
- The recommendation is based on all variant_id products in order to compare it correctly with the other models.


## 3 Truncated SVD model

The truncated SVD model algorithm is the real mathematical SVD performed on a matrix where the missing values are replaced by 0. 

In the definition of SVD, an original matrix A is can be factorized as a product A = UΣV* where U and V have orthonormal columns, and Σ is non-negative diagonal.

 Unlike regular SVDs, truncated SVD produces a factorization where the number of columns can be specified for a number of truncation. For example, given an n x n matrix, truncated SVD generates the matrices with the specified number of columns, whereas SVD outputs n columns of matrices.

By calculating UΣ, we will get our item features. From this point, we can compute the correlation matrix of UΣ to obtain the similarity between the items. The correlation matrix uses Pearson's Product-Moment Correlation.

Then, we propose similar items with the highest correlation value.


