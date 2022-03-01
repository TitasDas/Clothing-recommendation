# 1 Results 

| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | svd | 10 | 4.5891 | 0.2028 | 80.2 | 0.005101 | 0.044841 | 0.033194 | 0.009239 | 5.302307 | 1.317344 | 0.032495 | 0.932874 | 11.655815 | 0.008788 | 4.442072 |
| 1M | svd | 10 | 39.7332 | 10.2 | 893.3 | 0.003614 | 0.010738 | 0.006898 | 0.006461 | 2.466905 | 0.816845 | 0.107346 | 0.979418 | 12.466466 | 0.010248 | 4.600128 |
| all(8M) | svd | 10 | 2205.5820 | 20.6284 | 5173.3098 | 0.007453 | 0.010208 | 0.002405 | 0.014684 | 2.107780 | 0.736401 | -0.058925 | 0.988950 | 11.154896 | 0.031075 | 5.524555 |

```python
# Model definition 

# for 100k and 1M data 
svd = surprise.SVD(random_state=0, n_factors=3, n_epochs=700,biased=False,lr_all=0.0001,verbose=False)

# for all data 
svd = surprise.SVD(random_state=0, n_factors=3, n_epochs=200,biased=False,lr_all=0.001,verbose=False)

```

## 2 Matrix factorization algorithm

The SVD model algorithm is very similar to the ALS algorithm. The two differences between the two approaches are:

- SVD additionally models the user and item biases (also called baselines in the litterature) from users and items.
- The optimization technique in ALS is Alternating Least Squares (hence the name), while SVD uses stochastic gradient descent.

### 2.1 The SVD model

The method usually referred to as "SVD" that is used in the context of recommendations is not strictly speaking the mathematical Singular Value Decomposition of a matrix but rather an approximate way to compute the low-rank approximation of the matrix by minimizing the squared error loss. A more accurate, albeit more generic, way to call this would be Matrix Factorization.  It is important to note that the "true SVD" approach had been indeed applied to the same task years before, with not so much practical success. Billsus & Panzani, for example, described already in 1998 how to use this approach (see Page on aaai.org).

In ALS, the ratings are modeled as follows:
```math
\hat r_{u,i} = q_{i}^{T}p_{u}
```
SVD introduces two new scalar variables: the user biases $`b_u`$ and the item biases $`b_i`$. The user biases are supposed to capture the tendency of some users to rate items higher (or lower) than the average. The same goes for items: some items are usually rated higher than some others. The model is SVD is then as follows:

```math
\hat r_{u,i} = \mu + b_u + b_i + q_{i}^{T}p_{u}
```

Where $`\mu`$ is the global average of all the ratings in the dataset. The regularised optimization problem naturally becomes:

```math
\sum(r_{u,i} - (\mu + b_u + b_i + q_{i}^{T}p_{u}))^2 +     \lambda(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2)
```

where $`\lambda`$ is a the regularization parameter.


### 2.2 Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a very common algorithm for optimization where the parameters (here the biases and the factor vectors) are iteratively incremented with the negative gradients w.r.t the optimization function. The algorithm essentially performs the following steps for a given number of iterations:


```math
b_u \leftarrow b_u + \gamma (e_{ui} - \lambda b_u)
```
```math
b_i \leftarrow b_i + \gamma (e_{ui} - \lambda b_i)
```  
```math
p_u \leftarrow p_u + \gamma (e_{ui} \cdot q_i - \lambda p_u)
```
```math
q_i \leftarrow q_i + \gamma (e_{ui} \cdot p_u - \lambda q_i)
```

where $`\gamma`$ is the learning rate and $`e_{ui} =  r_{ui} - \hat r_{u,i} = r_{u,i} - (\mu + b_u + b_i + q_{i}^{T}p_{u})`$ is the error made by the model for the pair $`(u, i)`$.