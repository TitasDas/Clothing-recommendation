## Results 

| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | als | 10 | 3.0065 | nan | 4.2962 | 0.006227 | 0.055221 | 0.051667 | 0.015082 | nan | nan | nan | 0.854077 | 10.217964 | 0.205497 | 9.261189 |
| 1M | als | 10 | 8.6742 | nan | 135.8204 | 0.012856 | 0.038076 | 0.031262 | 0.030650 | nan | nan | nan | 0.920905 | 10.144708 | 0.234517 | 9.150357 |
| all(8M) | als | 10 | 326.4773 | nan | 8666.2828 | 0.070028 | 0.098556 | 0.024141 | 0.160735 | nan | nan | nan | 0.962114 | 10.042972 | 0.158767 | 9.149739 |

with factors = 50, regularization = 0.01, iterations=50 and alpha = 10

## ALS algorithm

## 1 Implementation

We have used the implementation by implicit. I tried using pyspark implementation but it was really slow.

If you want to run the pyspark notebook, you will need to install more dependencies. You can find them [here](https://github.com/microsoft/recommenders/blob/main/SETUP.md#dependencies-setup)

If you want to run implicit on MacOS, you need an  OpenMP compiler, which can be installed with homebrew: ```brew install gcc```. Running on Windows requires Python 3.5+. You can find more information [here](https://github.com/benfred/implicit).

## 2 Matrix factorization algorithm

### 2.1 Matrix factorization for collaborative filtering problem

Matrix factorization is a common technique used in recommendation tasks. Basically, a matrix factorization algorithm tries to find latent factors that represent intrinsic user and item attributes in a lower dimension. That is,

```math
\hat r_{u,i} = q_{i}^{T}p_{u}
```

where $`\hat r_{u,i}`$ is the predicted ratings for user $`u`$ and item $`i`$, and $`q_{i}^{T}`$ and $`p_{u}`$ are latent factors for item and user, respectively. The challenge to the matrix factorization problem is to find $`q_{i}^{T}`$ and $`p_{u}`$. This is achieved by methods such as matrix decomposition. A learning approach is therefore developed to converge the decomposition results close to the observed ratings as much as possible. Furthermore, to avoid overfitting issue, the learning process is regularized. For example, a basic form of such matrix factorization algorithm is represented as below.

```math
\min\sum(r_{u,i} - q_{i}^{T}p_{u})^2 + \lambda(||q_{i}||^2 + ||p_{u}||^2)
```

where $`\lambda`$ is a the regularization parameter. 

In case explict ratings are not available, implicit ratings which are usually derived from users' historical interactions with the items (e.g., clicks, views, purchases, etc.). To account for such implicit ratings, the original matrix factorization algorithm can be formulated as 

```math
\min\sum c_{u,i}(p_{u,i} - q_{i}^{T}p_{u})^2 + \lambda(||q_{i}||^2 + ||p_{u}||^2)
```

where $`c_{u,i}=1+\alpha r_{u,i}`$ and $`p_{u,i}=1`$ if $`r_{u,i}>0`$ and $`p_{u,i}=0`$ if $`r_{u,i}=0`$. $`r_{u,i}`$ is a numerical representation of users' preferences (e.g., number of clicks, etc.). 

### 2.2 Alternating Least Square (ALS)

Owing to the term of $`q_{i}^{T}p_{u}`$ the loss function is non-convex. Gradient descent method can be applied but this will incur expensive computations. An Alternating Least Square (ALS) algorithm was therefore developed to overcome this issue. 

The basic idea of ALS is to learn one of $`q`$ and $`p`$ at a time for optimization while keeping the other as constant. This makes the objective at each iteration convex and solvable. The alternating between $`q`$ and $`p`$ stops when there is convergence to the optimal. It is worth noting that this iterative computation can be parallelized and/or distributed, which makes the algorithm desirable for use cases where the dataset is large and thus the user-item rating matrix is super sparse (as is typical in recommendation scenarios). A comprehensive discussion of ALS and its distributed computation can be found [here](http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf).