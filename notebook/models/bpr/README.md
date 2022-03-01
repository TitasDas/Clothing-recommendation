# 1 Results
| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | bpr | 10 | 317.5505 | nan | 14.4185 | 0.008045 | 0.069304 | 0.064444 | 0.018639 | nan | nan | nan | 0.809253 | 9.924094 | 0.188482 | 8.515699 |
| 1M | bpr | 10 | 1593.6376 | nan | 2583.3877 | 0.013866 | 0.041259 | 0.033866 | 0.033263 | nan | nan | nan | 0.910324 | 10.176613 | 0.437249 | 9.188603 |
| all(8M) | bpr | 10 | X | X | X | X | X | X | X | X | X | X | X | X | X | X |

With the parameters : 
```python
bpr = cornac.models.BPR(
    k=500,
    max_iter=500,
    learning_rate=0.01,
    lambda_reg=0.01,
    verbose=True,
    seed=42
)
```


## 2 BPR Algorithm

### 1.1 Personalized Ranking from Implicit Feedback

The task of personalized ranking aims at providing each user a ranked list of items (recommendations).  This is very common in scenarios where recommender systems are based on implicit user behavior (e.g. purchases, clicks).  The available observations are only positive feedback where the non-observed ones are a mixture of real negative feedback and missing values.

One usual approach for item recommendation is directly predicting a preference score $`\hat{x}_{u,i}`$ given to item $`i`$ by user $`u`$.  BPR uses a different approach by using item pairs $`(i, j)`$ and optimizing for the correct ranking given preference of user $`u`$, thus, there are notions of *positive* and *negative* items.  The training data $`D_S : U \times I \times I`$ is defined as:

```math
D_S = \{(u, i, j) \mid i \in I^{+}_{u} \wedge j \in I \setminus I^{+}_{u}\}
```

where user $`u`$ is assumed to prefer $`i`$ over $`j`$ (i.e. $`i`$ is a *positive item* and $`j`$ is a *negative item*).


### 1.2 Objective Function

From the Bayesian perspective, BPR maximizes the posterior probability over the model parameters $`\Theta`$ by optimizing the likelihood function $`p(i >_{u} j | \Theta)`$ and the prior probability $`p(\Theta)`$.

```math
p(\Theta \mid >_{u}) \propto p(i >_{u} j \mid \Theta) \times p(\Theta)
```

The joint probability of the likelihood over all users $`u \in U`$ can be simplified to:

```math
\prod_{u \in U} p(>_{u} \mid \Theta) = \prod_{(u, i, j) \in D_S} p(i >_{u} j \mid \Theta)
```

The individual probability that a user $`u`$ prefers item $`i`$ to item $`j`$ can be defined as:

```math
p(i >_{u} j \mid \Theta) = \sigma (\hat{x}_{uij}(\Theta))
```

where $`\sigma`$ is the logistic sigmoid:

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

The preference scoring function $`\hat{x}_{uij}(\Theta)`$ could be an arbitrary real-valued function of the model parameter $`\Theta`$.  Thus, it makes BPR a general framework for modeling the relationship between triplets $`(u, i, j)`$ where different model classes like matrix factorization could be used for estimating $`\hat{x}_{uij}(\Theta)`$.

For the prior, one of the common pratices is to choose $`p(\Theta)`$ following a normal distribution, which results in a nice form of L2 regularization in the final log-form of the objective function.

```math
p(\Theta) \sim N(0, \Sigma_{\Theta})
```

To reduce the complexity of the model, all parameters $`\Theta`$ are assumed to be independent and having the same variance, which gives a simpler form of the co-variance matrix $`\Sigma_{\Theta} = \lambda_{\Theta}I`$.  Thus, there are less number of hyperparameters to be determined.

The final objective of the maximum posterior estimator:

```math
J = \sum_{(u, i, j) \in D_S} \text{ln } \sigma(\hat{x}_{uij}) - \lambda_{\Theta} ||\Theta||^2 
```

where $`\lambda_\Theta`$ are the model specific regularization paramerters.


### 1.3 Learning with Matrix Factorization

#### Stochastic Gradient Descent

As the defined objective function is differentible, gradient descent based method for optimization is naturally adopted.  The gradient of the objective $`J`$ with respect to the model parameters:

```math
\begin{align}
\frac{\partial J}{\partial \Theta} & = \sum_{(u, i, j) \in D_S} \frac{\partial}{\partial \Theta} \text{ln} \ \sigma(\hat{x}_{uij}) - \lambda_{\Theta} \frac{\partial}{\partial \Theta} ||\Theta||^2 \\
& \propto \sum_{(u, i, j) \in D_S} \frac{-e^{-\hat{x}_{uij}}}{1 + e^{-\hat{x}_{uij}}} \cdot  \frac{\partial}{\partial \Theta} \hat{x}_{uij} - \lambda_{\Theta} \Theta
\end{align}
```

Due to slow convergence of full gradient descent, we prefer using stochastic gradient descent to optimize the BPR model.  For each triplet $`(u, i, j) \in D_S`$, the update rule for the parameters:

```math
\Theta \leftarrow \Theta + \alpha \Big( \frac{e^{-\hat{x}_{uij}}}{1 + e^{-\hat{x}_{uij}}} \cdot \frac{\partial}{\partial \Theta} \hat{x}_{uij} + \lambda_\Theta \Theta \Big) 
```

#### Matrix Factorization for Preference Approximation

As mentioned earlier, the preference scoring function $`\hat{x}_{uij}(\Theta)`$ could be approximated by any real-valued function.  First, the estimator $`\hat{x}_{uij}`$ is decomposed into:

```math
\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}
```

The problem of estimating $`\hat{x}_{ui}`$ is a standard collaborative filtering formulation, where matrix factorization approach has shown to be very effective.  The prediction formula can written as dot product between user feature vector $`w_u`$ and item feature vector $`h_i`$:

```math
\hat{x}_{ui} = \langle w_u , h_i \rangle = \sum_{f=1}^{k} w_{uf} \cdot h_{if}
```

The  derivatives of matrix factorization with respect to the model parameters are:

```math
\frac{\partial}{\partial \theta} \hat{x}_{uij} = 
\begin{cases}
    (h_{if} - h_{jf})  & \text{if } \theta = w_{uf} \\
    w_{uf}             & \text{if } \theta = h_{if} \\
    -w_{uf}            & \text{if } \theta = h_{jf} \\
    0                  & \text{else}
\end{cases}
```

In theory, any kernel can be used to estimate $`\hat{x}_{ui}`$ besides the dot product $` \langle \cdot , \cdot \rangle `$.  For example, k-Nearest-Neighbor (kNN) has also been shown to achieve good performance.

#### Analogies to AUC optimization

By optimizing the objective function of BPR model, we effectively maximizing [AUC](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) measurement.  To keep the notebook focused, please refer to the [paper](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) for details of the analysis (Section 4.1.1).