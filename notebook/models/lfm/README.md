# Results
| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | lfm | 10 | 302.0318 | nan | 5604.3420 | 0.00733 | 0.062168 | 0.059444 | 0.017409 | nan | nan | nan | 0.836161 | 10.316668 | 0.259536 | 9.383213 |
| 1M | lfm | 10 | X | X | X | X | X | X | X | X | X | X | X | X | X | X |
| all(8M) | lfm | 10 | X | X | X | X | X | X | X | X | X | X | X | X | X | X |

with the parameters :
```python
model1 = LightFM(loss='warp', no_components=60,learning_rate = 0.01,random_state=np.random.RandomState(42))
```

## 1. Hybrid matrix factorisation model

This notebook explains the concept of a hybrid matrix factorisation based model for recommendation, it also outlines the steps to construct a pure matrix factorisation and a hybrid models using the [LightFM](https://github.com/lyst/lightfm) package. It also demonstrates how to extract both user and item affinity from a fitted hybrid model.

### 1.1 Background

In general, most recommendation models can be divided into two categories:
- Content based model,
- Collaborative filtering model.

The content-based model recommends based on similarity of the items and/or users using their description/metadata/profile. On the other hand, collaborative filtering model (discussion is limited to matrix factorisation approach in this notebook) computes the latent factors of the users and items. It works based on the assumption that if a group of people expressed similar opinions on an item, these peole would tend to have similar opinions on other items. For further background and detailed explanation between these two approaches, the reader can refer to machine learning literatures [3, 4].

The choice between the two models is largely based on the data availability. For example, the collaborative filtering model is usually adopted and effective when sufficient ratings/feedbacks have been recorded for a group of users and items.

However, if there is a lack of ratings, content based model can be used provided that the metadata of the users and items are available. This is also the common approach to address the cold-start issues, where there are insufficient historical collaborative interactions available to model new users and/or items.

<!-- In addition, most collaborative filtering models only consume explicit ratings e.g. movie 

**NOTE** add stuff about implicit and explicit ratings -->

### 1.2 Hybrid matrix factorisation algorithm

In view of the above problems, there have been a number of proposals to address the cold-start issues by combining both content-based and collaborative filtering approaches. The hybrid matrix factorisation model is among one of the solutions proposed [1].  

In general, most hybrid approaches proposed different ways of assessing and/or combining the feature data in conjunction with the collaborative information.

### 1.3 LightFM package 

LightFM is a Python implementation of a hybrid recommendation algorithms for both implicit and explicit feedbacks [1].

It is a hybrid content-collaborative model which represents users and items as linear combinations of their content features’ latent factors. The model learns **embeddings or latent representations of the users and items in such a way that it encodes user preferences over items**. These representations produce scores for every item for a given user; items scored highly are more likely to be interesting to the user.

The user and item embeddings are estimated for every feature, and these features are then added together to be the final representations for users and items. 

For example, for user i, the model retrieves the i-th row of the feature matrix to find the features with non-zero weights. The embeddings for these features will then be added together to become the user representation e.g. if user 10 has weight 1 in the 5th column of the user feature matrix, and weight 3 in the 20th column, the user 10’s representation is the sum of embedding for the 5th and the 20th features multiplying their corresponding weights. The representation for each items is computed in the same approach. 

#### 1.3.1 Modelling approach

Let $`U`$ be the set of users and $`I`$ be the set of items, and each user can be described by a set of user features $`f_{u} \subset F^{U}`$ whilst each items can be described by item features $`f_{i} \subset F^{I}`$. Both $`F^{U}`$ and $`F^{I}`$ are all the features which fully describe all users and items. 

The LightFM model operates based binary feedbacks, the ratings will be normalised into two groups. The user-item interaction pairs $`(u,i) \in U\times I`$ are the union of positive (favourable reviews) $`S^+`$ and negative interactions (negative reviews) $`S^-`$ for explicit ratings. For implicit feedbacks, these can be the observed and not observed interactions respectively.

For each user and item feature, their embeddings are $`e_{f}^{U}`$ and $`e_{f}^{I}`$ respectively. Furthermore, each feature is also has a scalar bias term ($`b_U^f`$ for user and $`b_I^f`$ for item features). The embedding (latent representation) of user $`u`$ and item $`i`$ are the sum of its respective features’ latent vectors:

```math
q_{u} = \sum_{j \in f_{u}} e_{j}^{U}
```

```math
p_{i} = \sum_{j \in f_{i}} e_{j}^{I}
```

Similarly the biases for user $`u`$ and item $`i`$ are the sum of its respective bias vectors. These variables capture the variation in behaviour across users and items:

```math
b_{u} = \sum_{j \in f_{u}} b_{j}^{U}
```

```math
b_{i} = \sum_{j \in f_{i}} b_{j}^{I}
```

In LightFM, the representation for each user/item is a linear weighted sum of its feature vectors.

The prediction for user $`u`$ and item $`i`$ can be modelled as sigmoid of the dot product of user and item vectors, adjusted by its feature biases as follows:

```math
\hat{r}_{ui} = \sigma (q_{u} \cdot p_{i} + b_{u} + b_{i})
```

As the LightFM is constructed to predict binary outcomes e.g. $`S^+`$ and $`S^-`$, the function $`\sigma()`$ is based on the [sigmoid function](https://mathworld.wolfram.com/SigmoidFunction.html). 

The LightFM algorithm estimates interaction latent vectors and bias for features. For model fitting, the cost function of the model consists of maximising the likelihood of data conditional on the parameters described above using stochastic gradient descent. The likelihood can be expressed as follows:

```math
L = \prod_{(u,i) \in S+}\hat{r}_{ui} \times \prod_{(u,i) \in S-}1 - \hat{r}_{ui}
```

Note that if the feature latent vectors are not available, the algorithm will behaves like a [logistic matrix factorisation model](http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf).