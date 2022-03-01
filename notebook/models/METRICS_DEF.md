# Metrics definition 

Several approaches for evaluating model performance are demonstrated along with their respective metrics.

1. Rating Metrics: These are used to evaluate how accurate a recommender is at predicting ratings that users gave to items
    * Root Mean Square Error (RMSE) - measure of average error in predicted ratings
    * Mean Absolute Error (MAE) - similar to RMSE but uses absolute value instead of squaring and taking the root of the average
2. Ranking Metrics: These are used to evaluate how relevant recommendations are for users
    * Precision - this measures the proportion of recommended items that are relevant
    * Recall - this measures the proportion of relevant items that are recommended
    * Normalized Discounted Cumulative Gain (NDCG) - evaluates how well the predicted items for a user are ranked based on relevance
    * Mean Average Precision (MAP) - average precision for each user normalized over all users
4. Non accuracy based metrics: These do not compare predictions against ground truth but instead evaluate the following properties of the recommendations
    * Novelty - measures of how novel recommendation items are by calculating their recommendation frequency among users 
    * Diversity - measures of how different items in a set are with respect to each other
    * Coverage - measures related to the distribution of items recommended by the system. 


## Summary

|Metric|Range|Selection criteria|Limitation|Reference|
|------|-------------------------------|---------|----------|---------|
|RMSE|$`> 0`$|The smaller the better.|May be biased, and less explainable than MSE|[link](https://en.wikipedia.org/wiki/Root-mean-square_deviation)|
|MAE|$`\geq 0`$|The smaller the better.|Dependent on variable scale.|[link](https://en.wikipedia.org/wiki/Mean_absolute_error)|
|Precision|$`\geq 0`$ and $`\leq 1`$|The closer to $`1`$ the better.|Only for hits in recommendations.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|Recall|$`\geq 0`$ and $`\leq 1`$|The closer to $`1`$ the better.|Only for hits in the ground truth.|[link](https://en.wikipedia.org/wiki/Precision_and_recall)|
|NDCG|$`\geq 0`$ and $`\leq 1`$|The closer to $`1`$ the better.|Does not penalize for bad/missing items, and does not perform for several equally good items.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|MAP|$`\geq 0`$ and $`\leq 1`$|The closer to $`1`$ the better.|Depend on variable distributions.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|Catalog coverage|$`\geq 0`$|The higher the better.||See section Coverage|
|Distributional coverage|$`\geq 0`$|The higher the better.||See Coverage section|
|Diversity|$`\geq 0`$|The higher the better.||See Diversity section|
|Novelty|$`\geq 0`$|The higher the better.||See Novelty section|

# Diversity Metrics

**Coverage**

We define _catalog coverage_ as the proportion of items showing in all users’ recommendations: 
```math
\textrm{CatalogCoverage} = \frac{|N_r|}{|N_t|}
```
where $`N_r`$ denotes the set of items in the recommendations (`reco_df` in the code below) and $`N_t`$ the set of items in the historical data (`train_df`).

_Distributional coverage_ measures how equally different items are recommended to users when a particular recommender system is used.
If  $`p(i|R)`$ denotes the probability that item $`i`$ is observed among all recommendation lists, we define distributional coverage as
```math
\textrm{DistributionalCoverage} = -\sum_{i \in N_t} p(i|R) \log_2 p(i)
```
where 
```math 
p(i|R) = \frac{|M_r (i)|}{|\textrm{reco\_df}|} 
```
and $`M_r (i)`$ denotes the users who are recommended item $`i`$.


**Diversity**

Diversity represents the variety present in a list of recommendations.
_Intra-List Similarity_ aggregates the pairwise similarity of all items in a set. A recommendation list with groups of very similar items will score a high intra-list similarity. Lower intra-list similarity indicates higher diversity.
To measure similarity between any two items we use cosine similarity :
```math
\textrm{Cosine Similarity}(i,j) =  \frac{|M_t^{l(i,j)}|} {\sqrt{|M_t^{l(i)}|} \sqrt{|M_t^{l(j)}|} }
```
where $`M_t^{l(i)}`$ denotes the set of users who liked item $`i`$ and $`M_t^{l(i,j)}`$ the users who liked both $`i`$ and $`j`$.
Intra-list similarity is then defined as 
```math
\textrm{IL} = \frac{1}{|M|} \sum_{u \in M} \frac{1}{\binom{N_r(u)}{2}} \sum_{i,j \in N_r (u),\, i<j} \textrm{Cosine Similarity}(i,j)
```
where $`M`$ is the set of users and $`N_r(u)`$ the set of recommendations for user $`u`$. Finally, diversity is defined as
```math
\textrm{diversity} = 1 - \textrm{IL}
```

**Novelty**

The novelty of an item is inverse to its _popularity_. If $`p(i)`$ represents the probability that item $`i`$ is observed (or known, interacted with etc.) by users, then  
```math
p(i) = \frac{|M_t (i)|} {|\textrm{train\_df}|}
```
where $`M_t (i)`$ is the set of users who have interacted with item $`i`$ in the historical data. 

The novelty of an item is then defined as
```math
\textrm{novelty}(i) = -\log_2 p(i)
```
and the novelty of the recommendations across all users is defined as
```math
\textrm{novelty} = \sum_{i \in N_r} \frac{|M_r (i)|}{|\textrm{reco\_df}|} \textrm{novelty}(i)
```


### References
The metric definitions / formulations are based on the following references:
- P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems: choice, discovery and relevance, ECIR 2011
- G. Shani and A. Gunawardana, Evaluating recommendation systems, Recommender Systems Handbook pp. 257-297, 2010.
- E. Yan, Serendipity: Accuracy’s unpopular best friend in recommender Systems, eugeneyan.com, April 2020
- Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist: introducing serendipity into music recommendation, WSDM 2012

