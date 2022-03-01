## Results 

| Data | Algo | K | Train time (s) | Predicting time (s) | Recommending time (s) | mAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R2 | Diversity | Novelty | Catalog coverage | Distributional coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | sar | 10 | 1.869315 | nan | 0.412482 | 0.015832 | 0.120409 | 0.11125 | 0.031310 | nan | nan | nan | 0.760300 | 11.058183 | 0.274308 | 8.97861 |
| 1M | sar | 10 | 4.480668 | nan | 9.005530 | 0.027378 | 0.074688 | 0.06103 | 0.059482 | nan | nan | nan | 0.904983 | 11.574736 | 0.774543 | 10.55634 |
| all(8M) | sar | 10 | 18.09361 | nan | 211.60146 | 0.100136 | 0.138245 | 0.034234 | 0.21693 | nan | nan | nan | 0.957637 | 11.336318 | 0.957958 | 11.040913 |

with the hyperparameters of the model: 
- similarity_type="jaccard"
- time_decay_coefficient=15
- timedecay_formula=True

## SAR algorithm

The following figure presents a high-level architecture of SAR. 

At a very high level, two intermediate matrices are created and used to generate a set of recommendation scores:

- An item similarity matrix $`S`$ estimates item-item relationships.
- An affinity matrix $`A`$ estimates user-item relationships.

Recommendation scores are then created by computing the matrix multiplication $`A\times S`$.

Optional steps (e.g. "time decay" and "remove seen items") are described in the details below.

<img src="https://recodatasets.z20.web.core.windows.net/images/sar_schema.svg?sanitize=true">

### 1.1 Compute item co-occurrence and item similarity

SAR defines similarity based on item-to-item co-occurrence data. Co-occurrence is defined as the number of times two items appear together for a given user. We can represent the co-occurrence of all items as a $`m\times m`$ matrix $`C`$, where $`c_{i,j}`$ is the number of times item $`i`$ occurred with item $`j`$, and $`m`$ is the total number of items.

The co-occurence matric $`C`$ has the following properties:

- It is symmetric, so $`c_{i,j} = c_{j,i}`$
- It is nonnegative: $`c_{i,j} \geq 0`$
- The occurrences are at least as large as the co-occurrences. I.e., the largest element for each row (and column) is on the main diagonal: $`\forall(i,j) C_{i,i},C_{j,j} \geq C_{i,j}`$.

Once we have a co-occurrence matrix, an item similarity matrix $`S`$ can be obtained by rescaling the co-occurrences according to a given metric. Options for the metric include `Jaccard`, `lift`, and `counts` (meaning no rescaling).


If $`c_{ii}`$ and $`c_{jj}`$ are the $`i`$th and $`j`$th diagonal elements of $`C`$, the rescaling options are:

- `Jaccard`: $`s_{ij}=\frac{c_{ij}}{(c_{ii}+c_{jj}-c_{ij})}`$
- `lift`: $`s_{ij}=\frac{c_{ij}}{(c_{ii} \times c_{jj})}`$
- `counts`: $`s_{ij}=c_{ij}`$

In general, using `counts` as a similarity metric favours predictability, meaning that the most popular items will be recommended most of the time. `lift` by contrast favours discoverability/serendipity: an item that is less popular overall but highly favoured by a small subset of users is more likely to be recommended. `Jaccard` is a compromise between the two.


### 1.2 Compute user affinity scores

The affinity matrix in SAR captures the strength of the relationship between each individual user and the items that user has already interacted with. SAR incorporates two factors that can impact users' affinities: 

- It can consider information about the **type** of user-item interaction through differential weighting of different events (e.g. it may weigh events in which a user rated a particular item more heavily than events in which a user viewed the item).
- It can consider information about **when** a user-item event occurred (e.g. it may discount the value of events that take place in the distant past.

Formalizing these factors produces us an expression for user-item affinity:

```math
a_{ij}=\sum_k w_k \left(\frac{1}{2}\right)^{\frac{t_0-t_k}{T}} 
```

where the affinity $`a_{ij}`$ for user $`i`$ and item $`j`$ is the weighted sum of all $`k`$ events involving user $`i`$ and item $`j`$. $`w_k`$ represents the weight of a particular event, and the power of 2 term reflects the temporally-discounted event. The $`(\frac{1}{2})^n`$ scaling factor causes the parameter $`T`$ to serve as a half-life: events $`T`$ units before $`t_0`$ will be given half the weight as those taking place at $`t_0`$.

Repeating this computation for all $`n`$ users and $`m`$ items results in an $`n\times m`$ matrix $`A`$. Simplifications of the above expression can be obtained by setting all the weights equal to 1 (effectively ignoring event types), or by setting the half-life parameter $`T`$ to infinity (ignoring transaction times).

### 1.3 Remove seen item

Optionally we remove items which have already been seen in the training set, i.e. don't recommend items which have been previously bought by the user again.

### 1.4 Top-k item calculation

The personalized recommendations for a set of users can then be obtained by multiplying the affinity matrix ($`A`$) by the similarity matrix ($`S`$). The result is a recommendation score matrix, where each row corresponds to a user, each column corresponds to an item, and each entry corresponds to a user / item pair. Higher scores correspond to more strongly recommended items.

It is worth noting that the complexity of recommending operation depends on the data size. SAR algorithm itself has $`O(n^3)`$ complexity. Therefore the single-node implementation is not supposed to handle large dataset in a scalable manner. Whenever one uses the algorithm, it is recommended to run with sufficiently large memory. 
