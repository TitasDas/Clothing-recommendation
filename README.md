## (Not maintained - setting up a more recent clothing recommender soon)


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#data">Data</a></li>
    <li><a href="#current-product-recommendation-engine ">Current product recommendation engine</a></li>
    <li><a href="#current-usage-of-the-engine">Current usage of the engine</a></li>
    <li><a href="#what-can-be-improved"> What can be improved ?</a></li>
    <li><a href="#ideas">Ideas</a></li>
    <li><a href="#bibliography">Bibliography</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Product recommendation engine

This project is about creating a product recommendation engine that will be used by Cerebra's customers. The main goal is to recommend products to customers based on what they bought. Another goal will be recommending similar items for each item. The  evaluation of our recommendation system will be based on rating metrics as well as ranking metrics. However, we will also introduce diversity metrics like diversity, novelty and value for money.

### Built With

* [python3](https://www.python.org/)

<!-- GETTING STARTED -->
## Getting Started
### Installation

1. Create a new virtual environment with python == 3.7
2. Install requirement.txt
  ```sh
  pip install -r requirements.txt
  ```

You can now run all the notebooks in the [notebook](notebook) section. 
You should **first** run the notebook [get_data.ipynb](notebook/data/get_data.ipynb) in order to get the data that will be used by the models.

<!-- Data -->
## Data 

For each customer, we have access to 2 useful dataframes for this specific task : 
- Transaction dataframe (which customer has bought which item)
- Item's stock dataframe (in order to know if the item is available or not)

A short EDA has been done on these dataframes in a jupyter notebook available at this [link](notebook/exploratory_data_analysis/transactions_df_eda.ipynb).

Get those data [here](notebook/data/get_data.ipynb).

<!-- Current product recommendation engine -->
## Current product recommendation engine 
The actual engine is based on a collaborative filtering method : matrix factorization using SVD with 3 components. The utility matrix is filled by the quantity the customer has bought for each item. 

To know more about this model, go [there](models/t_svd).

<!-- Current usage of the engine -->
## Current usage of the engine

Uniqlo is the customer who has requested this module. 
They are conducting A/B testing by sending product recommendation to customer by email.

<!-- What can be improved ? -->
## What can be improved ? 
- Data : Currently, we are utilizing only data from the past 4 months because the function pivot_table do not work when the matrix is huge (more than 1 billion rows). 
- Runtime : Creating the utility matrix and doing the SVD takes most of the time. 
- The evaluation metric : We do not evaluate the engine
- The objectives : No specific objectives.
- Features : We only use the quantity the customer has bought for each item.
- The model : matrix factorization with SVD

<!-- Ideas -->
## Ideas 
- Data : Work with sparse matrix or make the function pivot_table work (possible based on my research)
- Runtime : Work with sparse matrix
- The evaluation metric : Create a  test set and introduce classic metrics
- Features : Add selling price and other available features.
- The objectives : 
    - Add new features and modify the evaluation metric. 
    - Custom similarity function when we are suggesting similar items. 
    - Scoring model: score the generated candidates based on an other model. 
    - Re-ranking : filtering items at the end.
- The model : matrix factorization with SDG, SDV++

<!-- Bibliography -->
## Bibliography 

Google tutorial on recommendation system with a colab notebook : https://developers.google.com/machine-learning/recommendation

(2014) Video about recommendation system by Xavier Amatriain, former reseacher at Netflix
https://www.youtube.com/watch?v=mRToFXlNBpQ

(2021) Video about Trends in Recommendation & Personalization at Netflix by Justin Basilico Director, ML & Recommendations Engineering.
https://exchange.scale.com/public/videos/trends-in-recommendation-and-personalization-at-netflix

Microsoft github repo about recommendation systems : https://github.com/microsoft/recommenders
