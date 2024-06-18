# %%
#!sudo apt install -y curl unzip

# %% [markdown]
# # 오픈소스SW개론 과제 2

# %%
%conda install numpy pandas matplotlib scikit-learn scipy -y

# %% [markdown]
# ## 1. 데이터 준비
# 

# %% [markdown]
# ### 1.1. ml-1m.zip 다운로드
# 

# %%
!curl -SLJ https://files.grouplens.org/datasets/movielens/ml-1m.zip --output /tmp/ml-1m.zip

# %% [markdown]
# ### 1.2. ml-1m.zip 압축 해제
# 

# %%
!unzip -o /tmp/ml-1m.zip ml-1m/ratings.dat -d /tmp

# %%
!ls -l /tmp/ml-1m

# %% [markdown]
# ## 2. Group Recommender System 구현

# %% [markdown]
# ### 2.1 Clustering

# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ratings = pd.read_csv(
    "/tmp/ml-1m/ratings.dat",
    sep="::",
    header=None,
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
)

user_item_matrix = (
    ratings.pivot(index="UserID", columns="MovieID", values="Rating")
    .fillna(0)
    .values
)


def rank_with_argsort(arr):
    order = arr.argsort()
    ranks = order.argsort() + 1  # 1부터 시작하도록 +1
    return ranks


# %% [markdown]
# #### 2.1.1. Matrix 생성
# 

# %%
kmeans = KMeans(n_clusters=3, random_state=1)
user_groups = kmeans.fit_predict(user_item_matrix)

# %% [markdown]
# ### 2.2. Aggregation

# %%
def aggregation(user_item_matrix, user_groups, aggregation_func):
    for group_id in set(user_groups):
        group_ratings = user_item_matrix[user_groups == group_id]
        recommendations = aggregation_func(group_ratings)
        top_10_recommendations = np.argsort(recommendations)[-10:]
        print(top_10_recommendations)

# %% [markdown]
# #### 2.2.1. Additive Utilitarian

# %%
def additive_utilitarian(group_ratings):
    return np.sum(group_ratings, axis=0)

aggregation(user_item_matrix, user_groups, additive_utilitarian)


# %% [markdown]
# #### 2.2.2. Average

# %%
def average(group_ratings):
    return np.mean(group_ratings, axis=0)

aggregation(user_item_matrix, user_groups, average)

# %% [markdown]
# #### 2.2.3. Simple Count

# %%
def simple_count(group_ratings):
    return np.count_nonzero(group_ratings, axis=0)

aggregation(user_item_matrix, user_groups, simple_count)

# %% [markdown]
# #### 2.2.4. Approval Voting

# %%
def approval_voting(group_ratings):
    return np.sum(group_ratings >= 4, axis=0)

aggregation(user_item_matrix, user_groups, approval_voting)

# %% [markdown]
# #### 2.2.5. Borda Count

# %%
from scipy.stats import rankdata


def board_approval(group_ratings):
    return np.sum(
        np.apply_along_axis(
            lambda x: rankdata(x, method="average"),
            axis=1,
            arr=group_ratings,
        ),
        axis=0,
    )


aggregation(user_item_matrix, user_groups, board_approval)

# %% [markdown]
# #### 2.2.6. Copeland Rule

# %%
def copeland_rule(group_ratings):
    result = np.zeros(group_ratings.shape)
    matrix = np.array(group_ratings)
    
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            result[row, column] = np.sum(matrix[row] > matrix[row, column]) - np.sum(matrix[row] < matrix[row, column])

    return result.sum(axis=0)

aggregation(user_item_matrix, user_groups, copeland_rule)


